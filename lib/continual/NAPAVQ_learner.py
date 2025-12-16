import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import numpy as np

import os
import copy

from lib.continual.NAVQ import NAVQ
from lib.model import resnet_model, CrossEntropy
from lib.utils import get_optimizer, get_scheduler, AverageMeter


def _get_dist_each_class(feature, navq):
    features = feature.unsqueeze(1)
    cvs = navq.cvs.unsqueeze(0).repeat(feature.size(0), 1, 1)
    dist = torch.cdist(features, cvs).squeeze(1)

    return -dist


class NAPAVQ_learner:
    def __init__(self, cfg, dataset_handler, logger):
        self.cfg = cfg
        self.dataset_handler = dataset_handler
        self.logger = logger

        self.gpus = torch.cuda.device_count()
        self.device_ids = [] if cfg.availabel_cudas == "" else \
            [i for i in range(len(self.cfg.availabel_cudas.strip().split(',')))]
        self.device = torch.device("cuda")

        self.model = None
        self.old_model = None
        self.navq = None

        self.train_init()

        self.prototype_dict = {}

        self.best_model = None
        self.best_navq = None
        self.best_epoch = None
        self.best_acc = 0.

    def train_init(self):
        self.model = self.construct_model()
        # self.generator_optimizer = Adam(params=self.generator.parameters(), lr=self.deep_inv_params[0])
        if self.cfg.PRETRAINED.MODEL:
            self.model.load_model(self.cfg.PRETRAINED.MODEL)
        self.model = self.transfer_to_cuda(self.model)

    def transfer_to_cuda(self, model):
        if self.gpus > 1:
            if len(self.device_ids) > 1:
                model = torch.nn.DataParallel(model, device_ids=self.device_ids).cuda()
            else:
                model = torch.nn.DataParallel(model).cuda()
        else:
            model = model.to("cuda")
        return model

    def print_model(self):
        print("Network:", self.model)
        print("navq:", self.navq)
        pass

    def construct_model(self):
        model = resnet_model(self.cfg)
        return model
        pass

    def construct_navq(self, active_num_class):
        navq = NAVQ(
            num_classes=active_num_class * 4,
            feat_dim=self.cfg.extractor.output_feature_dim,
            device=self.device,
        )
        return navq
        pass

    def load_napavq(self, model_path, navq_path, train_dataset):
        self.model.load_model(model_path)
        self.navq = torch.load(navq_path)
        self.model = self.transfer_to_cuda(self.model)
        self.navq = self.transfer_to_cuda(self.navq)
        self.after_train(train_dataset)
        pass

    def build_optimize(self, model=None, base_lr=None, optimizer_type=None, momentum=None, weight_decay=None):
        # todo Done
        MODEL = model if model is not None else self.model
        optimizer = get_optimizer(MODEL, BASE_LR=base_lr, optimizer_type=optimizer_type, momentum=momentum,
                                  weight_decay=weight_decay)

        return optimizer

    def build_scheduler(self, optimizer, lr_type=None, step_size=None, lr_step=None, lr_factor=None, warmup_epochs=None,
                        MAX_EPOCH=200):
        # todo optimizer, lr_type=None, lr_step=None, lr_factor=None, warmup_epochs=None
        scheduler = get_scheduler(optimizer=optimizer, lr_type=lr_type, step_size=step_size, lr_step=lr_step,
                                  lr_factor=lr_factor, warmup_epochs=warmup_epochs, MAX_EPOCH=MAX_EPOCH)
        return scheduler

    def first_task_train_main(self, train_dataset, active_classes_num, task):
        self.before_train(active_classes_num=active_classes_num)
        dp_classes_num = 0
        optimizer = self.build_optimize(model=self.model,
                                        base_lr=self.cfg.model.TRAIN.OPTIMIZER.BASE_LR,
                                        optimizer_type=self.cfg.model.TRAIN.OPTIMIZER.TYPE,
                                        momentum=self.cfg.model.TRAIN.OPTIMIZER.MOMENTUM,
                                        weight_decay=self.cfg.model.TRAIN.OPTIMIZER.WEIGHT_DECAY)
        cvs_optimizer = self.build_optimize(model=self.navq,
                                            base_lr=self.cfg.model.navq.TRAIN.OPTIMIZER.BASE_LR,
                                            optimizer_type=self.cfg.model.navq.TRAIN.OPTIMIZER.TYPE,
                                            momentum=self.cfg.model.navq.TRAIN.OPTIMIZER.MOMENTUM,
                                            weight_decay=self.cfg.model.navq.TRAIN.OPTIMIZER.WEIGHT_DECAY)
        scheduler = self.build_scheduler(optimizer, lr_type=self.cfg.model.TRAIN.LR_SCHEDULER.TYPE,
                                         step_size=self.cfg.model.TRAIN.LR_SCHEDULER.step_size,
                                         lr_step=self.cfg.model.TRAIN.LR_SCHEDULER.LR_STEP,
                                         lr_factor=self.cfg.model.TRAIN.LR_SCHEDULER.LR_FACTOR,
                                         warmup_epochs=self.cfg.model.TRAIN.LR_SCHEDULER.WARM_EPOCH,
                                         MAX_EPOCH=self.cfg.model.TRAIN.MAX_EPOCH)
        cvs_scheduler = self.build_scheduler(cvs_optimizer, lr_type=self.cfg.model.navq.TRAIN.LR_SCHEDULER.TYPE,
                                             step_size=self.cfg.model.navq.TRAIN.LR_SCHEDULER.step_size,
                                             lr_step=self.cfg.model.navq.TRAIN.LR_SCHEDULER.LR_STEP,
                                             lr_factor=self.cfg.model.navq.TRAIN.LR_SCHEDULER.LR_FACTOR,
                                             warmup_epochs=self.cfg.model.navq.TRAIN.LR_SCHEDULER.WARM_EPOCH,
                                             MAX_EPOCH=self.cfg.model.TRAIN.MAX_EPOCH)
        best_acc = 0
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.cfg.model.TRAIN.BATCH_SIZE,
                                  num_workers=self.cfg.model.TRAIN.NUM_WORKERS, shuffle=True, drop_last=True,
                                  persistent_workers=True)
        iter_num = len(train_loader)
        assert self.navq.edges.shape[0] == self.navq.num_classes == active_classes_num * 4, f"self.navq.edges.shape[0]: " \
                                                                                            f"{self.navq.edges.shape[0]}"
        for epoch in range(1, self.cfg.model.TRAIN.MAX_EPOCH + 1):
            all_loss = AverageMeter()
            total_dce = AverageMeter()
            total_na = AverageMeter()
            if float(torch.__version__[:3]) < 1.3:
                scheduler.step()
                cvs_scheduler.step()
            iter_index = 0
            for x, y in train_loader:
                self.model.train()
                x, y = x.to(self.device), y.to(self.device)  # --> transfer them to correct device
                # self-supervised learning based label augmentation similar to PASS
                images = torch.stack([torch.rot90(x, k, (2, 3)) for k in range(4)], 1)
                images = images.view(-1, 3, self.cfg.INPUT_SIZE[0], self.cfg.INPUT_SIZE[1])
                target = torch.stack([y * 4 + k for k in range(4)], 1).view(-1)
                # ---> Train MAIN MODEL
                cnt = target.shape[0]

                loss_dce, loss_other, loss_na = self._compute_loss(images, target, old_class=dp_classes_num)
                loss = loss_dce + loss_other + loss_na

                optimizer.zero_grad()
                cvs_optimizer.zero_grad()
                loss.backward()

                self._before_update(old_classes_num=dp_classes_num)
                optimizer.step()
                cvs_optimizer.step()
                all_loss.update(loss.data.item(), cnt)
                total_dce.update(loss_dce.data.item(), cnt)
                total_na.update(loss_na.data.item(), cnt)
                if iter_index % self.cfg.SHOW_STEP == 0:
                    pbar_str = "Epoch: {} || Batch:{:>3d}/{}|| lr: {} || Batch_all_loss:{:>5.3f} " \
                               "|| Batch_dce_loss:{:>5.3f} || Batch_na_loss:{:>5.3f}".format(epoch, iter_index,
                                                                                             iter_num,
                                                                                             optimizer.param_groups[0][
                                                                                                 'lr'],
                                                                                             all_loss.val,
                                                                                             total_dce.val,
                                                                                             total_na.val
                                                                                             )
                    self.logger.info(pbar_str)
                iter_index += 1

            # if epoch % self.cfg.epoch_show_step == 0:
            if self.cfg.VALID_STEP != -1 and epoch % self.cfg.VALID_STEP == 0:
                pbar_str = "First task train, Validate Epoch: {} || lr: {} || epoch_Loss:{:>5.3f}".format(epoch,
                                                                                                          optimizer.param_groups[
                                                                                                              0]['lr'],
                                                                                                          all_loss.val)
                self.logger.info(pbar_str)

                val_acc = self.validate_with_navq(task=task)  # task_id 从1开始
                acc_avg = val_acc.mean()
                self.logger.info(
                    "--------------current_Epoch:{:>3d}    current_Acc:{:>5.2f}%--------------".format(
                        epoch, acc_avg * 100
                    )
                )
                if acc_avg > best_acc:
                    best_acc, best_epoch = acc_avg, epoch
                    self.best_model = copy.deepcopy(self.model)
                    self.best_navq = copy.deepcopy(self.navq)
                    self.best_epoch = best_epoch
                    self.best_acc = best_acc
                    self.logger.info(
                        "--------------Best_Epoch:{:>3d}    Best_Acc:{:>5.2f}%--------------".format(
                            best_epoch, best_acc * 100
                        )
                    )

            if float(torch.__version__[:3]) >= 1.3:
                scheduler.step()
                cvs_scheduler.step()
        del train_loader
        self.after_train(train_dataset)
        pass

    def learn_new_task(self, train_dataset, active_classes_num, task):
        self.before_train(train_dataset=train_dataset)
        dp_classes_num = active_classes_num - self.dataset_handler.classes_per_task
        optimizer = self.build_optimize(model=self.model,
                                        base_lr=self.cfg.model.TRAIN.OPTIMIZER.BASE_LR,
                                        optimizer_type=self.cfg.model.TRAIN.OPTIMIZER.TYPE,
                                        momentum=self.cfg.model.TRAIN.OPTIMIZER.MOMENTUM,
                                        weight_decay=self.cfg.model.TRAIN.OPTIMIZER.WEIGHT_DECAY)
        cvs_optimizer = self.build_optimize(model=self.navq,
                                            base_lr=self.cfg.model.navq.TRAIN.OPTIMIZER.BASE_LR,
                                            optimizer_type=self.cfg.model.navq.TRAIN.OPTIMIZER.TYPE,
                                            momentum=self.cfg.model.navq.TRAIN.OPTIMIZER.MOMENTUM,
                                            weight_decay=self.cfg.model.navq.TRAIN.OPTIMIZER.WEIGHT_DECAY)
        scheduler = self.build_scheduler(optimizer, lr_type=self.cfg.model.TRAIN.LR_SCHEDULER.TYPE,
                                         step_size=self.cfg.model.TRAIN.LR_SCHEDULER.step_size,
                                         lr_step=self.cfg.model.TRAIN.LR_SCHEDULER.LR_STEP,
                                         lr_factor=self.cfg.model.TRAIN.LR_SCHEDULER.LR_FACTOR,
                                         warmup_epochs=self.cfg.model.TRAIN.LR_SCHEDULER.WARM_EPOCH,
                                         MAX_EPOCH=self.cfg.model.TRAIN.MAX_EPOCH)
        cvs_scheduler = self.build_scheduler(cvs_optimizer, lr_type=self.cfg.model.navq.TRAIN.LR_SCHEDULER.TYPE,
                                             step_size=self.cfg.model.navq.TRAIN.LR_SCHEDULER.step_size,
                                             lr_step=self.cfg.model.navq.TRAIN.LR_SCHEDULER.LR_STEP,
                                             lr_factor=self.cfg.model.navq.TRAIN.LR_SCHEDULER.LR_FACTOR,
                                             warmup_epochs=self.cfg.model.navq.TRAIN.LR_SCHEDULER.WARM_EPOCH,
                                             MAX_EPOCH=self.cfg.model.TRAIN.MAX_EPOCH)
        best_acc = 0
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.cfg.model.TRAIN.BATCH_SIZE,
                                  num_workers=self.cfg.model.TRAIN.NUM_WORKERS, shuffle=True, drop_last=True,
                                  persistent_workers=True)
        iter_num = len(train_loader)
        assert self.navq.edges.shape[
                   1] == self.navq.num_classes == active_classes_num * 4, f"self.navq.edges.shape[0]: " \
                                                                          f"{self.navq.edges.shape[1]}"
        assert len(self.prototype_dict) == active_classes_num * 4, f"self.prototype_dict: " \
                                                                          f"{self.prototype_dict}"
        for epoch in range(1, self.cfg.model.TRAIN.MAX_EPOCH + 1):
            all_loss = AverageMeter()
            total_dce = AverageMeter()
            total_na = AverageMeter()
            if float(torch.__version__[:3]) < 1.3:
                scheduler.step()
                cvs_scheduler.step()
            iter_index = 0
            for x, y in train_loader:
                self.model.train()
                x, y = x.to(self.device), y.to(self.device)  # --> transfer them to correct device
                # self-supervised learning based label augmentation similar to PASS
                images = torch.stack([torch.rot90(x, k, (2, 3)) for k in range(4)], 1)
                images = images.view(-1, 3, self.cfg.INPUT_SIZE[0], self.cfg.INPUT_SIZE[1])
                target = torch.stack([y * 4 + k for k in range(4)], 1).view(-1)
                # ---> Train MAIN MODEL
                cnt = target.shape[0]

                loss_dce, loss_other, loss_na = self._compute_loss(images, target, old_class=dp_classes_num)
                loss = loss_dce + loss_other + loss_na

                optimizer.zero_grad()
                cvs_optimizer.zero_grad()
                loss.backward()

                self._before_update(old_classes_num=dp_classes_num)
                optimizer.step()
                cvs_optimizer.step()
                all_loss.update(loss.data.item(), cnt)
                total_dce.update(loss_dce.data.item(), cnt)
                total_na.update(loss_na.data.item(), cnt)
                if iter_index % self.cfg.SHOW_STEP == 0:
                    pbar_str = "Epoch: {} || Batch:{:>3d}/{}|| lr: {} || Batch_all_loss:{:>5.3f} " \
                               "|| Batch_dce_loss:{:>5.3f} || Batch_na_loss:{:>5.3f}".format(epoch, iter_index,
                                                                                             iter_num,
                                                                                             optimizer.param_groups[0][
                                                                                                 'lr'],
                                                                                             all_loss.val,
                                                                                             total_dce.val,
                                                                                             total_na.val
                                                                                             )
                    self.logger.info(pbar_str)
                iter_index += 1

            # if epoch % self.cfg.epoch_show_step == 0:
            if self.cfg.VALID_STEP != -1 and epoch % self.cfg.VALID_STEP == 0:
                pbar_str = "First task train, Validate Epoch: {} || lr: {} || epoch_Loss:{:>5.3f}".format(epoch,
                                                                                                          optimizer.param_groups[
                                                                                                              0]['lr'],
                                                                                                          all_loss.val)
                self.logger.info(pbar_str)

                val_acc = self.validate_with_navq(task=task)  # task_id 从1开始
                acc_avg = val_acc.mean()
                self.logger.info(
                    "--------------current_Epoch:{:>3d}    current_Acc:{:>5.2f}%--------------".format(
                        epoch, acc_avg * 100
                    )
                )
                if acc_avg > best_acc:
                    best_acc, best_epoch = acc_avg, epoch
                    self.best_model = copy.deepcopy(self.model)
                    self.best_navq = copy.deepcopy(self.navq)
                    self.best_epoch = best_epoch
                    self.best_acc = best_acc
                    self.logger.info(
                        "--------------Best_Epoch:{:>3d}    Best_Acc:{:>5.2f}%--------------".format(
                            best_epoch, best_acc * 100
                        )
                    )

            if float(torch.__version__[:3]) >= 1.3:
                scheduler.step()
                cvs_scheduler.step()
        del train_loader
        self.after_train(train_dataset)
        pass

    def _before_update(self, old_classes_num):
        # setting the gradients of old coding vectors to be 0
        if old_classes_num > 0:
            classes_old = range(old_classes_num * 4)
            self.navq.cvs.grad[classes_old, :] *= 0

    def before_train(self, active_classes_num=None, train_dataset=None):
        if self.old_model is None:
            assert active_classes_num is not None
            self.navq = NAVQ(
                num_classes=active_classes_num * 4,
                feat_dim=self.cfg.extractor.output_feature_dim,
                device=self.device)
            self.navq = self.transfer_to_cuda(self.navq)
        else:
            assert train_dataset is not None
            self.proto_save(self.old_model, train_dataset)
            self.navq.add_cvs(self.dataset_handler.classes_per_task * 4)

    def _compute_loss(self, imgs, target, old_class=0):

        feature = self.model(imgs, train_extractor=True)
        output = _get_dist_each_class(feature, self.navq)

        loss_dce = nn.CrossEntropyLoss()(output / self.cfg.model.temp, target)
        loss_na = self.navq(feature, target)

        if self.old_model is None:
            return loss_dce, 0, loss_na
        else:
            feature_old = self.old_model(imgs, is_nograd=True, feature_flag=True)
            loss_kd = torch.dist(feature, feature_old, 2)

            index = np.arange(old_class)

            # code for NA-PA
            random_indices = index[np.random.choice(len(index), size=self.cfg.model.TRAIN.BATCH_SIZE, replace=True)] * 4
            proto_list = [self.prototype_dict[i] for i in random_indices]
            proto_array = np.array(proto_list)
            proto_neighbours = self.navq.edges.cpu().numpy()[[random_indices]][0]
            picked_neighbour_indices = np.array([np.random.choice(r.nonzero()[0]) for r in proto_neighbours])
            picked_neighbours = np.array([self.prototype_dict[i] for i in picked_neighbour_indices])
            gammas = np.random.uniform(0.5, 1, self.cfg.model.TRAIN.BATCH_SIZE)
            proto_aug = proto_array * gammas[:, None] + picked_neighbours * (1 - gammas)[:, None]

            proto_aug = torch.tensor(proto_aug, dtype=torch.float).to(self.device)
            proto_aug_label = torch.from_numpy(random_indices).to(self.device)

            loss_na += self.navq(proto_aug, proto_aug_label)

            soft_feat_aug = _get_dist_each_class(proto_aug, self.navq)
            loss_protoAug = nn.CrossEntropyLoss()(soft_feat_aug / self.cfg.model.temp, proto_aug_label)

            return loss_dce, self.cfg.model.protoAug_weight * loss_protoAug + self.cfg.model.kd_weight * loss_kd, loss_na

    def after_train(self, train_dataset):
        self.old_model = copy.deepcopy(self.model).eval()
        self.proto_save(self.old_model, train_dataset)
        self.navq.old_class_indices = copy.deepcopy(self.navq.class_indices)

    def proto_save(self, model, train_dataset):
        features = []
        labels = []
        model.eval()
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.cfg.model.TRAIN.BATCH_SIZE,
                                  num_workers=self.cfg.model.TRAIN.NUM_WORKERS, shuffle=False, drop_last=False,
                                  persistent_workers=True)
        with torch.no_grad():
            for i, (images, target) in enumerate(train_loader):
                images, target = images.to(self.device), target.to(self.device)
                images = torch.stack([torch.rot90(images, k, (2, 3)) for k in range(4)], 1)
                images = images.view(-1, 3, 32, 32)
                target = torch.stack([target * 4 + k for k in range(4)], 1).view(-1)

                feature = model(images, is_nograd=True, feature_flag=True)
                # assert feature.shape[0] == self.cfg.model.TRAIN.BATCH_SIZE * 4, f"feature.shape[0]: {feature.shape[0]}"
                if feature.shape[0] == self.cfg.model.TRAIN.BATCH_SIZE * 4:
                    labels.append(target.cpu().numpy())
                    features.append(feature.cpu().numpy())
        labels_set = np.unique(labels)
        labels = np.array(labels)
        labels = np.reshape(labels, labels.shape[0] * labels.shape[1])
        features = np.array(features)
        features = np.reshape(features, (features.shape[0] * features.shape[1], features.shape[2]))

        for item in labels_set:
            index = np.where(item == labels)[0]
            feature_classwise = features[index]
            self.prototype_dict[item] = np.mean(feature_classwise, axis=0)

    def validate_with_navq(self, task, test_model=None, is_test=False):
        acc = []
        model = test_model if test_model is not None else self.model
        mode = model.training
        model.eval()
        for task_id in range(task):  # 这里的task 从0 开始
            if self.dataset_handler.val_datasets and (not is_test):
                predict_result = self.validate_with_navq_per_task(model, self.dataset_handler.val_datasets[task_id],
                                                                  task)
            else:
                predict_result = self.validate_with_navq_per_task(model, self.dataset_handler.test_datasets[task_id],
                                                                  task)
            acc.append(predict_result)
            self.logger.info(f"task: {task} || per task {task_id}, validate_with_FC acc:{predict_result}")
        acc = np.array(acc)
        model.train(mode=mode)
        return acc
        pass

    def validate_with_navq_per_task(self, model, val_dataset, task):
        # todo
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.cfg.model.TRAIN.BATCH_SIZE,
                                num_workers=self.cfg.model.TRAIN.NUM_WORKERS, shuffle=False, drop_last=False)
        top1 = AverageMeter()
        correct = 0
        active_classes_num = self.dataset_handler.classes_per_task * task
        for inputs, labels in val_loader:
            correct_temp = 0
            inputs, labels = inputs.to("cuda"), labels.to("cuda")
            features = model(x=inputs, is_nograd=True, feature_flag=True)
            features_norm = (features.T / torch.norm(features.T, dim=0)).T

            cvs_copy = self.navq.cvs.detach().clone()[::4, :]
            cvs_norm = (cvs_copy.T / torch.norm(cvs_copy.T, dim=0)).T.to(self.device)

            sqd = torch.cdist(cvs_norm, features_norm)
            predicts_ncm = torch.argmax((-sqd).T, dim=1)

            # cvs_copy = self.navq.cvs.detach().clone()
            # cvs_norm = (cvs_copy.T / torch.norm(cvs_copy.T, dim=0)).T.to(self.device)
            #
            # selected_class_indices = [self.navq.class_indices[i * 4] for i in range(0, active_classes_num)]
            # selected_class_indices_flat = [item for sublist in selected_class_indices for item in sublist]
            #
            # filtered_cvs = torch.index_select(cvs_norm, 0,
            #                                   torch.tensor(selected_class_indices_flat).to(self.device))
            # filtered_targets = [self.navq.cv_class[i] for i in selected_class_indices_flat]
            #
            # result = []
            # for target in features_norm.cpu().numpy():
            #     x = target - filtered_cvs.cpu().numpy()
            #     x = np.linalg.norm(x, ord=2, axis=1)
            #     x = np.argmin(x)
            #     result.append(filtered_targets[x] // 4)
            #
            # predicts_ncm = torch.tensor(result)
            correct_temp += (predicts_ncm.cpu() == labels.cpu()).sum()
            correct += correct_temp
            top1.update((correct_temp / inputs.size(0)).item(), inputs.size(0))
        return top1.avg

    def validate_with_navq_taskIL(self, task, is_test=False):
        acc = []
        mode = self.model.training
        self.model.eval()
        for task_id in range(task):  # 这里的task 从0 开始
            if self.dataset_handler.val_datasets and (not is_test):
                predict_result = self.validate_with_navq_per_task_taskIL(self.dataset_handler.val_datasets[task_id],
                                                                         task_id)
            else:
                predict_result = self.validate_with_navq_per_task_taskIL(self.dataset_handler.test_datasets[task_id],
                                                                         task_id)
            acc.append(predict_result)
            self.logger.info(
                f"task: {task} || per task {task_id}, validate_with_FC acc:{predict_result}"
            )
        acc = np.array(acc)
        self.model.train(mode=mode)
        return acc
        pass

    def validate_with_navq_per_task_taskIL(self, val_dataset, task_id):
        # todo
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.cfg.model.TRAIN.BATCH_SIZE,
                                num_workers=self.cfg.model.TRAIN.NUM_WORKERS, shuffle=False, drop_last=False)
        top1 = AverageMeter()
        correct = 0
        allowed_classes = [i for i in range(task_id * self.dataset_handler.classes_per_task,
                                            (task_id + 1) * self.dataset_handler.classes_per_task)]
        for inputs, labels in val_loader:
            correct_temp = 0
            inputs, labels = inputs.to("cuda"), labels.to("cuda")
            features = self.model(x=inputs, is_nograd=True, feature_flag=True)

            features_norm = (features.T / torch.norm(features.T, dim=0)).T

            cvs_copy = self.navq.cvs.detach().clone()
            cvs_norm = (cvs_copy.T / torch.norm(cvs_copy.T, dim=0)).T.to(self.device)

            selected_class_indices = [self.navq.class_indices[i * 4] for i in allowed_classes]
            selected_class_indices_flat = [item for sublist in selected_class_indices for item in sublist]

            filtered_cvs = torch.index_select(cvs_norm, 0,
                                              torch.tensor(selected_class_indices_flat).to(self.device))
            filtered_targets = [self.navq.cv_class[i] for i in selected_class_indices_flat]

            result = []
            for target in features_norm.cpu().numpy():
                x = target - filtered_cvs.cpu().numpy()
                x = np.linalg.norm(x, ord=2, axis=1)
                x = np.argmin(x)
                result.append(filtered_targets[x] // 4)

            predicts_ncm = torch.tensor(result)
            correct_temp += (predicts_ncm.cpu() == labels.cpu()).sum()
            correct += correct_temp
            top1.update((correct_temp / inputs.size(0)).item(), inputs.size(0))
        return top1.avg
        pass

    def save_best_latest_model_data(self, model_dir, task_id, acc, epoch):
        if self.best_model is None:
            self.best_model = self.model
            self.best_navq = self.navq
        self.latest_model = self.model
        self.latest_navq = self.navq
        split_selected_data = self.dataset_handler.get_split_selected_data()
        if task_id == 1 or self.cfg.use_base_half and task_id == int(self.dataset_handler.all_tasks / 2):
            torch.save({
                'state_dict': self.best_model.state_dict(),
                'acc_result': self.best_acc,
                'best_epoch': self.best_epoch,
                'task_id': task_id
            }, os.path.join(model_dir, "base_best_model.pth")
            )
            torch.save({
                'state_dict': self.latest_model.state_dict(),
                'acc_result': acc,
                'latest_epoch': epoch,
                'task_id': task_id,
                'split_selected_data': split_selected_data
            }, os.path.join(model_dir, "base_latest_model.pth")
            )
            torch.save(self.best_navq, os.path.join(model_dir, "base_best_navq.pth"))
            torch.save(self.latest_navq, os.path.join(model_dir, "base_latest_navq.pth"))
        else:
            torch.save({
                'state_dict': self.best_model.state_dict(),
                'acc_result': self.best_acc,
                'best_epoch': self.best_epoch,
                'task_id': task_id
            }, os.path.join(model_dir, "best_model.pth")
            )
            torch.save({
                'state_dict': self.latest_model.state_dict(),
                'acc_result': acc,
                'latest_epoch': epoch,
                'task_id': task_id,
                'split_selected_data': split_selected_data
            }, os.path.join(model_dir, "latest_model.pth")
            )
            torch.save(self.best_navq, os.path.join(model_dir, "best_navq.pth"))
            torch.save(self.latest_navq, os.path.join(model_dir, "latest_navq.pth"))
        pass
