import copy
import os

import torch
from scipy.spatial.distance import cdist
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
from lib.continual.FCS_models import tensor2numpy
from lib.continual.FCS_models.inc_net import FCSNet
import numpy as np
from lib.continual.utils import count_parameters
from lib.dataset import SubDataset
from lib.utils import AverageMeter, getModelSize, get_optimizer, get_scheduler

EPSILON = 1e-8


class SupContrastive(nn.Module):
    def __init__(self, reduction='mean'):
        super(SupContrastive, self).__init__()
        self.reduction = reduction

    def forward(self, y_pred, y_true):

        sum_neg = ((1 - y_true) * torch.exp(y_pred)).sum(1).unsqueeze(1)
        sum_pos = (y_true * torch.exp(-y_pred))
        num_pos = y_true.sum(1)
        loss = torch.log(1 + sum_neg * sum_pos).sum(1) / num_pos

        if self.reduction == 'mean':
            return torch.mean(loss)
        else:
            return loss


class FCS_trainer:
    def __init__(self, cfg, dataset_handler, logger):
        self.cfg = cfg
        self.dataset_handler = dataset_handler
        self.logger = logger
        self._network = FCSNet(cfg, False)

        self.gpus = torch.cuda.device_count()
        self.device_ids = [] if cfg.availabel_cudas == "" else \
            [i for i in range(len(self.cfg.availabel_cudas.strip().split(',')))]
        self.device = torch.device("cuda")

        self._protos = []
        self._covs = []
        self._radiuses = []
        self.contrast_loss = SupContrastive()
        self.encoder_k = FCSNet(cfg, False).convnet

        self.af = []

        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0

        self._old_network = None
        self.old_network_module_ptr = None
        self._data_memory, self._targets_memory = np.array([]), np.array([])
        self.topk = 1

        self.best_model = None
        self.latest_model = None

        self.best_epoch = -1
        self.best_acc = -1

    def transfer_to_cuda(self, model):
        if self.gpus > 1:
            if len(self.device_ids) > 1:
                model = torch.nn.DataParallel(model, device_ids=self.device_ids).cuda()
            else:
                model = torch.nn.DataParallel(model).cuda()
        else:
            model = model.to("cuda")
        return model

    def load_model(self, filename):
        assert filename is not None
        model = FCSNet(self.cfg, False)
        model.load_model(filename)
        self._network = self.transfer_to_cuda(model)

    def resume(self, resumed_model_path, checkpoint):
        acc_result = checkpoint['acc_result']
        self.start_task_id = checkpoint['task_id']
        self.load_model(resumed_model_path)
        self.is_resume_legal(acc_result)
        pass

    def is_resume_legal(self, acc_result):
        self.logger.info(f"Resumed acc_result: {acc_result}")
        FC_acc = self.validate_with_FC(task=self.start_task_id)  # todo
        self.logger.info(f"validate resumed model: {FC_acc} || {FC_acc.mean()}")

    def print_model(self):
        print(f"self.model: {self._network}")
        pass

    def build_optimize(self, model, base_lr, optimizer_type, momentum, weight_decay):
        # todo Done
        MODEL = model
        optimizer = get_optimizer(MODEL, BASE_LR=base_lr, optimizer_type=optimizer_type, momentum=momentum,
                                  weight_decay=weight_decay)

        return optimizer
        # todo Done
        # if typical_cls_train:
        #     optimizer = get_optimizer(self.cfg, self.model)
        #     return optimizer
        # get_optimizer(model=self.model, BASE_LR=None, optimizer_type=None, momentum=None, weight_decay=None, **kwargs)
        # optimizer = get_optimizer(self.cfg, self.model, BASE_LR=base_lr)

        # return optimizer

    def build_scheduler(self, optimizer, lr_type=None, lr_step=None, lr_factor=None, warmup_epochs=None):
        # todo optimizer, lr_type=None, lr_step=None, lr_factor=None, warmup_epochs=None
        scheduler = get_scheduler(optimizer=optimizer, lr_type=lr_type, lr_step=lr_step, lr_factor=lr_factor,
                                  warmup_epochs=warmup_epochs)
        return scheduler
        # todo
        # scheduler = get_scheduler(self.cfg, optimizer, lr_step=lr_step)
        # return scheduler

    def first_task_train_main_cls(self, train_dataset, active_classes_num, task_id):
        self._cur_task += 1
        task_size = active_classes_num - self._known_classes
        self._total_classes = active_classes_num

        self._network.update_fc(self._known_classes * 4, self._total_classes * 4, int((task_size - 1) * task_size / 2))
        self._network = self.transfer_to_cuda(self._network)
        self._network_module_ptr = self._network
        self.logger.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        self.logger.info('All params: {}'.format(count_parameters(self._network)))  # todo Done!
        self.logger.info('Trainable params: {}'.format(count_parameters(self._network, True)))

        self.logger.info('getModelSize -> All params: {}'.format(getModelSize(self._network)))  # todo Done!

        optimizer = self.build_optimize(model=self._network,
                                        base_lr=self.cfg.model.TRAIN.OPTIMIZER.BASE_LR,
                                        optimizer_type=self.cfg.model.TRAIN.OPTIMIZER.TYPE,
                                        momentum=self.cfg.model.TRAIN.OPTIMIZER.MOMENTUM,
                                        weight_decay=self.cfg.model.TRAIN.OPTIMIZER.WEIGHT_DECAY)
        scheduler = self.build_scheduler(optimizer, lr_type=self.cfg.model.TRAIN.LR_SCHEDULER.TYPE,
                                         lr_step=self.cfg.model.TRAIN.LR_SCHEDULER.LR_STEP,
                                         lr_factor=self.cfg.model.TRAIN.LR_SCHEDULER.LR_FACTOR,
                                         warmup_epochs=self.cfg.model.TRAIN.LR_SCHEDULER.WARM_EPOCH)

        # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self._network.parameters()),
        #                              lr=self.cfg.model.TRAIN.OPTIMIZER.BASE_LR,
        #                              weight_decay=self.cfg.model.TRAIN.OPTIMIZER.WEIGHT_DECAY)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.cfg.model.TRAIN.LR_SCHEDULER.LR_STEP,
        #                                             gamma=self.cfg.model.TRAIN.LR_SCHEDULER.LR_FACTOR)

        train_loader = DataLoader(dataset=train_dataset, batch_size=self.cfg.model.TRAIN.BATCH_SIZE,
                                  num_workers=self.cfg.model.TRAIN.NUM_WORKERS, shuffle=True, drop_last=True,
                                  persistent_workers=True)
        iter_num = len(train_loader)
        best_acc = 0.
        for epoch in range(1, self.cfg.model.TRAIN.MAX_EPOCH + 1):
            all_loss = AverageMeter()
            if float(torch.__version__[:3]) < 1.3:
                scheduler.step()
            iter_index = 0
            for x, y, aug_x in train_loader:
                self._network.train()
                x, y = x.to(self.device), y.to(self.device)  # --> transfer them to correct device
                aug_x = aug_x.to(self.device)
                # ---> Train MAIN MODEL
                cnt = y.shape[0]
                inputs, targets, inputs_aug = self._class_aug(x, y, inputs_aug=aug_x)

                logits, losses_all = self._compute_il2a_loss(inputs, targets, image_k=inputs_aug)
                loss_clf = losses_all["loss_clf"]
                loss_fkd = losses_all["loss_fkd"]
                loss_proto = losses_all["loss_proto"]
                loss_transfer = losses_all["loss_transfer"]
                loss_contrast = losses_all["loss_contrast"]
                # loss = loss_clf + loss_fkd + loss_proto + loss_transfer + loss_contrast
                loss = loss_clf
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                all_loss.update(loss.data.item(), cnt)
                if iter_index % self.cfg.SHOW_STEP == 0:
                    pbar_str = "Epoch: {} || Batch:{:>3d}/{}|| lr: {} || Batch_cls_Loss:{:>5.3f}" \
                               "".format(epoch, iter_index, iter_num,
                                         optimizer.param_groups[0]['lr'],
                                         all_loss.val
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

                val_acc = self.validate_with_FC(task=task_id)  # task_id 从1开始 todo
                acc_avg = val_acc.mean()
                self.logger.info(
                    "--------------current_Epoch:{:>3d}    current_Acc:{:>5.2f}%--------------".format(
                        epoch, acc_avg * 100
                    )
                )
                if acc_avg > best_acc:
                    best_acc, best_epoch = acc_avg, epoch
                    self.best_model = copy.deepcopy(self._network)
                    self.best_epoch = best_epoch
                    self.best_acc = best_acc
                    self.logger.info(
                        "--------------Best_Epoch:{:>3d}    Best_Acc:{:>5.2f}%--------------".format(
                            best_epoch, best_acc * 100
                        )
                    )

            if float(torch.__version__[:3]) >= 1.3:
                scheduler.step()

    def first_task_train_main(self, train_dataset, active_classes_num, task_id):
        self._cur_task += 1
        task_size = active_classes_num - self._known_classes
        self._total_classes = active_classes_num

        self._network.update_fc(self._known_classes * 4, self._total_classes * 4, int((task_size - 1) * task_size / 2))
        self._network = self.transfer_to_cuda(self._network)
        self._network_module_ptr = self._network
        self.logger.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        self.logger.info('All params: {}'.format(count_parameters(self._network)))  # todo Done!
        self.logger.info('Trainable params: {}'.format(count_parameters(self._network, True)))

        self.logger.info('getModelSize -> All params: {}'.format(getModelSize(self._network)))  # todo Done!

        optimizer = self.build_optimize(model=self._network,
                                        base_lr=self.cfg.model.TRAIN.OPTIMIZER.BASE_LR,
                                        optimizer_type=self.cfg.model.TRAIN.OPTIMIZER.TYPE,
                                        momentum=self.cfg.model.TRAIN.OPTIMIZER.MOMENTUM,
                                        weight_decay=self.cfg.model.TRAIN.OPTIMIZER.WEIGHT_DECAY)
        scheduler = self.build_scheduler(optimizer, lr_type=self.cfg.model.TRAIN.LR_SCHEDULER.TYPE,
                                         lr_step=self.cfg.model.TRAIN.LR_SCHEDULER.LR_STEP,
                                         lr_factor=self.cfg.model.TRAIN.LR_SCHEDULER.LR_FACTOR,
                                         warmup_epochs=self.cfg.model.TRAIN.LR_SCHEDULER.WARM_EPOCH)

        # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self._network.parameters()),
        #                              lr=self.cfg.model.TRAIN.OPTIMIZER.BASE_LR,
        #                              weight_decay=self.cfg.model.TRAIN.OPTIMIZER.WEIGHT_DECAY)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.cfg.model.TRAIN.LR_SCHEDULER.LR_STEP,
        #                                             gamma=self.cfg.model.TRAIN.LR_SCHEDULER.LR_FACTOR)

        train_loader = DataLoader(dataset=train_dataset, batch_size=self.cfg.model.TRAIN.BATCH_SIZE,
                                  num_workers=self.cfg.model.TRAIN.NUM_WORKERS, shuffle=True, drop_last=True,
                                  persistent_workers=True)
        iter_num = len(train_loader)
        best_acc = 0.
        for epoch in range(1, self.cfg.model.TRAIN.MAX_EPOCH + 1):
            all_loss = AverageMeter()
            if float(torch.__version__[:3]) < 1.3:
                scheduler.step()
            iter_index = 0
            for x, y, aug_x in train_loader:
                self._network.train()
                x, y = x.to(self.device), y.to(self.device)  # --> transfer them to correct device
                aug_x = aug_x.to(self.device)
                # ---> Train MAIN MODEL
                cnt = y.shape[0]
                inputs, targets, inputs_aug = self._class_aug(x, y, inputs_aug=aug_x)

                logits, losses_all = self._compute_il2a_loss(inputs, targets, image_k=inputs_aug)
                loss_clf = losses_all["loss_clf"]
                loss_fkd = losses_all["loss_fkd"]
                loss_proto = losses_all["loss_proto"]
                loss_transfer = losses_all["loss_transfer"]
                loss_contrast = losses_all["loss_contrast"]
                loss = loss_clf + loss_fkd + loss_proto + loss_transfer + loss_contrast
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                all_loss.update(loss.data.item(), cnt)
                if iter_index % self.cfg.SHOW_STEP == 0:
                    pbar_str = "Epoch: {} || Batch:{:>3d}/{}|| lr: {} || Batch_cls_Loss:{:>5.3f}" \
                               "".format(epoch, iter_index, iter_num,
                                         optimizer.param_groups[0]['lr'],
                                         all_loss.val
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

                val_acc = self.validate_with_FC(task=task_id)  # task_id 从1开始 todo
                acc_avg = val_acc.mean()
                self.logger.info(
                    "--------------current_Epoch:{:>3d}    current_Acc:{:>5.2f}%--------------".format(
                        epoch, acc_avg * 100
                    )
                )
                if acc_avg > best_acc:
                    best_acc, best_epoch = acc_avg, epoch
                    self.best_model = copy.deepcopy(self._network)
                    self.best_epoch = best_epoch
                    self.best_acc = best_acc
                    self.logger.info(
                        "--------------Best_Epoch:{:>3d}    Best_Acc:{:>5.2f}%--------------".format(
                            best_epoch, best_acc * 100
                        )
                    )

            if float(torch.__version__[:3]) >= 1.3:
                scheduler.step()

    def learn_new_task(self, train_dataset, active_classes_num, task):
        self._cur_task += 1
        task_size = self.dataset_handler.classes_per_task
        self._total_classes = self._known_classes + task_size
        self._network = self._network.to("cpu")
        self._network.update_fc(self._known_classes * 4, self._total_classes * 4, int((task_size - 1) * task_size / 2))
        self._network = self.transfer_to_cuda(self._network)
        self._network_module_ptr = self._network

        self.train_model(train_dataset, active_classes_num, task)
        # self.logger.info(f"Ablation study, dual training model.")
        # self.train_model(train_dataset, active_classes_num, task)

    def train_model(self, train_dataset, active_classes_num, task):
        dp_classes_num = active_classes_num - self.dataset_handler.classes_per_task
        # trains

        optimizer = self.build_optimize(model=self._network,
                                        base_lr=self.cfg.model.TRAIN.OPTIMIZER.BASE_LR,
                                        optimizer_type=self.cfg.model.TRAIN.OPTIMIZER.TYPE,
                                        momentum=self.cfg.model.TRAIN.OPTIMIZER.MOMENTUM,
                                        weight_decay=self.cfg.model.TRAIN.OPTIMIZER.WEIGHT_DECAY)
        scheduler = self.build_scheduler(optimizer, lr_type=self.cfg.model.TRAIN.LR_SCHEDULER.TYPE,
                                         lr_step=self.cfg.model.TRAIN.LR_SCHEDULER.LR_STEP,
                                         lr_factor=self.cfg.model.TRAIN.LR_SCHEDULER.LR_FACTOR,
                                         warmup_epochs=self.cfg.model.TRAIN.LR_SCHEDULER.WARM_EPOCH)

        # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self._network.parameters()),
        #                              lr=self.cfg.model.TRAIN.OPTIMIZER.BASE_LR,
        #                              weight_decay=self.cfg.model.TRAIN.OPTIMIZER.WEIGHT_DECAY)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.cfg.model.TRAIN.LR_SCHEDULER.LR_STEP,
        #                                             gamma=self.cfg.model.TRAIN.LR_SCHEDULER.LR_FACTOR)

        train_loader = DataLoader(dataset=train_dataset, batch_size=self.cfg.model.TRAIN.BATCH_SIZE,
                                  num_workers=self.cfg.model.TRAIN.NUM_WORKERS, shuffle=True, drop_last=True,
                                  persistent_workers=True)
        iter_num = len(train_loader)

        best_acc = 0.
        for epoch in range(1, self.cfg.model.TRAIN.MAX_EPOCH + 1):
            all_loss = AverageMeter()
            if float(torch.__version__[:3]) < 1.3:
                scheduler.step()
            iter_index = 0
            for x, y, aug_x in train_loader:
                self._network.train()
                inputs, targets = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                inputs_aug = aug_x.to(self.device, non_blocking=True)
                # image_q, image_k = image_q.to(
                #    self._device, non_blocking=True), image_k.to(self._device, non_blocking=True)

                inputs, targets, inputs_aug = self._class_aug(inputs, targets, inputs_aug=inputs_aug)

                logits, losses_all = self._compute_il2a_loss(inputs, targets, image_k=inputs_aug)
                loss_clf = losses_all["loss_clf"]
                loss_fkd = losses_all["loss_fkd"]
                loss_proto = losses_all["loss_proto"]
                loss_transfer = losses_all["loss_transfer"]
                loss_contrast = losses_all["loss_contrast"]
                loss = loss_clf + loss_fkd + loss_proto + loss_transfer + loss_contrast
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # measure accuracy and record loss
                all_loss.update(loss.data.item(), y.shape[0])
                if iter_index % self.cfg.SHOW_STEP == 0:
                    pbar_str = "Approach: {}| | Epoch: {} || Batch:{:>3d}/{}|| lr: {} " \
                               "|| Batch_Loss:{:>5.3f}".format("FCS", epoch, iter_index,
                                                               iter_num,
                                                               optimizer.param_groups[
                                                                   0]['lr'],
                                                               all_loss.val
                                                               )
                    self.logger.info(pbar_str)
                iter_index += 1
            if self.cfg.VALID_STEP != -1 and epoch % self.cfg.VALID_STEP == 0:

                val_acc = self.validate_with_FC(task=task)  # task_id 从1开始 todo

                if val_acc.mean() > best_acc:
                    best_acc, best_epoch = val_acc.mean(), epoch
                    self.best_model = copy.deepcopy(self._network)
                    self.best_epoch = best_epoch
                    self.best_acc = best_acc
                    self.logger.info(
                        "--------------Best_Epoch:{:>3d}    Best_Acc:{:>5.2f}%--------------".format(
                            best_epoch, best_acc * 100
                        )
                    )

            if float(torch.__version__[:3]) >= 1.3:
                scheduler.step()

    def _build_protos(self, train_dataset_for_EM):
        if self._cur_task != 0:
            proto = torch.tensor(self._protos).float().cuda()
            self._network.transfer.eval()
            with torch.no_grad():
                proto_transfer = self._network.transfer(proto)["logits"].cpu().tolist()
            self._network.transfer.train()
            for i in range(len(self._protos)):
                self._protos[i] = np.array(proto_transfer[i])
        with torch.no_grad():
            for class_idx in range(self._known_classes, self._total_classes):
                self.logger.info(f"construct protos of class_id: {class_idx}")
                class_dataset = SubDataset(original_dataset=train_dataset_for_EM,
                                           sub_labels=[class_idx])
                idx_loader = DataLoader(class_dataset, batch_size=self.cfg.model.TRAIN.BATCH_SIZE,
                                        num_workers=self.cfg.model.TRAIN.NUM_WORKERS, shuffle=False,
                                        pin_memory=False, drop_last=False, persistent_workers=True)
                vectors, _ = self._extract_vectors(idx_loader)
                class_mean = np.mean(vectors, axis=0)
                self._protos.append(class_mean)

                cov = np.cov(vectors.T)
                self._covs.append(cov)
                self._radiuses.append(np.trace(cov) / vectors.shape[1])
            self._radius = np.sqrt(np.mean(self._radiuses))

    def _class_aug(self, inputs, targets, alpha=20., mix_time=4, inputs_aug=None):

        inputs2 = torch.stack([torch.rot90(inputs, k, (2, 3)) for k in range(4)], 1)
        inputs2 = inputs2.view(-1, 3, inputs2.shape[-2], inputs2.shape[-1])
        targets2 = torch.stack([targets * 4 + k for k in range(4)], 1).view(-1)

        inputs_aug2 = torch.stack([torch.rot90(inputs_aug, k, (2, 3)) for k in range(4)], 1)
        inputs_aug2 = inputs_aug2.view(-1, 3, inputs_aug2.shape[-2], inputs_aug2.shape[-1])

        mixup_inputs = []
        mixup_targets = []

        for _ in range(mix_time):
            index = torch.randperm(inputs.shape[0])
            perm_inputs = inputs[index]
            perm_targets = targets[index]
            mask = perm_targets != targets

            select_inputs = inputs[mask]
            select_targets = targets[mask]
            perm_inputs = perm_inputs[mask]
            perm_targets = perm_targets[mask]

            lams = np.random.beta(alpha, alpha, sum(mask))
            lams = np.where((lams < 0.4) | (lams > 0.6), 0.5, lams)
            lams = torch.from_numpy(lams).to(self.device)[:, None, None, None].float()

            mixup_inputs.append(lams * select_inputs + (1 - lams) * perm_inputs)
            mixup_targets.append(self._map_targets(select_targets, perm_targets))

        mixup_inputs = torch.cat(mixup_inputs, dim=0)

        mixup_targets = torch.cat(mixup_targets, dim=0)

        inputs = torch.cat([inputs2, mixup_inputs], dim=0)
        targets = torch.cat([targets2, mixup_targets], dim=0)

        return inputs, targets, inputs_aug2

    def _compute_il2a_loss(self, inputs, targets, image_k=None):
        loss_clf, loss_fkd, loss_proto, loss_transfer, loss_contrast = \
            torch.tensor(0.), torch.tensor(0.), torch.tensor(0.), torch.tensor(0.), torch.tensor(0.)

        network_output = self._network(inputs)

        features = network_output["features"]

        if image_k != None and (self._cur_task == 0):
            b = image_k.shape[0]
            targets_part = targets[:b].clone()

            with torch.no_grad():
                # a_ = self._network.convnet.layer4[1].bn1.running_mean

                self._copy_key_encoder()
                # self.encoder_k.to(self._device)
                # features_q_ = self._network(image_k)["features"]
                features_k = self.encoder_k(image_k)["features"]
                features_k = nn.functional.normalize(features_k, dim=-1)

            features_q = nn.functional.normalize(features[:b], dim=-1)

            l_pos_global = (features_q * features_k).sum(-1).view(-1, 1)

            l_neg_global = torch.einsum('nc,ck->nk', [features_q, features_k.T])

            # logits: Nx(1+K)
            logits_global = torch.cat([l_pos_global, l_neg_global], dim=1)

            # apply temperature
            logits_global /= self.cfg.model.contrast.contrast_T

            # one-hot target from augmented image
            positive_target = torch.ones((b, 1)).cuda()
            # find same label images from label queue
            # for the query with -1, all
            negative_targets = (
                    (targets_part[:, None] == targets_part[None, :]) & (targets_part[:, None] != -1)).float().cuda()
            targets_global = torch.cat([positive_target, negative_targets], dim=1)

            loss_contrast = self.contrast_loss(logits_global, targets_global) * self.cfg.model.contrast[
                "lambda_contrast"]

        # print(network_output.keys())
        logits = network_output["logits"]
        loss_clf = F.cross_entropy(logits / self.cfg.model["temp"], targets)

        if self._cur_task != 0:
            features_old = self.old_network_module_ptr.extract_vector(inputs)

        if self._cur_task == 0:
            losses_all = {
                "loss_clf": loss_clf,
                "loss_fkd": loss_fkd,
                "loss_proto": loss_proto,
                "loss_transfer": loss_transfer,
                "loss_contrast": loss_contrast,
            }
            return logits, losses_all

        feature_transfer = self._network.transfer(features_old)["logits"]
        loss_transfer = self.cfg.model.transfer["lambda_transfer"] * self.l2loss(features, feature_transfer)

        loss_fkd = self.cfg.model.fkd["lambda_fkd"] * self.l2loss(features, features_old, mean=False)

        index = np.random.choice(range(self._known_classes), size=self.cfg.model.TRAIN.BATCH_SIZE, replace=True)

        proto_features_raw = np.array(self._protos)[index]
        proto_targets = index * 4

        proto_features = proto_features_raw + np.random.normal(0, 1, proto_features_raw.shape) * self._radius
        proto_features = torch.from_numpy(proto_features).float().to(self.device, non_blocking=True)
        proto_targets = torch.from_numpy(proto_targets).to(self.device, non_blocking=True)

        proto_features_transfer = self._network.transfer(proto_features)["logits"].detach().clone()
        proto_logits = self._network_module_ptr.fc(proto_features_transfer)["logits"][:, :self._total_classes * 4]

        loss_proto = self.cfg.model.proto["lambda_proto"] * F.cross_entropy(proto_logits / self.cfg.model["temp"],
                                                                            proto_targets)

        if image_k != None and (self._cur_task > 0):
            b = image_k.shape[0]
            targets_part = targets[:b].clone()
            targets_part_neg = targets[:b].clone()
            with torch.no_grad():
                self._copy_key_encoder()
                features_k = self.encoder_k(image_k)["features"]
                features_k = torch.cat((features_k, proto_features), dim=0)
                features_k = nn.functional.normalize(features_k, dim=-1)
                targets_part_neg = torch.cat((targets_part_neg, proto_targets), dim=0)
            # print(features_k.shape,targets_part_neg.shape,b,proto_features.shape)
            features_q = nn.functional.normalize(features[:b], dim=-1)

            l_pos_global = (features_q * features_k[:b]).sum(-1).view(-1, 1)
            l_neg_global = torch.einsum('nc,ck->nk', [features_q, features_k.T])

            # logits: Nx(1+K)
            logits_global = torch.cat([l_pos_global, l_neg_global], dim=1)
            # apply temperature
            logits_global /= self.cfg.model.contrast["lambda_contrast"]

            # one-hot target from augmented image
            positive_target = torch.ones((b, 1)).cuda()
            # find same label images from label queue
            # for the query with -1, all
            negative_targets = ((targets_part[:, None] == targets_part_neg[None, :]) & (
                    targets_part[:, None] != -1)).float().cuda()
            targets_global = torch.cat([positive_target, negative_targets], dim=1)
            loss_contrast = self.contrast_loss(logits_global, targets_global) * self.cfg.model.contrast[
                "lambda_contrast"]

        losses_all = {
            "loss_clf": loss_clf,
            "loss_fkd": loss_fkd,
            "loss_proto": loss_proto,
            "loss_transfer": loss_transfer,
            "loss_contrast": loss_contrast,
        }

        return logits, losses_all

    @torch.no_grad()
    def _copy_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        self.encoder_k.to(self.device)
        for param_q, param_k in zip(
                self._network.convnet.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_q.data

    def after_task(self):
        self._known_classes = self._total_classes
        self._old_network = self._network.copy().freeze()
        if hasattr(self._old_network, "module"):
            self.old_network_module_ptr = self._old_network.module
        else:
            self.old_network_module_ptr = self._old_network

    def _extract_vectors(self, loader, model=None):
        Model = model if model is not None else self._network
        mode = Model.training
        Model.eval()
        vectors, targets = [], []
        for data_item in loader:
            _targets = data_item[1].numpy()
            if isinstance(self._network, nn.DataParallel):
                _vectors = tensor2numpy(
                    self._network.module.extract_vector(data_item[0].to(self.device))
                )
            else:
                _vectors = tensor2numpy(
                    self._network.extract_vector(data_item[0].to(self.device))
                )

            vectors.append(_vectors)
            targets.append(_targets)
        Model.train(mode=mode)
        return np.concatenate(vectors), np.concatenate(targets)

    def _map_targets(self, select_targets, perm_targets):
        assert (select_targets != perm_targets).all()
        large_targets = torch.max(select_targets, perm_targets) - self._known_classes
        small_targets = torch.min(select_targets, perm_targets) - self._known_classes

        mixup_targets = (large_targets * (large_targets - 1) / 2 + small_targets + self._total_classes * 4).long()
        return mixup_targets

    def save_best_latest_model_data(self, model_dir, task_id, acc, epoch):
        if self.best_model is None:
            self.best_model = self._network
        if self.latest_model is None:
            self.latest_model = self._network
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
                'task_id': task_id
            }, os.path.join(model_dir, "base_latest_model.pth")
            )

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
                'task_id': task_id
            }, os.path.join(model_dir, "latest_model.pth")
            )

        pass

    def l2loss(self, inputs, targets, mean=True):

        if not mean:
            delta = torch.sqrt(torch.sum(torch.pow(inputs - targets, 2)))
            return delta
        else:
            delta = torch.sqrt(torch.sum(torch.pow(inputs - targets, 2), dim=-1))

            return torch.mean(delta)

    def validate_with_FC(self, model=None, task=None, is_test=False):
        acc = []
        Model = model if model is not None else self._network
        mode = Model.training
        Model.eval()
        for task_id in range(task):  # 这里的task 从0 开始
            if self.dataset_handler.val_datasets and (not is_test):
                predict_result = self.validate_with_FC_per_task(Model, self.dataset_handler.val_datasets[task_id], task)
            else:
                predict_result = self.validate_with_FC_per_task(Model, self.dataset_handler.test_datasets[task_id],
                                                                task)
            acc.append(predict_result)
            self.logger.info(
                f"task: {task} || per task {task_id}, validate_with_FC acc:{predict_result}"
            )
        acc = np.array(acc)
        Model.train(mode=mode)
        # print(
        #     f"task {task} validate_with_exemplars, acc_avg:{acc.mean()}")
        # self.logger.info(
        #     f"per task {task}, validate_with_exemplars, avg acc:{acc.mean()}"
        #     f"-------------------------------------------------------------"
        # )
        return acc
        pass

    def validate_with_FC_per_task(self, Model, val_dataset, task):
        # todo
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.cfg.model.TRAIN.BATCH_SIZE,
                                num_workers=self.cfg.model.TRAIN.NUM_WORKERS, shuffle=False, drop_last=False)
        top1 = AverageMeter()
        correct = 0
        active_classes_num = self.dataset_handler.classes_per_task * task
        for inputs, labels in val_loader:
            correct_temp = 0
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            with torch.no_grad():
                out = Model(inputs)["logits"][:, :self._total_classes * 4][:, ::4]
                _, balance_fc_y_hat = torch.max(out[:, 0:active_classes_num], 1)
            correct_temp += balance_fc_y_hat.eq(labels.data).cpu().sum()
            correct += correct_temp
            top1.update((correct_temp / inputs.size(0)).item(), inputs.size(0))
        return top1.avg
        pass

    def validate_with_FC_taskIL(self, task, is_test=False):
        acc = []
        mode = self._network.training
        self._network.eval()
        for task_id in range(task):  # 这里的task 从0 开始
            if self.dataset_handler.val_datasets and (not is_test):
                predict_result = self.validate_with_FC_per_task_taskIL(self.dataset_handler.val_datasets[task_id],
                                                                       task_id)
            else:
                predict_result = self.validate_with_FC_per_task_taskIL(self.dataset_handler.test_datasets[task_id],
                                                                       task_id)
            acc.append(predict_result)
            self.logger.info(
                f"task: {task} || per task {task_id}, validate_with_FC acc:{predict_result}"
            )
        acc = np.array(acc)
        self._network.train(mode=mode)
        return acc
        pass

    def validate_with_FC_per_task_taskIL(self, val_dataset, task_id):
        # todo
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.cfg.model.TRAIN.BATCH_SIZE,
                                num_workers=self.cfg.model.TRAIN.NUM_WORKERS, shuffle=False, drop_last=False)
        top1 = AverageMeter()
        correct = 0
        allowed_classes = [i for i in range(task_id * self.dataset_handler.classes_per_task,
                                            (task_id + 1) * self.dataset_handler.classes_per_task)]
        for inputs, labels in val_loader:
            correct_temp = 0
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            with torch.no_grad():
                out = self._network(inputs)["logits"][:, :self._total_classes * 4][:, ::4]
            _, balance_fc_y_hat = torch.max(out[:, allowed_classes], 1)
            balance_fc_y_hat += task_id * self.dataset_handler.classes_per_task
            correct_temp += balance_fc_y_hat.eq(labels.data).cpu().sum()
            correct += correct_temp
            top1.update((correct_temp / inputs.size(0)).item(), inputs.size(0))
        return top1.avg
        pass

    def validate_with_prototype(self, model=None, task=None, is_test=False):
        acc = []
        Model = model if model is not None else self._network
        mode = Model.training
        Model.eval()
        for task_id in range(task):  # 这里的task 从0 开始
            if self.dataset_handler.val_datasets and (not is_test):
                predict_result = self.validate_with_nme_per_task(Model, self.dataset_handler.val_datasets[task_id])
            else:
                predict_result = self.validate_with_nme_per_task(Model, self.dataset_handler.test_datasets[task_id])
            acc.append(predict_result)
            self.logger.info(
                f"task: {task} || per task {task_id}, validate_with_FC acc:{predict_result}"
            )
        acc = np.array(acc)
        Model.train(mode=mode)
        # print(
        #     f"task {task} validate_with_exemplars, acc_avg:{acc.mean()}")
        # self.logger.info(
        #     f"per task {task}, validate_with_exemplars, avg acc:{acc.mean()}"
        #     f"-------------------------------------------------------------"
        # )
        return acc

        pass

    def validate_with_nme_per_task(self, Model, val_dataset):
        # todo
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.cfg.model.TRAIN.BATCH_SIZE,
                                num_workers=self.cfg.model.TRAIN.NUM_WORKERS, shuffle=False, drop_last=False)
        assert len(self._protos) > 0
        means = self._protos / np.linalg.norm(self._protos, axis=1)[:, None]
        with torch.no_grad():
            vectors, y_true = self._extract_vectors(val_loader, Model)
        vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T

        dists = cdist(means, vectors, "sqeuclidean")  # [nb_classes, N]
        scores = dists.T  # [N, nb_classes], choose the one with the smallest distance
        pred = np.argsort(scores, axis=1)[:, :1]
        # self.logger.info(f"pred: {pred}")
        # self.logger.info(f"y_true: {y_true}")
        # self.logger.info(f"len(y_true): {len(y_true)}")
        assert len(pred) == len(y_true)
        acc = np.around((pred.T[0] == y_true).sum() * 100 / len(y_true), decimals=2)
        return acc
        pass
