import copy

import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from torchmetrics.functional import pairwise_cosine_similarity

from .DualConsist_Teacher import dual_consist_Teacher
from .deepInversionGenBN import DeepInversionGenBN
from lib.model import CrossEntropy, resnet_model
from lib.utils import AverageMeter, get_optimizer, get_scheduler, construct_prototypes, map_class_2_task
from .. import generator_models
from ..dataset import SubDataset

'''Code for <Dual-consistency Model Inversion for EFCL>'''


class DualConsistencyMI:
    def __init__(self, cfg, dataset_handler, logger):
        # self.criterion_fn = nn.CrossEntropyLoss(reduction="none")
        # self.kd_criterion = nn.MSELoss(reduction="none")
        self.cfg = cfg
        self.dataset_handler = dataset_handler
        self.logger = logger

        # moduel parameters
        self.model = None
        self.generator = None
        self.discriminator = None
        self.previous_teacher = None  # a class object including generator and the model
        self.previous_previous_teacher = None

        self.gpus = torch.cuda.device_count()
        self.device_ids = [] if cfg.availabel_cudas == "" else \
            [i for i in range(len(self.cfg.availabel_cudas.strip().split(',')))]
        self.device = torch.device("cuda:0")

        self.train_init()
        self.acc_result = None
        self.start_task_id = None
        # data weighting
        self.sample_num_per_class = None

        self.best_model = None
        self.latest_model = None
        self.best_epoch = None
        self.best_acc = None

        self.prototypes = None
        self.mse_criterion = nn.MSELoss(reduction="mean").cuda()
        self.cos_criterion = nn.CosineEmbeddingLoss(reduction='mean').cuda()
        self.criterion_fn = nn.CrossEntropyLoss(reduction="none").cuda()

    def construct_model(self):
        model = resnet_model(self.cfg)
        return model
        pass

    def print_model(self):
        print(f"self.model: {self.model}")
        # print(f"self.generator: {self.generator}")
        pass

    def build_optimize(self, model=None, base_lr=None, optimizer_type=None, momentum=None, weight_decay=None):
        # todo Done
        MODEL = model if model is not None else self.model
        optimizer = get_optimizer(MODEL, BASE_LR=base_lr, optimizer_type=optimizer_type, momentum=momentum,
                                  weight_decay=weight_decay)

        return optimizer

    def build_scheduler(self, optimizer, lr_type=None, lr_step=None, lr_factor=None, warmup_epochs=None, MAX_EPOCH=200):
        # todo optimizer, lr_type=None, lr_step=None, lr_factor=None, warmup_epochs=None
        scheduler = get_scheduler(optimizer=optimizer, lr_type=lr_type, lr_step=lr_step, lr_factor=lr_factor,
                                  warmup_epochs=warmup_epochs, MAX_EPOCH=MAX_EPOCH)
        return scheduler

    def train_init(self):
        self.model = self.construct_model()
        # self.generator = self.create_generator()
        # self.discriminator = self.create_discriminator()
        if "PRETRAINED" in self.cfg and self.cfg.PRETRAINED.MODEL:
            self.load_model(self.cfg.PRETRAINED.MODEL)
        self.model = self.transfer_to_cuda(self.model)
        # self.generator = self.transfer_to_cuda(self.generator)
        # self.discriminator = self.transfer_to_cuda(self.discriminator)

    def create_generator(self):
        # Define the backbone (MLP, LeNet, VGG, ResNet ... etc) of model
        generator = generator_models.dual_consist_generator.__dict__[self.cfg.generator.gen_model_name](
            prototypes=self.prototypes)
        return generator

    def create_discriminator(self):
        discriminator = generator_models.dual_consist_generator.__dict__[self.cfg.generator.dis_model_name]()
        return discriminator

    def transfer_to_cuda(self, model):
        if self.gpus > 1:
            if len(self.device_ids) > 1:
                model = torch.nn.DataParallel(model, device_ids=self.device_ids).to(self.device)
            else:
                model = torch.nn.DataParallel(model).to(self.device)
        else:
            model = model.to("cuda")
        return model

    def load_model(self, filename):
        assert filename is not None
        del self.model
        model = self.construct_model()
        model.load_model(filename)
        self.model = self.transfer_to_cuda(model)

    def first_task_train_main(self, train_dataset, active_classes_num, task_id):
        optimizer = self.build_optimize(model=self.model,
                                        base_lr=self.cfg.model.TRAIN.OPTIMIZER.BASE_LR,
                                        optimizer_type=self.cfg.model.TRAIN.OPTIMIZER.TYPE,
                                        momentum=self.cfg.model.TRAIN.OPTIMIZER.MOMENTUM,
                                        weight_decay=self.cfg.model.TRAIN.OPTIMIZER.WEIGHT_DECAY)
        scheduler = self.build_scheduler(optimizer, lr_type=self.cfg.model.TRAIN.LR_SCHEDULER.TYPE,
                                         lr_step=self.cfg.model.TRAIN.LR_SCHEDULER.LR_STEP,
                                         lr_factor=self.cfg.model.TRAIN.LR_SCHEDULER.LR_FACTOR,
                                         warmup_epochs=self.cfg.model.TRAIN.LR_SCHEDULER.WARM_EPOCH,
                                         MAX_EPOCH=self.cfg.model.TRAIN.MAX_EPOCH)
        criterion = CrossEntropy().to(self.device)
        best_acc = 0
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.cfg.model.TRAIN.BATCH_SIZE,
                                  num_workers=self.cfg.model.TRAIN.NUM_WORKERS, shuffle=True, drop_last=True,
                                  persistent_workers=True)
        iter_num = len(train_loader)
        self.logger.info(f"type(self.model): {type(self.model)}.")
        for epoch in range(1, self.cfg.model.TRAIN.MAX_EPOCH + 1):
            all_loss = AverageMeter()
            if float(torch.__version__[:3]) < 1.3:
                scheduler.step()
            iter_index = 0
            for x, y in train_loader:
                self.model.train()
                x, y = x.to(self.device), y.to(self.device)  # --> transfer them to correct device
                # ---> Train MAIN MODEL
                cnt = y.shape[0]
                output, _ = self.model(x)
                output = output[:, 0:active_classes_num]
                loss = criterion(output, y)
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

                val_acc = self.validate_with_FC(task=task_id)  # task_id 从1开始
                acc_avg = val_acc.mean()
                self.logger.info(
                    "--------------current_Epoch:{:>3d}    current_Acc:{:>5.2f}%--------------".format(
                        epoch, acc_avg * 100
                    )
                )
                if acc_avg > best_acc:
                    best_acc, best_epoch = acc_avg, epoch
                    self.best_model = copy.deepcopy(self.model)
                    self.best_epoch = best_epoch
                    self.best_acc = best_acc
                    self.logger.info(
                        "--------------Best_Epoch:{:>3d}    Best_Acc:{:>5.2f}%--------------".format(
                            best_epoch, best_acc * 100
                        )
                    )

            if float(torch.__version__[:3]) >= 1.3:
                scheduler.step()
        del train_loader


    def pre_steps(self, train_dataset_for_gan, dp_classes_num, model=None):
        # self.model = self.model.module
        # self.generator = self.generator.module
        # self.model = self.transfer_to_cuda(self.model)
        # self.generator = self.transfer_to_cuda(self.generator)

        # print(next(self.model.parameters()).device)
        # print(next(self.generator.parameters()).device)
        if self.generator is None:
            self.generator = self.create_generator()
            self.discriminator = self.create_discriminator()
            self.generator = self.transfer_to_cuda(self.generator)
            self.discriminator = self.transfer_to_cuda(self.discriminator)
        else:
            if hasattr(self.generator, "module"):
                self.generator.module.update_embeddings(self.prototypes)
            else:
                self.generator.update_embeddings(self.prototypes)
        Model = model if model is not None else self.model
        # for eval
        assert self.prototypes.size(0) == dp_classes_num, f"{self.prototypes.size(0)} vs. {dp_classes_num}"
        if self.previous_teacher is not None:
            self.previous_previous_teacher = self.previous_teacher
        # new teacher: cfg, solver, generator, discriminator, prototypes, class_idx
        self.previous_teacher = dual_consist_Teacher(cfg=self.cfg, solver=copy.deepcopy(Model).eval(),
                                                     generator=self.generator,
                                                     discriminator=self.discriminator,
                                                     prototypes=self.prototypes,
                                                     class_idx=np.arange(dp_classes_num))
        self.previous_teacher.train_generator(train_dataset_for_gan=train_dataset_for_gan,
                                              batch_size=self.cfg.generator.batch_size)
        pass

    def learn_new_task(self, train_dataset, active_classes_num, task):
        dp_classes_num = active_classes_num - self.dataset_handler.classes_per_task
        self.pre_steps(train_dataset_for_gan=train_dataset, dp_classes_num=dp_classes_num)
        self.train_model(train_dataset, active_classes_num, task)
        # self.train_model_with_ABD(train_dataset, active_classes_num, task)

    def after_steps(self, train_dataset, task):
        if self.cfg.use_base_half and task == int(self.dataset_handler.all_tasks / 2):
            new_classes = list(range(0, self.dataset_handler.classes_per_task * task))
        else:
            new_classes = list(range(self.dataset_handler.classes_per_task * (task - 1),
                                     self.dataset_handler.classes_per_task * task))
        new_prototypes = construct_prototypes(train_dataset, self.model, new_classes)
        new_prototypes = new_prototypes.cuda()
        if self.prototypes is None:
            self.prototypes = new_prototypes
        else:
            self.prototypes = torch.cat([self.prototypes, new_prototypes], dim=0)
        pass

    def sample(self, teacher, batch_size, dp_classes_num, return_scores=True):
        return teacher.sample(batch_size, dp_classes_num, return_scores=return_scores)

    def train_model(self, train_dataset, active_classes_num, task):
        dp_classes_num = active_classes_num - self.dataset_handler.classes_per_task

        if hasattr(self.model, "module"):
            self.model.module.freeze()
            self.model.module.unfreeze_module()
        else:
            self.model.freeze()
            self.model.unfreeze_module()

        # trains
        optimizer = self.build_optimize(model=self.model,
                                        base_lr=self.cfg.model.TRAIN.OPTIMIZER.BASE_LR,
                                        optimizer_type=self.cfg.model.TRAIN.OPTIMIZER.TYPE,
                                        momentum=self.cfg.model.TRAIN.OPTIMIZER.MOMENTUM,
                                        weight_decay=self.cfg.model.TRAIN.OPTIMIZER.WEIGHT_DECAY)
        scheduler = self.build_scheduler(optimizer, lr_type=self.cfg.model.TRAIN.LR_SCHEDULER.TYPE,
                                         lr_step=self.cfg.model.TRAIN.LR_SCHEDULER.LR_STEP,
                                         lr_factor=self.cfg.model.TRAIN.LR_SCHEDULER.LR_FACTOR,
                                         warmup_epochs=self.cfg.model.TRAIN.LR_SCHEDULER.WARM_EPOCH,
                                         MAX_EPOCH=self.cfg.model.TRAIN.MAX_EPOCH)

        train_loader = DataLoader(dataset=train_dataset, batch_size=self.cfg.model.TRAIN.BATCH_SIZE,
                                  num_workers=self.cfg.model.TRAIN.NUM_WORKERS, shuffle=True, drop_last=True,
                                  persistent_workers=True)
        iter_num = len(train_loader)
        criterion = CrossEntropy().to(self.device)
        best_acc = 0.

        for epoch in range(1, self.cfg.model.TRAIN.MAX_EPOCH + 1):
            all_loss = AverageMeter()
            if float(torch.__version__[:3]) < 1.3:
                scheduler.step()
            iter_index = 0
            for x, y in train_loader:
                self.model.train()
                x, y = x.to(self.device), y.to(self.device)  # --> transfer them to correct device
                # ---> Train MAIN MODEL
                # data replay
                x_replay, _, _ = self.sample(self.previous_teacher, len(x), dp_classes_num)
                x_com = torch.cat([x, x_replay], dim=0)
                logits_com, features_com = self.model(x_com)
                logits_com = logits_com[:, 0:active_classes_num]

                loss_class = criterion(logits_com[:y.shape[0], dp_classes_num:active_classes_num],
                                       y % self.dataset_handler.classes_per_task)

                logits_com_for_distill = logits_com[:, 0:dp_classes_num]

                pre_logits, pre_features = self.previous_teacher.get_pre_outputs(x_com)
                pre_logits_for_distill = pre_logits[:, 0:dp_classes_num]
                loss_kd = self.mse_criterion(logits_com_for_distill, pre_logits_for_distill)
                loss_flag = torch.ones([features_com.size(0)]).cuda()  # 需要初始化一个N维的1或-1
                loss_kd += self.cos_criterion(features_com, pre_features, loss_flag)
                total_loss = loss_class + self.cfg.model.kd_gamma * loss_kd
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                # measure accuracy and record loss
                all_loss.update(total_loss.data.item(), x_com.size(0))
                if iter_index % self.cfg.SHOW_STEP == 0:
                    pbar_str = "Approach: {}| | Epoch: {} || Batch:{:>3d}/{}|| lr: {} " \
                               "|| Batch_Loss:{:>5.3f}".format(self.cfg.trainer_name, epoch, iter_index,
                                                               iter_num,
                                                               optimizer.param_groups[
                                                                   0]['lr'],
                                                               all_loss.val
                                                               )
                    self.logger.info(pbar_str)
                iter_index += 1
            if self.cfg.VALID_STEP != -1 and epoch % self.cfg.VALID_STEP == 0:

                val_acc = self.validate_with_FC(task=task)  # task_id 从1开始

                if val_acc.mean() > best_acc:
                    best_acc, best_epoch = val_acc.mean(), epoch
                    self.best_model = copy.deepcopy(self.model)
                    self.best_epoch = best_epoch
                    self.best_acc = best_acc
                    self.logger.info(
                        "--------------Best_Epoch:{:>3d}    Best_Acc:{:>5.2f}%--------------".format(
                            best_epoch, best_acc * 100
                        )
                    )

            if float(torch.__version__[:3]) >= 1.3:
                scheduler.step()
        if hasattr(self.model, "module"):
            self.model.module.unfreeze_all()
        else:
            self.model.unfreeze_all()

    def train_model_with_ABD(self, train_dataset, active_classes_num, task):
        dp_classes_num = active_classes_num - self.dataset_handler.classes_per_task
        # trains
        optimizer = self.build_optimize(model=self.model,
                                        base_lr=self.cfg.model.TRAIN.OPTIMIZER.BASE_LR,
                                        optimizer_type=self.cfg.model.TRAIN.OPTIMIZER.TYPE,
                                        momentum=self.cfg.model.TRAIN.OPTIMIZER.MOMENTUM,
                                        weight_decay=self.cfg.model.TRAIN.OPTIMIZER.WEIGHT_DECAY)
        scheduler = self.build_scheduler(optimizer, lr_type=self.cfg.model.TRAIN.LR_SCHEDULER.TYPE,
                                         lr_step=self.cfg.model.TRAIN.LR_SCHEDULER.LR_STEP,
                                         lr_factor=self.cfg.model.TRAIN.LR_SCHEDULER.LR_FACTOR,
                                         warmup_epochs=self.cfg.model.TRAIN.LR_SCHEDULER.WARM_EPOCH,
                                         MAX_EPOCH=self.cfg.model.TRAIN.MAX_EPOCH)

        train_loader = DataLoader(dataset=train_dataset, batch_size=self.cfg.model.TRAIN.BATCH_SIZE,
                                  num_workers=self.cfg.model.TRAIN.NUM_WORKERS, shuffle=True, drop_last=True,
                                  persistent_workers=True)
        iter_num = len(train_loader)
        criterion = CrossEntropy().to(self.device)
        best_acc = 0.
        cls_criterion = CrossEntropy().to(self.device)
        for epoch in range(1, self.cfg.model.TRAIN.MAX_EPOCH + 1):
            all_loss = AverageMeter()
            if float(torch.__version__[:3]) < 1.3:
                scheduler.step()
            iter_index = 0
            for x, y in train_loader:
                self.model.train()
                x, y = x.to(self.device), y.to(self.device)  # --> transfer them to correct device
                # ---> Train MAIN MODEL
                # data replay
                x_replay, y_replay, _ = self.sample(self.previous_teacher, len(x), dp_classes_num)
                x_com = torch.cat([x, x_replay], dim=0)
                y_com = torch.cat([y, y_replay], dim=0)
                logits_com, features_com = self.model(x_com)
                logits_com = logits_com[:, 0:active_classes_num]
                pre_logits_com, pre_features_com = self.previous_teacher.get_pre_outputs(x_com)
                pre_logits_com = pre_logits_com[:, 0:dp_classes_num]

                mappings = torch.ones(y_com.size(), dtype=torch.float32).to(self.device)
                rnt = 1.0 * dp_classes_num / active_classes_num
                mappings[:dp_classes_num] = rnt
                mappings[dp_classes_num:] = 1 - rnt
                dw_cls = mappings[y_com.long()]
                total_loss = self.ABD_loss(x_com=x_com, y_com=y_com, y=y, logits_com=logits_com,
                                           features_com=features_com, pre_logits_com=pre_logits_com,
                                           cls_criterion=cls_criterion, dp_classes_num=dp_classes_num,
                                           active_classes_num=active_classes_num, dw_cls=dw_cls)


                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                # measure accuracy and record loss
                all_loss.update(total_loss.data.item(), x_com.size(0))
                if iter_index % self.cfg.SHOW_STEP == 0:
                    pbar_str = "Approach: {}| | Epoch: {} || Batch:{:>3d}/{}|| lr: {} " \
                               "|| Batch_Loss:{:>5.3f}".format(self.cfg.trainer_name, epoch, iter_index,
                                                               iter_num,
                                                               optimizer.param_groups[
                                                                   0]['lr'],
                                                               all_loss.val
                                                               )
                    self.logger.info(pbar_str)
                iter_index += 1
            if self.cfg.VALID_STEP != -1 and epoch % self.cfg.VALID_STEP == 0:

                val_acc = self.validate_with_FC(task=task)  # task_id 从1开始

                if val_acc.mean() > best_acc:
                    best_acc, best_epoch = val_acc.mean(), epoch
                    self.best_model = copy.deepcopy(self.model)
                    self.best_epoch = best_epoch
                    self.best_acc = best_acc
                    self.logger.info(
                        "--------------Best_Epoch:{:>3d}    Best_Acc:{:>5.2f}%--------------".format(
                            best_epoch, best_acc * 100
                        )
                    )

            if float(torch.__version__[:3]) >= 1.3:
                scheduler.step()
        if hasattr(self.model, "module"):
            self.model.module.unfreeze_all()
        else:
            self.model.unfreeze_all()

    def ft_criterion(self, logits, targets, data_weights):
        loss_supervised = (self.criterion_fn(logits, targets.long()) * data_weights).mean()
        return loss_supervised

    def ABD_loss(self, x_com, y_com, y, logits_com, features_com, pre_logits_com, cls_criterion, dp_classes_num,
                 active_classes_num, dw_cls):
        loss_class = cls_criterion(logits_com[:y.shape[0], dp_classes_num:active_classes_num],
                                   y % self.dataset_handler.classes_per_task)

        with torch.no_grad():
            feat_class = self.model(x_com, is_nograd=True, feature_flag=True).detach()
        ft_logits = self.model(feat_class, train_cls_use_features=True)
        ft_logits = ft_logits[:, 0:active_classes_num]
        loss_class += self.ft_criterion(ft_logits, y_com, dw_cls)
        logits_KD = self.previous_teacher.generate_score_by_features(features_com,
                                                                     active_classes_num=dp_classes_num,
                                                                     is_nograd=False)
        logits_KD_past = pre_logits_com
        loss_kd = self.cfg.model.kd_lambda * self.mse_criterion(logits_KD, logits_KD_past)
        total_loss = loss_class + loss_kd
        return total_loss

    def validate_with_FC(self, test_model=None, task=None):
        acc = []
        model = test_model if test_model is not None else self.model
        mode = model.training
        model.eval()
        for task_id in range(task):  # 这里的task 从0 开始
            predict_result = self.validate_with_FC_per_task(model,
                                                            self.dataset_handler.test_datasets[task_id],
                                                            task)
            acc.append(predict_result)
            self.logger.info(
                f"task: {task} || per task {task_id}, validate_with_FC acc:{predict_result}"
            )
        acc = np.array(acc)
        model.train(mode=mode)
        return acc
        pass

    def validate_with_FC_Prototypical_Routing(self, test_model=None, task=None):
        acc = []
        model = test_model if test_model is not None else self.model
        mode = model.training
        model.eval()
        if (self.cfg.use_base_half and task == int(self.dataset_handler.all_tasks / 2)) or task == 1:
            for task_id in range(task):  # 这里的task 从0 开始
                predict_result = self.validate_with_FC_per_task(model,
                                                                self.dataset_handler.test_datasets[task_id],
                                                                task)
                acc.append(predict_result)
                self.logger.info(
                    f"task: {task} || per task {task_id}, validate_with_FC acc:{predict_result}"
                )
        else:
            for task_id in range(task):  # 这里的task 从0 开始
                predict_result = self.validate_with_Prototypical_Routing_per_task(model,
                                                                                  self.dataset_handler.test_datasets[
                                                                                      task_id],
                                                                                  task)
                acc.append(predict_result)
                self.logger.info(
                    f"task: {task} || per task {task_id}, validate_with_FC acc:{predict_result}"
                )
        acc = np.array(acc)
        model.train(mode=mode)
        return acc
        pass

    def validate_with_FC_per_task(self, model, val_dataset, task):
        # todo
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.cfg.model.TRAIN.BATCH_SIZE,
                                num_workers=self.cfg.model.TRAIN.NUM_WORKERS, shuffle=False, drop_last=False)
        top1 = AverageMeter()
        correct = 0
        active_classes_num = self.dataset_handler.classes_per_task * task
        for inputs, labels in val_loader:
            correct_temp = 0
            inputs, labels = inputs.to("cuda"), labels.to("cuda")
            out = model(x=inputs, is_nograd=True, get_classifier=True)
            out = out[:, 0:active_classes_num]
            _, balance_fc_y_hat = torch.max(out, 1)
            correct_temp += balance_fc_y_hat.eq(labels.data).cpu().sum()
            correct += correct_temp
            top1.update((correct_temp / inputs.size(0)).item(), inputs.size(0))
        return top1.avg
        pass

    def validate_with_Prototypical_Routing_per_task(self, model, val_dataset, task):
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.cfg.model.TRAIN.BATCH_SIZE,
                                num_workers=self.cfg.model.TRAIN.NUM_WORKERS, shuffle=False, drop_last=False)
        top1 = AverageMeter()
        correct = 0
        for inputs, labels in val_loader:
            correct_temp = 0
            inputs, labels = inputs.to("cuda"), labels.to("cuda")
            logits, features = model(x=inputs, is_nograd=True)
            logits_for_features = pairwise_cosine_similarity(features, self.prototypes)
            _, class_index = torch.max(logits_for_features, 1)
            '''task_index start from 0'''
            task_index = map_class_2_task(class_index, self.dataset_handler.classes_per_task)
            allowed_classes = torch.tensor([[i for i in range(j * self.dataset_handler.classes_per_task,
                                                              (j + 1) * self.dataset_handler.classes_per_task)]
                                            for j in task_index]).cuda()
            seletected_logits = []
            for index in range(allowed_classes.size(0)):
                seletected_logits.append(logits[index, allowed_classes[index]])
            seletected_logits = torch.stack(seletected_logits, dim=0)
            _, balance_fc_y_hat = torch.max(seletected_logits, 1)
            balance_fc_y_hat += task_index * self.dataset_handler.classes_per_task
            correct_temp += balance_fc_y_hat.eq(labels.data).cpu().sum()
            correct += correct_temp
            top1.update((correct_temp / inputs.size(0)).item(), inputs.size(0))
        return top1.avg
        pass

    def validate_with_FC_taskIL(self, task, test_model=None, is_test=False):
        acc = []
        model = test_model if test_model is not None else self.model
        mode = model.training
        model.eval()
        for task_id in range(task):  # 这里的task 从0 开始
            if self.dataset_handler.val_datasets and (not is_test):
                predict_result = self.validate_with_FC_per_task_taskIL(self.dataset_handler.val_datasets[task_id],
                                                                       task_id, model)
            else:
                predict_result = self.validate_with_FC_per_task_taskIL(self.dataset_handler.test_datasets[task_id],
                                                                       task_id, model)
            acc.append(predict_result)
            self.logger.info(
                f"task: {task} || per task {task_id}, validate_with_FC acc:{predict_result}"
            )
        acc = np.array(acc)
        self.model.train(mode=mode)
        return acc
        pass

    def validate_with_FC_per_task_taskIL(self, val_dataset, task_id, model):
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
            out = model(x=inputs, is_nograd=True, get_classifier=True)
            _, balance_fc_y_hat = torch.max(out[:, allowed_classes], 1)
            balance_fc_y_hat += task_id * self.dataset_handler.classes_per_task
            correct_temp += balance_fc_y_hat.eq(labels.data).cpu().sum()
            correct += correct_temp
            top1.update((correct_temp / inputs.size(0)).item(), inputs.size(0))
        return top1.avg
        pass


