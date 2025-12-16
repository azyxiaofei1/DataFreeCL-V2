import copy
from math import log, sqrt
import torch
from torch import nn
from torch.utils.data import DataLoader

from .deepInversionGenBN import DeepInversionGenBN
from lib.model import CrossEntropy, BalancedSoftmax
from lib.model.distill_relation import RKDAngleLoss
from lib.utils import AverageMeter


'''Code for <Always Be Dreaming: A New Approach for Data-Free Class-Incremental Learning>'''

'''Code for <R-DFCIL: Relation-Guided Representation Learning for Data-Free Class Incremental Learning>'''


class RDFCL(DeepInversionGenBN):

    def __init__(self, cfg, dataset_handler, logger):
        super(RDFCL, self).__init__(cfg, dataset_handler, logger)
        self.rkd = RKDAngleLoss(
            self.cfg.extractor.output_feature_dim,
            proj_dim=2 * self.cfg.extractor.output_feature_dim,
        )
        self.rkd = self.rkd.to(self.device)
        self.criterion_fn = nn.CrossEntropyLoss(reduction="none")

    def ft_criterion(self, logits, targets, data_weights):
        loss_supervised = (self.criterion_fn(logits, targets.long()) * data_weights).mean()
        return loss_supervised

    def learn_new_task(self, train_dataset, active_classes_num, task):
        dp_classes_num = active_classes_num - self.dataset_handler.classes_per_task
        self.pre_steps(dp_classes_num)
        self.train_model(train_dataset, active_classes_num, task)
        # self.logger.info(f"Ablation study: Dual training model.")
        # self.train_model(train_dataset, active_classes_num, task)

    def train_model(self, train_dataset, active_classes_num, task):
        dp_classes_num = active_classes_num - self.dataset_handler.classes_per_task
        model_plus_rkd = nn.ModuleList([self.model, self.rkd])
        alpha = log(self.dataset_handler.classes_per_task / 2 + 1, 2)
        beta2 = 1.0 * dp_classes_num / self.dataset_handler.classes_per_task
        beta = sqrt(beta2)
        # trains
        optimizer = self.build_optimize(model=model_plus_rkd,
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
                x_replay, y_replay, pre_logits_replay = self.sample(self.previous_teacher, len(x), dp_classes_num)

                logits_new_classes, features_new_classes = self.model(x)
                logits_new_classes = logits_new_classes[:, dp_classes_num:active_classes_num]
                pre_features_new_classes = self.previous_teacher.generate_features(x)

                loss_class = criterion(logits_new_classes, y % self.dataset_handler.classes_per_task) \
                             * (self.cfg.model.ce_lambda * (1 + 1 / alpha) / beta)

                logits_old_classes, features_old_classes = self.model(x_replay)
                logits_old_classes = logits_old_classes[:, :dp_classes_num]
                loss_hkd = (self.kd_criterion(logits_old_classes, pre_logits_replay).sum(dim=1)).mean() / (
                    logits_old_classes.size(1)) * self.cfg.model.hkd_lambda

                loss_rkd = self.rkd(student=features_new_classes,
                                    teacher=pre_features_new_classes) * self.cfg.model.rkd_lambda
                loss_kd = (loss_hkd + loss_rkd) * (alpha * beta)

                total_loss = loss_class + loss_kd
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                # measure accuracy and record loss
                all_loss.update(total_loss.data.item(), y.shape[0] + y_replay.shape[0])
                if iter_index % self.cfg.SHOW_STEP == 0:
                    pbar_str = "Approach: {}| | Epoch: {} || Batch:{:>3d}/{}|| lr: {} " \
                               "|| Batch_Loss:{:>5.3f}".format("RDFCL", epoch, iter_index,
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

        # self.fine_tune(train_dataset, active_classes_num, task)
        self.fine_tune_bsce(train_dataset, active_classes_num, task)

    def fine_tune(self, train_dataset, active_classes_num, task_id):
        dp_classes_num = active_classes_num - self.dataset_handler.classes_per_task
        alpha = log(self.dataset_handler.classes_per_task / 2 + 1, 2)
        beta2 = 1.0 * dp_classes_num / self.dataset_handler.classes_per_task
        beta = sqrt(beta2)
        # trains
        ft_optimizer = self.build_optimize(model=self.model,
                                           base_lr=self.cfg.model.finetune.BASE_LR,
                                           optimizer_type=self.cfg.model.finetune.TYPE,
                                           momentum=self.cfg.model.finetune.MOMENTUM,
                                           weight_decay=self.cfg.model.finetune.WEIGHT_DECAY)
        ft_scheduler = self.build_scheduler(ft_optimizer, lr_type=self.cfg.model.finetune.LR_TYPE,
                                            lr_step=self.cfg.model.finetune.LR_STEP,
                                            lr_factor=self.cfg.model.finetune.LR_FACTOR,
                                            warmup_epochs=self.cfg.model.finetune.WARM_EPOCH,
                                            MAX_EPOCH=self.cfg.model.finetune.MAX_EPOCH)

        train_loader = DataLoader(dataset=train_dataset, batch_size=self.cfg.model.finetune.BATCH_SIZE,
                                  num_workers=self.cfg.model.finetune.NUM_WORKERS, shuffle=True, drop_last=True,
                                  persistent_workers=True)
        iter_num = len(train_loader)
        best_acc = 0.
        dw_cls = torch.ones(2 * self.cfg.model.finetune.BATCH_SIZE, dtype=torch.float32).to(self.device)
        dw_sum = dp_classes_num ** 2 + self.dataset_handler.classes_per_task ** 2
        dw_cls[:self.cfg.model.finetune.BATCH_SIZE] = self.dataset_handler.classes_per_task / dw_sum
        dw_cls[self.cfg.model.finetune.BATCH_SIZE:] = dp_classes_num / dw_sum
        for epoch in range(1, self.cfg.model.finetune.MAX_EPOCH + 1):
            all_loss = AverageMeter()
            if float(torch.__version__[:3]) < 1.3:
                ft_scheduler.step()
            iter_index = 0
            for x, y in train_loader:
                self.model.train()
                x, y = x.to(self.device), y.to(self.device)  # --> transfer them to correct device
                # ---> Train MAIN MODEL
                # data replay
                x_replay, y_replay, pre_logits_replay = self.sample(self.previous_teacher, len(x), dp_classes_num)
                assert y.shape[0] == y_replay.shape[0] == self.cfg.model.finetune.BATCH_SIZE
                x_com = torch.cat([x, x_replay], dim=0)
                y_com = torch.cat([y, y_replay], dim=0)

                logits_com = self.model(x_com, train_classifier=True)
                logits_com = logits_com[:, 0:active_classes_num]

                logits_old_classes = logits_com[y.shape[0]:, :dp_classes_num]
                # loss_hkd = (self.kd_criterion(logits_old_classes, pre_logits_replay).sum(dim=1)).mean() / (
                #     logits_old_classes.size(1)) * (self.cfg.model.hkd_lambda * alpha * beta)

                # loss_class = self.ft_criterion(logits_com, y_com, dw_cls) * self.cfg.model.ce_lambda
                loss_class = self.ft_criterion(logits_com, y_com, dw_cls)

                # total_loss = loss_class + loss_hkd
                total_loss = loss_class
                ft_optimizer.zero_grad()
                total_loss.backward()
                ft_optimizer.step()
                # measure accuracy and record loss
                all_loss.update(total_loss.data.item(), y.shape[0] + y_replay.shape[0])
                if iter_index % self.cfg.SHOW_STEP == 0:
                    pbar_str = "Approach: {}-Finetune || Epoch: {} || Batch:{:>3d}/{}|| lr: {} " \
                               "|| Batch_Loss:{:>5.3f}".format("RDFCL", epoch, iter_index,
                                                               iter_num,
                                                               ft_optimizer.param_groups[
                                                                   0]['lr'],
                                                               all_loss.val
                                                               )
                    self.logger.info(pbar_str)
                iter_index += 1
            if self.cfg.VALID_STEP != -1 and epoch % self.cfg.VALID_STEP == 0:

                val_acc = self.validate_with_FC(task=task_id)  # task_id 从1开始

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
                ft_scheduler.step()

        pass

    def fine_tune_bsce(self, train_dataset, active_classes_num, task_id):
        dp_classes_num = active_classes_num - self.dataset_handler.classes_per_task
        alpha = log(self.dataset_handler.classes_per_task / 2 + 1, 2)
        beta2 = 1.0 * dp_classes_num / self.dataset_handler.classes_per_task
        beta = sqrt(beta2)
        # trains
        ft_optimizer = self.build_optimize(model=self.model,
                                           base_lr=self.cfg.model.finetune.BASE_LR,
                                           optimizer_type=self.cfg.model.finetune.TYPE,
                                           momentum=self.cfg.model.finetune.MOMENTUM,
                                           weight_decay=self.cfg.model.finetune.WEIGHT_DECAY)
        ft_scheduler = self.build_scheduler(ft_optimizer, lr_type=self.cfg.model.finetune.LR_TYPE,
                                            lr_step=self.cfg.model.finetune.LR_STEP,
                                            lr_factor=self.cfg.model.finetune.LR_FACTOR,
                                            warmup_epochs=self.cfg.model.finetune.WARM_EPOCH,
                                            MAX_EPOCH=self.cfg.model.finetune.MAX_EPOCH)

        train_loader = DataLoader(dataset=train_dataset, batch_size=self.cfg.model.finetune.BATCH_SIZE,
                                  num_workers=self.cfg.model.finetune.NUM_WORKERS, shuffle=True, drop_last=True,
                                  persistent_workers=True)
        iter_num = len(train_loader)
        best_acc = 0.
        sample_num_per_class = torch.ones(active_classes_num, dtype=torch.float32).to(self.device)
        sample_num_per_class[:dp_classes_num] = self.cfg.model.TRAIN.BATCH_SIZE / dp_classes_num
        sample_num_per_class[
        dp_classes_num:] = self.cfg.model.TRAIN.BATCH_SIZE / self.dataset_handler.classes_per_task

        bsce_criterion = BalancedSoftmax(sample_per_class=sample_num_per_class).to(self.device)
        for epoch in range(1, self.cfg.model.finetune.MAX_EPOCH + 1):
            all_loss = AverageMeter()
            if float(torch.__version__[:3]) < 1.3:
                ft_scheduler.step()
            iter_index = 0
            for x, y in train_loader:
                self.model.train()
                x, y = x.to(self.device), y.to(self.device)  # --> transfer them to correct device
                # ---> Train MAIN MODEL
                # data replay
                x_replay, y_replay, pre_logits_replay = self.sample(self.previous_teacher, len(x), dp_classes_num)
                assert y.shape[0] == y_replay.shape[0] == self.cfg.model.finetune.BATCH_SIZE
                x_com = torch.cat([x, x_replay], dim=0)
                y_com = torch.cat([y, y_replay], dim=0)

                logits_com = self.model(x_com, train_classifier=True)
                logits_com = logits_com[:, 0:active_classes_num]

                logits_old_classes = logits_com[y.shape[0]:, :dp_classes_num]
                # loss_hkd = (self.kd_criterion(logits_old_classes, pre_logits_replay).sum(dim=1)).mean() / (
                #     logits_old_classes.size(1)) * (self.cfg.model.hkd_lambda * alpha * beta)
                loss_class = bsce_criterion(input=logits_com, label=y_com)

                # total_loss = loss_class + loss_hkd
                total_loss = loss_class
                ft_optimizer.zero_grad()
                total_loss.backward()
                ft_optimizer.step()
                # measure accuracy and record loss
                all_loss.update(total_loss.data.item(), y.shape[0] + y_replay.shape[0])
                if iter_index % self.cfg.SHOW_STEP == 0:
                    pbar_str = "Approach: {}-Finetune || Epoch: {} || Batch:{:>3d}/{}|| lr: {} " \
                               "|| Batch_Loss:{:>5.3f}".format("RDFCL", epoch, iter_index,
                                                               iter_num,
                                                               ft_optimizer.param_groups[
                                                                   0]['lr'],
                                                               all_loss.val
                                                               )
                    self.logger.info(pbar_str)
                iter_index += 1
            if self.cfg.VALID_STEP != -1 and epoch % self.cfg.VALID_STEP == 0:

                val_acc = self.validate_with_FC(task=task_id)  # task_id 从1开始

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
                ft_scheduler.step()

        pass
