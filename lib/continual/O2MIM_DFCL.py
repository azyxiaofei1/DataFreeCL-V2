from __future__ import print_function

import torch
import torch.nn as nn

import copy

from torch.utils.data import DataLoader

from .deepInversionGenBN import DeepInversionGenBN
from lib.model import CrossEntropy
from lib.utils import AverageMeter



class O2MIM_DFCL(DeepInversionGenBN):

    def __init__(self, cfg, dataset_handler, logger):
        super(O2MIM_DFCL, self).__init__(cfg, dataset_handler, logger)
        self.criterion_fn = nn.CrossEntropyLoss(reduction="none")

    def ft_criterion(self, logits, targets, data_weights):
        loss_supervised = (self.criterion_fn(logits, targets.long()) * data_weights).mean()
        return loss_supervised

    def learn_new_task(self, train_dataset, active_classes_num, task):
        dp_classes_num = active_classes_num - self.dataset_handler.classes_per_task
        self.pre_steps(dp_classes_num)
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
        criterion = CrossEntropy().cuda()
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
                x_replay, y_replay, _ = self.sample(self.previous_teacher, len(x), dp_classes_num)

                x_com = torch.cat([x, x_replay], dim=0)
                y_com = torch.cat([y, y_replay], dim=0)
                pre_logits_com = torch.cat([pre_logits, pre_logits_replay], dim=0)

                mappings = torch.ones(y_com.size(), dtype=torch.float32).cuda()
                rnt = 1.0 * dp_classes_num / active_classes_num
                mappings[:dp_classes_num] = rnt
                mappings[dp_classes_num:] = 1 - rnt
                dw_cls = mappings[y_com.long()]

                logits_com, features_com = self.model(x_com)
                logits_com = logits_com[:, 0:active_classes_num]

                loss_class = criterion(logits_com[:y.shape[0], dp_classes_num:active_classes_num],
                                       y % self.dataset_handler.classes_per_task) * (1 - rnt)

                with torch.no_grad():
                    feat_class = self.model(x_com, is_nograd=True, feature_flag=True).detach()
                ft_logits = self.model(feat_class, train_cls_use_features=True)
                loss_class += self.ft_criterion(ft_logits, y_com, dw_cls)

                logits_KD = self.previous_teacher.generate_score_by_features(features_com,
                                                                             active_classes_num=dp_classes_num,
                                                                             is_nograd=False)
                logits_KD_past = pre_logits_com
                loss_kd = self.cfg.model.kd_lambda * (
                    self.kd_criterion(logits_KD, logits_KD_past).sum(dim=1)).mean() / (
                              logits_KD.size(1))

                total_loss = loss_class + loss_kd
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                # measure accuracy and record loss
                all_loss.update(total_loss.data.item(), y_com.shape[0])
                if iter_index % self.cfg.SHOW_STEP == 0:
                    pbar_str = "Approach: {}| | Epoch: {} || Batch:{:>3d}/{}|| lr: {} " \
                               "|| Batch_Loss:{:>5.3f}".format("AlwaysBeDreaming", epoch, iter_index,
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

