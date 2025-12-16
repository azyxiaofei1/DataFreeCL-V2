

import os
from math import log, sqrt
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import copy
from torch.optim import Adam
from torch.utils.data import DataLoader, ConcatDataset
from torch.nn import functional as Func

from OldDatasets import SynthesizedDataset
from lib import generator_models
from lib.continual.datafree_helper import Teacher

from lib.utils.util_class import FCTM_GradCAM, GradCAM
from lib.model.distill_relation import RKDAngleLoss
from .deepInversionGenBN import DeepInversionGenBN

from .datafree_helper import FCTM_Teacher
from lib.model import BalancedSoftmax, FCTM_model
from lib.utils import count_sample_num_per_class, contruct_data_weight, construct_condition_labels


class DEA_DFCL(DeepInversionGenBN):

    def __init__(self, cfg, dataset_handler, logger):
        super(DEA_DFCL, self).__init__(cfg, dataset_handler, logger)
        self.FCTM = None
        self.fctm_teacher = None
        self.previous_fctm_teacher = None

        self.criterion_fn = nn.CrossEntropyLoss(reduction="none")
        self.rkd = RKDAngleLoss(
            self.cfg.extractor.output_feature_dim,
            proj_dim=2 * self.cfg.extractor.output_feature_dim,
        )
        self.rkd = self.transfer_to_cuda(self.rkd)

    def ft_criterion(self, logits, targets, data_weights):
        loss_supervised = (self.criterion_fn(logits, targets.long()) * data_weights).mean()
        return loss_supervised

    def bsce_criterion(self, logits, targets, sample_num_per_class):
        bsce_criterion = BalancedSoftmax(sample_num_per_class).cuda()
        loss_supervised = bsce_criterion(logits, targets)
        return loss_supervised

    def learn_new_task(self, train_dataset, active_classes_num, task):
        dp_classes_num = active_classes_num - self.dataset_handler.classes_per_task
        self.pre_steps(dp_classes_num)  # 训练生成器
        self.train_FCTM(train_dataset, active_classes_num, task)
        # self.RDFCL_train_FCTM(train_dataset, active_classes_num, task)  # 训练扩展模型

        oldmodel_val_newclass = self.validate_with_FC(test_model=self.model, task=task)
        val_acc_FC_str = f'task: {task} classififer:{"FC"} oldmodel_val_newclass——val_acc: ' \
                         f'{oldmodel_val_newclass}, avg: {oldmodel_val_newclass.mean()} '
        self.logger.info(val_acc_FC_str)
        fctm_val_previous_tasks = self.validate_with_FCTM(task - 1, is_test=False)
        val_acc_FC_str = f'task: {task} classififer:{"FC"} fctm_val_previous_tasks_acc: {fctm_val_previous_tasks}, ' \
                         f'avg: {fctm_val_previous_tasks.mean()} '
        self.logger.info(val_acc_FC_str)
        fctm_val = self.validate_with_FCTM(task, is_test=False)
        val_acc_FC_str = f'task: {task} classififer:{"FC"} fctm_val_acc: {fctm_val}, avg: {fctm_val.mean()}, ' \
                         f'previous_avg: {fctm_val[0:-1].mean()} '
        self.logger.info(val_acc_FC_str)
        self.pre_steps_before_train_model(active_classes_num,
                                          active_classes_num - self.dataset_handler.classes_per_task)  # 训练生成器
        # self.train_model(train_dataset, active_classes_num, task)
        # self.train_model_no_dataset(train_dataset, active_classes_num, task)
        self.train_model_old_classes_generator(train_dataset, active_classes_num, task)
        # self.train_model_factory(train_dataset, active_classes_num, task)
        # self.ABD_train_model(train_dataset, active_classes_num, task)
        # self.RDFCL_train_FCTM(train_dataset, active_classes_num, task_id) #训练扩展模型
        # self.RDFCL_train_model(train_dataset, active_classes_num, task_id) #压缩扩展模型
        # self.fine_tune(train_dataset, active_classes_num, task)
        self.fine_tune_old_classes_generator(train_dataset, active_classes_num, task)

        # trains

    def train_FCTM(self, train_dataset, active_classes_num, task):
        dpt_classes_num = active_classes_num - self.dataset_handler.classes_per_task
        if self.FCTM is None:
            self.FCTM = FCTM_model(self.cfg, backbone_model=copy.deepcopy(self.model))
            self.FCTM = self.transfer_to_cuda(self.FCTM)
            print("FCTM:", self.FCTM)
        optimizer = self.build_optimize(model=self.FCTM,
                                        base_lr=self.cfg.FCTM.TRAIN.OPTIMIZER.BASE_LR,
                                        optimizer_type=self.cfg.FCTM.TRAIN.OPTIMIZER.TYPE,
                                        momentum=self.cfg.FCTM.TRAIN.OPTIMIZER.MOMENTUM,
                                        weight_decay=self.cfg.FCTM.TRAIN.OPTIMIZER.WEIGHT_DECAY)
        scheduler = self.build_scheduler(optimizer, lr_type=self.cfg.FCTM.TRAIN.LR_SCHEDULER.TYPE,
                                         lr_step=self.cfg.FCTM.TRAIN.LR_SCHEDULER.LR_STEP,
                                         lr_factor=self.cfg.FCTM.TRAIN.LR_SCHEDULER.LR_FACTOR,
                                         warmup_epochs=self.cfg.FCTM.TRAIN.LR_SCHEDULER.WARM_EPOCH,
                                         MAX_EPOCH=self.cfg.FCTM.TRAIN.MAX_EPOCH)
        MAX_EPOCH = self.cfg.FCTM.TRAIN.MAX_EPOCH
        best_acc = 0

        all_cls_criterion = CrossEntropy().cuda()
        add_cls_criterion = CrossEntropy().cuda()

        train_loader = DataLoader(dataset=train_dataset, batch_size=self.cfg.FCTM.TRAIN.BATCH_SIZE,
                                  num_workers=self.cfg.FCTM.TRAIN.NUM_WORKERS, shuffle=True, drop_last=True,
                                  persistent_workers=True)
        iter_num = len(train_loader)
        for epoch in range(1, MAX_EPOCH + 1):
            all_loss = AverageMeter()
            acc = AverageMeter()
            iter_index = 0
            if float(torch.__version__[:3]) < 1.3:
                scheduler.step()
            for x, y in train_loader:
                # Update # iters left on current data-loader(s) and, if needed, create new one(s)
                x, y = x.to(self.device), y.to(self.device)  # --> transfer them to correct device
                # ---> Train MAIN MODEL
                cnt = y.shape[0]
                x_replay, y_replay, pre_logits_replay = self.sample(self.previous_teacher, len(x), dpt_classes_num)
                all_x = torch.cat([x, x_replay], dim=0)
                all_y = torch.cat([y, y_replay], dim=0)
                loss, now_acc, now_cnt = self.FCTM_train_a_batch_datafree(optimizer, all_cls_criterion,
                                                                          add_cls_criterion, all_x, all_y,
                                                                          active_classes_num)
                all_loss.update(loss.data.item(), cnt)
                acc.update(now_acc, y.shape[0])
                if iter_index % self.cfg.SHOW_STEP == 0:
                    pbar_str = "FCTM train, Epoch: {} || Batch:{:>3d}/{} || lr : {} || Loss:{:>5.3f} || " \
                               "Accuracy:{:>5.2f}".format(epoch,
                                                          iter_index,
                                                          iter_num,
                                                          optimizer.param_groups[
                                                              0]['lr'],
                                                          all_loss.val,
                                                          acc.val * 100,
                                                          )

                    self.logger.info(pbar_str)
                iter_index += 1

            if self.cfg.VALID_STEP != -1 and epoch % self.cfg.VALID_STEP == 0:

                val_acc = self.validate_with_FCTM(task)  # task_id 从1开始
                acc_avg = val_acc.mean()
                self.logger.info(
                    "--------------FCTM train Epoch:{:>3d}    FCTM val_Acc:{:>5.2f}%--------------".format(
                        epoch, acc_avg * 100
                    )
                )
                if acc_avg > best_acc:
                    best_acc, best_epoch = acc_avg, epoch
                    self.best_epoch = best_epoch
                    self.best_acc = best_acc
                    self.logger.info(
                        "--------------FCTM train Best_Epoch:{:>3d}   FCTM Best_Acc:{:>5.2f}%--------------".format(
                            best_epoch, best_acc * 100
                        )
                    )

            if float(torch.__version__[:3]) >= 1.3:
                scheduler.step()
        pass

    def RDFCL_train_FCTM(self, train_dataset, active_classes_num, task_id):
        dpt_classes_num = active_classes_num - self.dataset_handler.classes_per_task
        if self.FCTM is None:
            self.FCTM = FCTM_model(self.cfg, backbone_model=copy.deepcopy(self.model))
            self.FCTM = self.transfer_to_cuda(self.FCTM)
            print("FCTM:", self.FCTM)
        model_plus_rkd = nn.ModuleList([self.FCTM, self.rkd])
        alpha = log(self.dataset_handler.classes_per_task / 2 + 1, 2)
        beta2 = 1.0 * dpt_classes_num / self.dataset_handler.classes_per_task
        beta = sqrt(beta2)
        optimizer = self.build_optimize(model=model_plus_rkd,
                                        base_lr=self.cfg.FCTM.TRAIN.OPTIMIZER.BASE_LR,
                                        optimizer_type=self.cfg.FCTM.TRAIN.OPTIMIZER.TYPE,
                                        momentum=self.cfg.FCTM.TRAIN.OPTIMIZER.MOMENTUM,
                                        weight_decay=self.cfg.FCTM.TRAIN.OPTIMIZER.WEIGHT_DECAY)
        scheduler = self.build_scheduler(optimizer, lr_type=self.cfg.FCTM.TRAIN.LR_SCHEDULER.TYPE,
                                         lr_step=self.cfg.FCTM.TRAIN.LR_SCHEDULER.LR_STEP,
                                         lr_factor=self.cfg.FCTM.TRAIN.LR_SCHEDULER.LR_FACTOR,
                                         warmup_epochs=self.cfg.FCTM.TRAIN.LR_SCHEDULER.WARM_EPOCH,
                                         MAX_EPOCH=self.cfg.FCTM.TRAIN.MAX_EPOCH)
        MAX_EPOCH = self.cfg.FCTM.TRAIN.MAX_EPOCH
        best_acc = 0

        all_cls_criterion = CrossEntropy().cuda()
        add_cls_criterion = CrossEntropy().cuda()

        train_loader = DataLoader(dataset=train_dataset, batch_size=self.cfg.FCTM.TRAIN.BATCH_SIZE,
                                  num_workers=self.cfg.FCTM.TRAIN.NUM_WORKERS, shuffle=True, drop_last=True,
                                  persistent_workers=True)
        iter_num = len(train_loader)
        for epoch in range(1, MAX_EPOCH + 1):
            all_loss = AverageMeter()
            acc = AverageMeter()
            iter_index = 0
            if float(torch.__version__[:3]) < 1.3:
                scheduler.step()
            for x, y in train_loader:
                self.FCTM.train()
                # Update # iters left on current data-loader(s) and, if needed, create new one(s)
                x, y = x.to(self.device), y.to(self.device)  # --> transfer them to correct device
                # ---> Train MAIN MODEL
                cnt = y.shape[0]
                x_replay, y_replay, pre_logits_replay = self.sample(self.previous_teacher, len(x), dpt_classes_num)

                pre_features_new_classes = self.previous_teacher.generate_features(x)
                FCTM_outputs = self.FCTM(x, pre_features_new_classes)
                logits_new_classes_for_ssce = FCTM_outputs["all_logits"][:, dpt_classes_num:active_classes_num]

                loss_class = all_cls_criterion(logits_new_classes_for_ssce, y % self.dataset_handler.classes_per_task) \
                             * (self.cfg.FCTM.ce_lambda * (1 + 1 / alpha) / beta)

                pre_features_old_classes = self.previous_teacher.generate_features(x_replay)
                FCTM_outputs_replay = self.FCTM(x_replay, pre_features_old_classes)

                logits_old_classes_for_distill = FCTM_outputs_replay["all_logits"][:, :dpt_classes_num]

                loss_hkd = (self.kd_criterion(logits_old_classes_for_distill, pre_logits_replay).sum(dim=1)).mean() / (
                    logits_old_classes_for_distill.size(1)) * self.cfg.FCTM.hkd_lambda

                loss_rkd = self.rkd(student=FCTM_outputs["features"],
                                    teacher=pre_features_new_classes) * self.cfg.FCTM.rkd_lambda
                loss_kd = (loss_hkd + loss_rkd) * (alpha * beta)

                all_logits = torch.cat([FCTM_outputs["all_logits"][:, :active_classes_num],
                                        FCTM_outputs_replay["all_logits"][:, :active_classes_num]], dim=0)
                all_y = torch.cat([y, y_replay], dim=0)
                add_cls_loss = add_cls_criterion(all_logits, all_y) * (self.cfg.FCTM.ce_lambda * (1 + 1 / alpha) / beta)

                total_loss = loss_class + loss_kd + add_cls_loss
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                all_loss.update(total_loss.data.item(), cnt)

                current_task_data_result = torch.argmax(all_logits, 1)
                # current_task_acc, current_task_cnt = cuda_accuracy(current_task_data_result.cpu().numpy(), labels.cpu().numpy())
                current_task_acc, current_task_cnt = cuda_accuracy(current_task_data_result, all_y)
                acc.update(current_task_acc, current_task_cnt)

                if iter_index % self.cfg.SHOW_STEP == 0:
                    pbar_str = "FCTM train, Epoch: {} || Batch:{:>3d}/{} || lr : {} || Loss:{:>5.3f} || " \
                               "Accuracy:{:>5.2f}".format(epoch,
                                                          iter_index,
                                                          iter_num,
                                                          optimizer.param_groups[
                                                              0]['lr'],
                                                          all_loss.val,
                                                          acc.val * 100,
                                                          )

                    self.logger.info(pbar_str)
                iter_index += 1

            if self.cfg.VALID_STEP != -1 and epoch % self.cfg.VALID_STEP == 0:

                val_acc = self.validate_with_FCTM(task_id)  # task_id 从1开始
                acc_avg = val_acc.mean()
                self.logger.info(
                    "--------------FCTM train Epoch:{:>3d}    FCTM val_Acc:{:>5.2f}%--------------".format(
                        epoch, acc_avg * 100
                    )
                )
                if acc_avg > best_acc:
                    best_acc, best_epoch = acc_avg, epoch
                    self.best_epoch = best_epoch
                    self.best_acc = best_acc
                    self.logger.info(
                        "--------------FCTM train Best_Epoch:{:>3d}   FCTM Best_Acc:{:>5.2f}%--------------".format(
                            best_epoch, best_acc * 100
                        )
                    )

            if float(torch.__version__[:3]) >= 1.3:
                scheduler.step()
        pass

    def train_model_no_dataset(self, train_dataset, active_classes_num, task):
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
        best_acc = 0.
        cls_criterion = CrossEntropy().cuda()
        sample_num_grade_classes = torch.from_numpy(
            np.array([active_classes_num - self.dataset_handler.classes_per_task,
                      self.dataset_handler.classes_per_task])).long().to(self.device)
        dw_kd = torch.ones(sample_num_grade_classes.size(), dtype=torch.float32).cuda()
        dw_kd[:-1] = 1.
        dpt_classes_num = active_classes_num - self.dataset_handler.classes_per_task
        dw_kd[-1] = self.dataset_handler.classes_per_task / dpt_classes_num
        rnt = 1.0 * dpt_classes_num / active_classes_num
        for epoch in range(1, self.cfg.model.TRAIN.MAX_EPOCH + 1):
            all_loss = AverageMeter()
            if float(torch.__version__[:3]) < 1.3:
                scheduler.step()
            for iter_index in range(1, iter_num + 1):
                self.model.train()
                self.FCTM.eval()
                # ---> Train MAIN MODEL
                # data replay
                x_replay, y_replay = self.fctm_teacher.sample(
                    batch_size=self.cfg.model.TRAIN.BATCH_SIZE,
                    classes_num=active_classes_num,
                    return_scores=False)
                # sample_num_per_class  =
                # per_class_weights = self.compute_weight_per_class(sample_num_per_class, beta=0.96)

                all_outputs, all_features = self.model(x_replay)
                all_outputs = all_outputs[:, :active_classes_num]
                fctm_all_output, pre_all_features = self.fctm_teacher.generate_features_scores(
                    x_replay)  # 获取classifier_output

                FCTM_outputs = fctm_all_output[:, 0:active_classes_num]

                if "ReKD" == self.cfg.model.KD_type:
                    loss_kd = compute_ReKD_loss(all_outputs,
                                                FCTM_outputs,
                                                sample_num_grade_classes,
                                                temp=self.cfg.model.T,
                                                dw_kd=dw_kd) * self.cfg.model.kd_lambda  # todo Done
                elif "TKD" == self.cfg.model.KD_type:
                    loss_KD = torch.zeros(task - 1).cuda()
                    all_ouput_for_dis = all_outputs[:, 0:dpt_classes_num]
                    all_pre_model_output_for_dis = FCTM_outputs[:, 0:dpt_classes_num]
                    for task_item in range(task - 1):
                        task_id_ouput = all_ouput_for_dis[:, self.dataset_handler.classes_per_task * task_item:
                                                             self.dataset_handler.classes_per_task * (task_item + 1)]
                        task_id_pre_model_output = all_pre_model_output_for_dis[:,
                                                   self.dataset_handler.classes_per_task * task_item:
                                                   self.dataset_handler.classes_per_task * (task_item + 1)]
                        soft_target = Func.softmax(task_id_pre_model_output / self.cfg.model.T,
                                                   dim=1)
                        output_log = Func.log_softmax(task_id_ouput / self.cfg.model.T, dim=1)
                        loss_KD[task_item] = Func.kl_div(output_log, soft_target, reduction='batchmean') * (
                                self.cfg.model.T ** 2)
                    loss_kd = loss_KD.sum() * self.cfg.model.kd_lambda
                elif "RTKD" == self.cfg.model.KD_type:
                    loss_KD = torch.zeros(task).cuda()
                    all_ouput_for_dis = all_outputs
                    all_pre_model_output_for_dis = FCTM_outputs
                    for task_item in range(task):
                        task_id_ouput = all_ouput_for_dis[:, self.dataset_handler.classes_per_task * task_item:
                                                             self.dataset_handler.classes_per_task * (task_item + 1)]
                        task_id_pre_model_output = all_pre_model_output_for_dis[:,
                                                   self.dataset_handler.classes_per_task * task_item:
                                                   self.dataset_handler.classes_per_task * (task_item + 1)]
                        soft_target = Func.softmax(task_id_pre_model_output / self.cfg.model.T,
                                                   dim=1)
                        output_log = Func.log_softmax(task_id_ouput / self.cfg.model.T, dim=1)
                        loss_KD[task_item] = Func.kl_div(output_log, soft_target, reduction='batchmean') * (
                                self.cfg.model.T ** 2)
                    loss_kd = loss_KD.sum() * self.cfg.model.kd_lambda
                # elif "BKD" == self.cfg.model.KD_type:
                #     all_ouput_for_dis = all_outputs
                #     all_pre_model_output_for_dis = FCTM_outputs
                #     loss_kd = BKD(pred=all_ouput_for_dis, soft=all_pre_model_output_for_dis,
                #                   per_cls_weights=per_class_weights, T=self.cfg.model.T)
                elif "null" == self.cfg.model.KD_type:
                    loss_kd = 0.
                if self.cfg.model.use_featureKD:
                    # loss_kd += self.cfg.model.fkd_lambda * (self.kd_criterion(all_features,
                    #                                                           FCTM_features).mean(dim=1)).mean()
                    # logits_KD = self.previous_teacher.generate_score_by_features(all_features,
                    #                                                              active_classes_num=dpt_classes_num,
                    #                                                              is_nograd=False)
                    self.FCTM.zero_grad()
                    logits_KD = self.FCTM(x=all_features, pre_model_feature=None, train_cls_use_features=True)[
                        "all_logits"]
                    logits_KD = logits_KD[:, :active_classes_num]
                    logits_KD_past = FCTM_outputs
                    # # loss_kd += self.cfg.FCTM.kd_lambda * (
                    # #     self.kd_criterion(logits_KD, logits_KD_past).sum(dim=1)).mean() / (
                    # #               logits_KD.size(1))
                    loss_kd += self.cfg.model.fkd_lambda * (
                        self.kd_criterion(logits_KD, logits_KD_past).sum(dim=1)).mean() / (
                                   logits_KD.size(1))
                    # loss_kd += 0.5 * (
                    #     self.kd_criterion(logits_KD, logits_KD_past).sum(dim=1)).mean() / (
                    #                logits_KD.size(1))

                if self.cfg.model.use_cls_loss:
                    if "LwF" == self.cfg.model.cls_type:
                        loss_cls = cls_criterion(all_outputs[:self.cfg.model.TRAIN.BATCH_SIZE,
                                                 -self.dataset_handler.classes_per_task:],
                                                 y_replay[:self.cfg.model.TRAIN.BATCH_SIZE] %
                                                 self.dataset_handler.classes_per_task) * (1 - rnt)

                        FCTM_output_for_pre_distill = FCTM_outputs[:, 0:dpt_classes_num]
                        loss_cls += compute_distill_loss(all_outputs[:, 0:dpt_classes_num], FCTM_output_for_pre_distill,
                                                         temp=self.cfg.model.T) * rnt

                        pass
                    elif "iCaRL" == self.cfg.model.cls_type:
                        FCTM_output_for_pre_distill = FCTM_outputs[:, 0:dpt_classes_num]
                        loss_cls = compute_cls_distill_binary_loss(y_replay, all_outputs,
                                                                   self.dataset_handler.classes_per_task,
                                                                   pre_model_output_for_distill=FCTM_output_for_pre_distill)

                        pass


                    else:
                        raise ValueError(f"self.cfg.model.cls_type is illegal.")
                else:
                    loss_cls = 0.

                total_loss = loss_cls + loss_kd
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                # measure accuracy and record loss
                all_loss.update(total_loss.data.item(), y_replay.shape[0])
                if iter_index % self.cfg.SHOW_STEP == 0:
                    pbar_str = "Approach: {}| | Epoch: {} || Batch:{:>3d}/{}|| lr: {} " \
                               "|| Batch_Loss:{:>5.3f}".format("EARS_DFCL", epoch, iter_index,
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

    def train_model(self, train_dataset, active_classes_num, task):
        if hasattr(self.FCTM, "module"):
            global_classifier_weights, global_classifier_bias = \
                get_classifier_weight_bias(self.FCTM.module.global_fc)
        else:
            global_classifier_weights, global_classifier_bias = \
                get_classifier_weight_bias(self.FCTM.global_fc)
        # global_classifier_paras = torch.cat((global_classifier_weights, global_classifier_bias), 1)
        global_classifier_weights_norm = torch.norm(global_classifier_weights.detach(), 2, 1)
        global_classifier_weights_norm = global_classifier_weights_norm[0: active_classes_num]
        # global_classifier_paras_norm = torch.norm(global_classifier_paras.detach(), 2, 1)
        dpt_classes_num = active_classes_num - self.dataset_handler.classes_per_task
        self.logger.info(f"global_classifier_weights_norm: {global_classifier_weights_norm}")
        self.logger.info(f"global_classifier_weights_norm mean of old classes: "
                         f"{torch.mean(global_classifier_weights_norm[0:dpt_classes_num])}")
        self.logger.info(f"global_classifier_weights_norm mean of new classes: "
                         f"{torch.mean(global_classifier_weights_norm[dpt_classes_num: active_classes_num])}")
        class_weight_from_feature_norm = global_classifier_weights_norm.sum() / global_classifier_weights_norm
        sample_num_grade_classes = torch.from_numpy(
            np.array([active_classes_num - self.dataset_handler.classes_per_task,
                      self.dataset_handler.classes_per_task])).long().to(self.device)

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
        best_acc = 0.
        cls_criterion = CrossEntropy().cuda()
        dw_kd = torch.ones(sample_num_grade_classes.size(), dtype=torch.float32).cuda()
        rnt = 1.0 * dpt_classes_num / active_classes_num
        dw_kd[:-1] = 1.
        dw_kd[-1] = self.dataset_handler.classes_per_task / dpt_classes_num
        mappings = torch.ones(int(2 * self.cfg.model.TRAIN.BATCH_SIZE), dtype=torch.float32).cuda()
        mappings[:self.cfg.model.TRAIN.BATCH_SIZE] = 1 - rnt
        mappings[self.cfg.model.TRAIN.BATCH_SIZE:] = rnt
        dw_cls = mappings

        sample_num_per_class = torch.ones(active_classes_num, dtype=torch.float32)
        sample_num_per_class[
        :dpt_classes_num] = self.cfg.model.TRAIN.BATCH_SIZE / dpt_classes_num
        sample_num_per_class[dpt_classes_num:] = self.cfg.model.TRAIN.BATCH_SIZE / self.dataset_handler.classes_per_task
        per_class_weights = self.compute_weight_per_class(sample_num_per_class, beta=0.96)
        for epoch in range(1, self.cfg.model.TRAIN.MAX_EPOCH + 1):
            all_loss = AverageMeter()
            if float(torch.__version__[:3]) < 1.3:
                scheduler.step()
            iter_index = 0
            for x, y in train_loader:
                self.model.train()
                self.FCTM.eval()
                x, y = x.to(self.device), y.to(self.device)  # --> transfer them to correct device
                # ---> Train MAIN MODEL
                # data replay
                x_replay, y_replay, pre_logits_replay = self.sample(self.previous_teacher, len(x), dpt_classes_num)

                all_x = torch.cat([x, x_replay], dim=0)
                all_y = torch.cat([y, y_replay], dim=0)
                assert all_y.shape[0] == 2 * self.cfg.model.TRAIN.BATCH_SIZE

                all_outputs, all_features = self.model(all_x)
                all_outputs = all_outputs[:, :active_classes_num]
                pre_all_output, pre_all_features = self.previous_teacher.generate_features_scores(
                    all_x)  # 获取classifier_output
                pre_model_output_for_distill = pre_all_output[:, 0:dpt_classes_num]
                with torch.no_grad():
                    FCTM_outs = self.FCTM(all_x, pre_all_features)
                FCTM_outputs = FCTM_outs["all_logits"][:, 0:active_classes_num]
                FCTM_features = FCTM_outs["features"]

                if "ReKD" == self.cfg.model.KD_type:
                    loss_kd = compute_ReKD_loss(all_outputs,
                                                FCTM_outputs,
                                                sample_num_grade_classes,
                                                temp=self.cfg.model.T,
                                                dw_kd=dw_kd) * self.cfg.model.kd_lambda  # todo Done
                elif "TKD" == self.cfg.model.KD_type:
                    loss_KD = torch.zeros(task - 1).cuda()
                    all_ouput_for_dis = all_outputs[:, 0:dpt_classes_num]
                    all_pre_model_output_for_dis = FCTM_outputs[:, 0:dpt_classes_num]
                    for task_item in range(task - 1):
                        task_id_ouput = all_ouput_for_dis[:, self.dataset_handler.classes_per_task * task_item:
                                                             self.dataset_handler.classes_per_task * (task_item + 1)]
                        task_id_pre_model_output = all_pre_model_output_for_dis[:,
                                                   self.dataset_handler.classes_per_task * task_item:
                                                   self.dataset_handler.classes_per_task * (task_item + 1)]
                        soft_target = Func.softmax(task_id_pre_model_output / self.cfg.model.T,
                                                   dim=1)
                        output_log = Func.log_softmax(task_id_ouput / self.cfg.model.T, dim=1)
                        loss_KD[task_item] = Func.kl_div(output_log, soft_target, reduction='batchmean') * (
                                self.cfg.model.T ** 2)
                    loss_kd = loss_KD.sum() * self.cfg.model.kd_lambda
                elif "RTKD" == self.cfg.model.KD_type:
                    loss_KD = torch.zeros(task).cuda()
                    all_ouput_for_dis = all_outputs
                    all_pre_model_output_for_dis = FCTM_outputs
                    for task_item in range(task):
                        task_id_ouput = all_ouput_for_dis[:, self.dataset_handler.classes_per_task * task_item:
                                                             self.dataset_handler.classes_per_task * (task_item + 1)]
                        task_id_pre_model_output = all_pre_model_output_for_dis[:,
                                                   self.dataset_handler.classes_per_task * task_item:
                                                   self.dataset_handler.classes_per_task * (task_item + 1)]
                        soft_target = Func.softmax(task_id_pre_model_output / self.cfg.model.T,
                                                   dim=1)
                        output_log = Func.log_softmax(task_id_ouput / self.cfg.model.T, dim=1)
                        loss_KD[task_item] = Func.kl_div(output_log, soft_target, reduction='batchmean') * (
                                self.cfg.model.T ** 2)
                    loss_kd = loss_KD.sum() * self.cfg.model.kd_lambda
                elif "BKD" == self.cfg.model.KD_type:
                    all_ouput_for_dis = all_outputs
                    all_pre_model_output_for_dis = FCTM_outputs
                    loss_kd = BKD(pred=all_ouput_for_dis, soft=all_pre_model_output_for_dis,
                                  per_cls_weights=per_class_weights, T=self.cfg.model.T)
                elif "null" == self.cfg.model.KD_type:
                    loss_kd = 0.
                if self.cfg.model.use_featureKD:
                    # loss_kd += self.cfg.model.fkd_lambda * (self.kd_criterion(all_features,
                    #                                                           FCTM_features).mean(dim=1)).mean()
                    # logits_KD = self.previous_teacher.generate_score_by_features(all_features,
                    #                                                              active_classes_num=dpt_classes_num,
                    #                                                              is_nograd=False)
                    self.FCTM.zero_grad()
                    logits_KD = self.FCTM(x=all_features, pre_model_feature=None, train_cls_use_features=True)[
                        "all_logits"]
                    logits_KD = logits_KD[:, :active_classes_num]
                    logits_KD_past = FCTM_outputs
                    # # loss_kd += self.cfg.FCTM.kd_lambda * (
                    # #     self.kd_criterion(logits_KD, logits_KD_past).sum(dim=1)).mean() / (
                    # #               logits_KD.size(1))
                    loss_kd += self.cfg.model.fkd_lambda * (
                        self.kd_criterion(logits_KD, logits_KD_past).sum(dim=1)).mean() / (
                                   logits_KD.size(1))
                    # loss_kd += 0.5 * (
                    #     self.kd_criterion(logits_KD, logits_KD_past).sum(dim=1)).mean() / (
                    #                logits_KD.size(1))

                if self.cfg.model.use_cls_loss:
                    if "softTarget" == self.cfg.model.cls_type:
                        loss_cls = reweight_KD(all_outputs, FCTM_outputs, all_y,
                                               class_weight_from_feature_norm, temp=self.cfg.model.T)
                    elif "adjusted_logit" == self.cfg.model.cls_type:
                        loss_cls = adjusted_KD(all_outputs, FCTM_outputs, 1. / class_weight_from_feature_norm,
                                               temp=self.cfg.model.T)
                    elif "LwF" == self.cfg.model.cls_type:
                        loss_cls = cls_criterion(all_outputs[:self.cfg.model.TRAIN.BATCH_SIZE,
                                                 -self.dataset_handler.classes_per_task:],
                                                 all_y[:self.cfg.model.TRAIN.BATCH_SIZE] %
                                                 self.dataset_handler.classes_per_task) * (1 - rnt)

                        FCTM_output_for_pre_distill = FCTM_outputs[:, 0:dpt_classes_num]
                        loss_cls += compute_distill_loss(all_outputs[:, 0:dpt_classes_num], FCTM_output_for_pre_distill,
                                                         temp=self.cfg.model.T) * rnt

                        pass
                    elif "iCaRL" == self.cfg.model.cls_type:
                        FCTM_output_for_pre_distill = FCTM_outputs[:, 0:dpt_classes_num]
                        loss_cls = compute_cls_distill_binary_loss(all_y, all_outputs,
                                                                   self.dataset_handler.classes_per_task,
                                                                   pre_model_output_for_distill=FCTM_output_for_pre_distill)

                        pass

                    elif "FTCE" == self.cfg.model.cls_type:
                        # loss_cls = self.ft_criterion(all_outputs, all_y, dw_cls)
                        # feat_class = all_features.detach()
                        # ft_outputs = self.model(feat_class, train_cls_use_features=True)
                        # ft_logits = ft_outputs[:, 0:active_classes_num]
                        # loss_cls = self.ft_criterion(ft_logits, all_y, dw_cls)
                        ft_outputs = self.model(all_x, train_classifier=True)
                        ft_logits = ft_outputs[:, 0:active_classes_num]
                        loss_cls = self.ft_criterion(ft_logits, all_y, dw_cls)


                    else:
                        raise ValueError(f"self.cfg.model.cls_type is illegal.")
                else:
                    loss_cls = 0.

                total_loss = loss_cls + loss_kd
                if self.cfg.model.use_ABD_loss:
                    # ft_outputs = self.model(all_x, train_classifier=True)
                    # ft_logits = ft_outputs[:, 0:active_classes_num]
                    # total_loss += self.ft_criterion(ft_logits, all_y, dw_cls)

                    logits_KD = self.previous_teacher.generate_score_by_features(all_features,
                                                                                 active_classes_num=dpt_classes_num,
                                                                                 is_nograd=False)
                    logits_KD_past = pre_model_output_for_distill
                    ABD_loss = self.cfg.FCTM.kd_lambda * (
                        self.kd_criterion(logits_KD, logits_KD_past).sum(dim=1)).mean() / (
                                   logits_KD.size(1))
                    total_loss += ABD_loss
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                # measure accuracy and record loss
                all_loss.update(total_loss.data.item(), y.shape[0] + y_replay.shape[0])
                if iter_index % self.cfg.SHOW_STEP == 0:
                    pbar_str = "Approach: {}| | Epoch: {} || Batch:{:>3d}/{}|| lr: {} " \
                               "|| Batch_Loss:{:>5.3f}".format("EARS_DFCL", epoch, iter_index,
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

    def train_model_old_classes_generator(self, train_dataset, active_classes_num, task):
        if hasattr(self.FCTM, "module"):
            global_classifier_weights, global_classifier_bias = \
                get_classifier_weight_bias(self.FCTM.module.global_fc)
        else:
            global_classifier_weights, global_classifier_bias = \
                get_classifier_weight_bias(self.FCTM.global_fc)
        # global_classifier_paras = torch.cat((global_classifier_weights, global_classifier_bias), 1)
        global_classifier_weights_norm = torch.norm(global_classifier_weights.detach(), 2, 1)
        global_classifier_weights_norm = global_classifier_weights_norm[0: active_classes_num]
        # global_classifier_paras_norm = torch.norm(global_classifier_paras.detach(), 2, 1)
        dpt_classes_num = active_classes_num - self.dataset_handler.classes_per_task
        self.logger.info(f"global_classifier_weights_norm: {global_classifier_weights_norm}")
        self.logger.info(f"global_classifier_weights_norm mean of old classes: "
                         f"{torch.mean(global_classifier_weights_norm[0:dpt_classes_num])}")
        self.logger.info(f"global_classifier_weights_norm mean of new classes: "
                         f"{torch.mean(global_classifier_weights_norm[dpt_classes_num: active_classes_num])}")
        class_weight_from_feature_norm = global_classifier_weights_norm.sum() / global_classifier_weights_norm
        sample_num_grade_classes = torch.from_numpy(
            np.array([active_classes_num - self.dataset_handler.classes_per_task,
                      self.dataset_handler.classes_per_task])).long().to(self.device)

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
        best_acc = 0.
        cls_criterion = CrossEntropy().cuda()
        dw_kd = torch.ones(sample_num_grade_classes.size(), dtype=torch.float32).cuda()
        rnt = 1.0 * dpt_classes_num / active_classes_num
        dw_kd[:-1] = 1.
        dw_kd[-1] = self.dataset_handler.classes_per_task / dpt_classes_num
        mappings = torch.ones(int(2 * self.cfg.model.TRAIN.BATCH_SIZE), dtype=torch.float32).cuda()
        mappings[:self.cfg.model.TRAIN.BATCH_SIZE] = 1 - rnt
        mappings[self.cfg.model.TRAIN.BATCH_SIZE:] = rnt
        dw_cls = mappings

        sample_num_per_class = torch.ones(active_classes_num, dtype=torch.float32)
        sample_num_per_class[
        :dpt_classes_num] = self.cfg.model.TRAIN.BATCH_SIZE / dpt_classes_num
        sample_num_per_class[dpt_classes_num:] = self.cfg.model.TRAIN.BATCH_SIZE / self.dataset_handler.classes_per_task
        per_class_weights = self.compute_weight_per_class(sample_num_per_class, beta=0.96)

        condition_old_labels = construct_condition_labels(self.cfg.model.TRAIN.BATCH_SIZE, dpt_classes_num)
        for epoch in range(1, self.cfg.model.TRAIN.MAX_EPOCH + 1):
            all_loss = AverageMeter()
            if float(torch.__version__[:3]) < 1.3:
                scheduler.step()
            iter_index = 0
            for x, y in train_loader:
                self.model.train()
                self.FCTM.eval()
                x, y = x.to(self.device), y.to(self.device)  # --> transfer them to correct device
                # ---> Train MAIN MODEL
                # data replay
                x_replay, y_replay, pre_logits_replay = self.condition_sample(self.fctm_teacher,
                                                                              self.cfg.model.TRAIN.BATCH_SIZE,
                                                                              condition_old_labels)

                all_x = torch.cat([x, x_replay], dim=0)
                all_y = torch.cat([y, y_replay], dim=0)
                assert all_y.shape[0] == 2 * self.cfg.model.TRAIN.BATCH_SIZE

                all_outputs, all_features = self.model(all_x)
                all_outputs = all_outputs[:, :active_classes_num]
                pre_all_output, pre_all_features = self.previous_teacher.generate_features_scores(
                    all_x)  # 获取classifier_output
                pre_model_output_for_distill = pre_all_output[:, 0:dpt_classes_num]
                with torch.no_grad():
                    FCTM_outs = self.FCTM(all_x, pre_all_features)
                FCTM_outputs = FCTM_outs["all_logits"][:, 0:active_classes_num]
                FCTM_features = FCTM_outs["features"]

                if "ReKD" == self.cfg.model.KD_type:
                    loss_kd = compute_ReKD_loss(all_outputs,
                                                FCTM_outputs,
                                                sample_num_grade_classes,
                                                temp=self.cfg.model.T,
                                                dw_kd=dw_kd) * self.cfg.model.kd_lambda  # todo Done
                elif "TKD" == self.cfg.model.KD_type:
                    loss_KD = torch.zeros(task - 1).cuda()
                    all_ouput_for_dis = all_outputs[:, 0:dpt_classes_num]
                    all_pre_model_output_for_dis = FCTM_outputs[:, 0:dpt_classes_num]
                    for task_item in range(task - 1):
                        task_id_ouput = all_ouput_for_dis[:, self.dataset_handler.classes_per_task * task_item:
                                                             self.dataset_handler.classes_per_task * (task_item + 1)]
                        task_id_pre_model_output = all_pre_model_output_for_dis[:,
                                                   self.dataset_handler.classes_per_task * task_item:
                                                   self.dataset_handler.classes_per_task * (task_item + 1)]
                        soft_target = Func.softmax(task_id_pre_model_output / self.cfg.model.T,
                                                   dim=1)
                        output_log = Func.log_softmax(task_id_ouput / self.cfg.model.T, dim=1)
                        loss_KD[task_item] = Func.kl_div(output_log, soft_target, reduction='batchmean') * (
                                self.cfg.model.T ** 2)
                    loss_kd = loss_KD.sum() * self.cfg.model.kd_lambda
                elif "RTKD" == self.cfg.model.KD_type:
                    loss_KD = torch.zeros(task).cuda()
                    all_ouput_for_dis = all_outputs
                    all_pre_model_output_for_dis = FCTM_outputs
                    for task_item in range(task):
                        task_id_ouput = all_ouput_for_dis[:, self.dataset_handler.classes_per_task * task_item:
                                                             self.dataset_handler.classes_per_task * (task_item + 1)]
                        task_id_pre_model_output = all_pre_model_output_for_dis[:,
                                                   self.dataset_handler.classes_per_task * task_item:
                                                   self.dataset_handler.classes_per_task * (task_item + 1)]
                        soft_target = Func.softmax(task_id_pre_model_output / self.cfg.model.T,
                                                   dim=1)
                        output_log = Func.log_softmax(task_id_ouput / self.cfg.model.T, dim=1)
                        loss_KD[task_item] = Func.kl_div(output_log, soft_target, reduction='batchmean') * (
                                self.cfg.model.T ** 2)
                    loss_kd = loss_KD.sum() * self.cfg.model.kd_lambda
                elif "BKD" == self.cfg.model.KD_type:
                    all_ouput_for_dis = all_outputs
                    all_pre_model_output_for_dis = FCTM_outputs
                    loss_kd = BKD(pred=all_ouput_for_dis, soft=all_pre_model_output_for_dis,
                                  per_cls_weights=per_class_weights, T=self.cfg.model.T)
                elif "null" == self.cfg.model.KD_type:
                    loss_kd = 0.
                if self.cfg.model.use_featureKD:
                    # loss_kd += self.cfg.model.fkd_lambda * (self.kd_criterion(all_features,
                    #                                                           FCTM_features).mean(dim=1)).mean()
                    # logits_KD = self.previous_teacher.generate_score_by_features(all_features,
                    #                                                              active_classes_num=dpt_classes_num,
                    #                                                              is_nograd=False)
                    self.FCTM.zero_grad()
                    logits_KD = self.FCTM(x=all_features, pre_model_feature=None, train_cls_use_features=True)[
                        "all_logits"]
                    logits_KD = logits_KD[:, :active_classes_num]
                    logits_KD_past = FCTM_outputs
                    # # loss_kd += self.cfg.FCTM.kd_lambda * (
                    # #     self.kd_criterion(logits_KD, logits_KD_past).sum(dim=1)).mean() / (
                    # #               logits_KD.size(1))
                    loss_kd += self.cfg.model.fkd_lambda * (
                        self.kd_criterion(logits_KD, logits_KD_past).sum(dim=1)).mean() / (
                                   logits_KD.size(1))
                    # loss_kd += 0.5 * (
                    #     self.kd_criterion(logits_KD, logits_KD_past).sum(dim=1)).mean() / (
                    #                logits_KD.size(1))

                if self.cfg.model.use_cls_loss:
                    if "softTarget" == self.cfg.model.cls_type:
                        loss_cls = reweight_KD(all_outputs, FCTM_outputs, all_y,
                                               class_weight_from_feature_norm, temp=self.cfg.model.T)
                    elif "adjusted_logit" == self.cfg.model.cls_type:
                        loss_cls = adjusted_KD(all_outputs, FCTM_outputs, 1. / class_weight_from_feature_norm,
                                               temp=self.cfg.model.T)
                    elif "LwF" == self.cfg.model.cls_type:
                        loss_cls = cls_criterion(all_outputs[:self.cfg.model.TRAIN.BATCH_SIZE,
                                                 -self.dataset_handler.classes_per_task:],
                                                 all_y[:self.cfg.model.TRAIN.BATCH_SIZE] %
                                                 self.dataset_handler.classes_per_task) * (1 - rnt)

                        FCTM_output_for_pre_distill = FCTM_outputs[:, 0:dpt_classes_num]
                        loss_cls += compute_distill_loss(all_outputs[:, 0:dpt_classes_num], FCTM_output_for_pre_distill,
                                                         temp=self.cfg.model.T) * rnt

                        pass
                    elif "iCaRL" == self.cfg.model.cls_type:
                        FCTM_output_for_pre_distill = FCTM_outputs[:, 0:dpt_classes_num]
                        loss_cls = compute_cls_distill_binary_loss(all_y, all_outputs,
                                                                   self.dataset_handler.classes_per_task,
                                                                   pre_model_output_for_distill=FCTM_output_for_pre_distill)

                        pass

                    elif "FTCE" == self.cfg.model.cls_type:
                        # loss_cls = self.ft_criterion(all_outputs, all_y, dw_cls)
                        # feat_class = all_features.detach()
                        # ft_outputs = self.model(feat_class, train_cls_use_features=True)
                        # ft_logits = ft_outputs[:, 0:active_classes_num]
                        # loss_cls = self.ft_criterion(ft_logits, all_y, dw_cls)
                        ft_outputs = self.model(all_x, train_classifier=True)
                        ft_logits = ft_outputs[:, 0:active_classes_num]
                        loss_cls = self.ft_criterion(ft_logits, all_y, dw_cls)


                    else:
                        raise ValueError(f"self.cfg.model.cls_type is illegal.")
                else:
                    loss_cls = 0.

                total_loss = loss_cls + loss_kd
                if self.cfg.model.use_ABD_loss:
                    # ft_outputs = self.model(all_x, train_classifier=True)
                    # ft_logits = ft_outputs[:, 0:active_classes_num]
                    # total_loss += self.ft_criterion(ft_logits, all_y, dw_cls)

                    logits_KD = self.previous_teacher.generate_score_by_features(all_features,
                                                                                 active_classes_num=dpt_classes_num,
                                                                                 is_nograd=False)
                    # logits_KD = self.fctm_teacher.generate_score_by_features(features=all_features,
                    #                                                          active_classes_num=active_classes_num,
                    #                                                          is_nograd=False)
                    logits_KD_past = pre_model_output_for_distill
                    # logits_KD_past = FCTM_outputs
                    ABD_loss = self.cfg.FCTM.kd_lambda * (
                        self.kd_criterion(logits_KD, logits_KD_past).sum(dim=1)).mean() / (
                                   logits_KD.size(1))
                    total_loss += ABD_loss
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                # measure accuracy and record loss
                all_loss.update(total_loss.data.item(), y.shape[0] + y_replay.shape[0])
                if iter_index % self.cfg.SHOW_STEP == 0:
                    pbar_str = "Approach: {}| | Epoch: {} || Batch:{:>3d}/{}|| lr: {} " \
                               "|| Batch_Loss:{:>5.3f}".format("EARS_DFCL", epoch, iter_index,
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

    def train_model_factory(self, train_dataset, active_classes_num, task):
        if hasattr(self.FCTM, "module"):
            global_classifier_weights, global_classifier_bias = \
                get_classifier_weight_bias(self.FCTM.module.global_fc)
        else:
            global_classifier_weights, global_classifier_bias = \
                get_classifier_weight_bias(self.FCTM.global_fc)
        # global_classifier_paras = torch.cat((global_classifier_weights, global_classifier_bias), 1)
        global_classifier_weights_norm = torch.norm(global_classifier_weights.detach(), 2, 1)
        global_classifier_weights_norm = global_classifier_weights_norm[0: active_classes_num]
        # global_classifier_paras_norm = torch.norm(global_classifier_paras.detach(), 2, 1)
        dpt_classes_num = active_classes_num - self.dataset_handler.classes_per_task
        self.logger.info(f"global_classifier_weights_norm: {global_classifier_weights_norm}")
        self.logger.info(f"global_classifier_weights_norm mean of old classes: "
                         f"{torch.mean(global_classifier_weights_norm[0:dpt_classes_num])}")
        self.logger.info(f"global_classifier_weights_norm mean of new classes: "
                         f"{torch.mean(global_classifier_weights_norm[dpt_classes_num: active_classes_num])}")
        class_weight_from_feature_norm = global_classifier_weights_norm.sum() / global_classifier_weights_norm
        sample_num_grade_classes = torch.from_numpy(
            np.array([active_classes_num - self.dataset_handler.classes_per_task,
                      self.dataset_handler.classes_per_task])).long().to(self.device)

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
        best_acc = 0.
        cls_criterion = CrossEntropy().cuda()
        dw_kd = torch.ones(sample_num_grade_classes.size(), dtype=torch.float32).cuda()
        rnt = 1.0 * dpt_classes_num / active_classes_num
        dw_kd[:-1] = 1.
        dw_kd[-1] = self.dataset_handler.classes_per_task / dpt_classes_num
        mappings = torch.ones(int(2 * self.cfg.model.TRAIN.BATCH_SIZE), dtype=torch.float32).cuda()
        mappings[:self.cfg.model.TRAIN.BATCH_SIZE] = 1 - rnt
        mappings[self.cfg.model.TRAIN.BATCH_SIZE:] = rnt
        dw_cls = mappings

        sample_num_per_class = torch.ones(active_classes_num, dtype=torch.float32)
        sample_num_per_class[
        :dpt_classes_num] = self.cfg.model.TRAIN.BATCH_SIZE / dpt_classes_num
        sample_num_per_class[dpt_classes_num:] = self.cfg.model.TRAIN.BATCH_SIZE / self.dataset_handler.classes_per_task
        per_class_weights = self.compute_weight_per_class(sample_num_per_class, beta=0.96)

        alpha = log(self.dataset_handler.classes_per_task / 2 + 1, 2)
        beta2 = 1.0 * dpt_classes_num / self.dataset_handler.classes_per_task
        beta = sqrt(beta2)
        for epoch in range(1, self.cfg.model.TRAIN.MAX_EPOCH + 1):
            all_loss = AverageMeter()
            if float(torch.__version__[:3]) < 1.3:
                scheduler.step()
            iter_index = 0
            for x, y in train_loader:
                self.model.train()
                self.FCTM.eval()
                x, y = x.to(self.device), y.to(self.device)  # --> transfer them to correct device
                # ---> Train MAIN MODEL
                # data replay
                x_replay, y_replay, pre_logits_replay = self.sample(self.previous_teacher, len(x), dpt_classes_num)

                all_x = torch.cat([x, x_replay], dim=0)
                all_y = torch.cat([y, y_replay], dim=0)
                assert all_y.shape[0] == 2 * self.cfg.model.TRAIN.BATCH_SIZE

                all_outputs, all_features = self.model(all_x)
                all_outputs = all_outputs[:, :active_classes_num]
                pre_all_output, pre_all_features = self.previous_teacher.generate_features_scores(
                    all_x)  # 获取classifier_output
                pre_all_output = pre_all_output[:, 0:dpt_classes_num]

                frozen_features = self.model(all_x, is_nograd=True, feature_flag=True)

                logits_for_frozen_features = self.model(frozen_features, train_cls_use_features=True)[:,
                                             0:active_classes_num]

                with torch.no_grad():
                    FCTM_outs = self.FCTM(all_x, pre_all_features)
                FCTM_outputs = FCTM_outs["all_logits"][:, 0:active_classes_num]
                FCTM_features = FCTM_outs["features"]
                self.FCTM.zero_grad()
                fctm_logits_for_all_features = self.FCTM(x=all_features, pre_model_feature=None,
                                                         train_cls_use_features=True)["all_logits"][:,
                                               :active_classes_num]

                # total_loss = self.ABD_loss(all_y, all_outputs, logits_for_frozen_features, fctm_logits_for_all_features,
                #                            FCTM_outputs, cls_criterion, active_classes_num, dpt_classes_num, dw_cls)

                # total_loss = self.LwF_loss(all_y, all_outputs, FCTM_outputs, cls_criterion, active_classes_num,
                #                            dpt_classes_num)

                total_loss = self.RDFCIL_loss(y=all_y, logits_com=all_outputs, pre_logits_replay=FCTM_outputs,
                                              features_new_classes=all_features[0:self.cfg.model.TRAIN.BATCH_SIZE, :],
                                              pre_features_new_classes=FCTM_features[0:self.cfg.model.TRAIN.BATCH_SIZE,
                                                                       :],
                                              cls_criterion=cls_criterion, dp_classes_num=dpt_classes_num,
                                              active_classes_num=active_classes_num,
                                              alpha=alpha, beta=beta)

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                # measure accuracy and record loss
                all_loss.update(total_loss.data.item(), y.shape[0] + y_replay.shape[0])
                if iter_index % self.cfg.SHOW_STEP == 0:
                    pbar_str = "Approach: {}| | Epoch: {} || Batch:{:>3d}/{}|| lr: {} " \
                               "|| Batch_Loss:{:>5.3f}".format("EARS_DFCL", epoch, iter_index,
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

    def ERA_loss(self, all_x, all_y, all_outputs, all_features, FCTM_outputs, pre_model_output_for_distill,
                 active_classes_num, dpt_classes_num, task, cls_criterion, sample_num_grade_classes=None, dw_cls=None,
                 dw_kd=None, per_class_weights=None, class_weight_from_feature_norm=None):
        rnt = dpt_classes_num / active_classes_num
        if "ReKD" == self.cfg.model.KD_type:
            loss_kd = compute_ReKD_loss(all_outputs,
                                        FCTM_outputs,
                                        sample_num_grade_classes,
                                        temp=self.cfg.model.T,
                                        dw_kd=dw_kd) * self.cfg.model.kd_lambda  # todo Done
        elif "TKD" == self.cfg.model.KD_type:
            loss_KD = torch.zeros(task - 1).cuda()
            all_ouput_for_dis = all_outputs[:, 0:dpt_classes_num]
            all_pre_model_output_for_dis = FCTM_outputs[:, 0:dpt_classes_num]
            for task_item in range(task - 1):
                task_id_ouput = all_ouput_for_dis[:, self.dataset_handler.classes_per_task * task_item:
                                                     self.dataset_handler.classes_per_task * (task_item + 1)]
                task_id_pre_model_output = all_pre_model_output_for_dis[:,
                                           self.dataset_handler.classes_per_task * task_item:
                                           self.dataset_handler.classes_per_task * (task_item + 1)]
                soft_target = Func.softmax(task_id_pre_model_output / self.cfg.model.T,
                                           dim=1)
                output_log = Func.log_softmax(task_id_ouput / self.cfg.model.T, dim=1)
                loss_KD[task_item] = Func.kl_div(output_log, soft_target, reduction='batchmean') * (
                        self.cfg.model.T ** 2)
            loss_kd = loss_KD.sum() * self.cfg.model.kd_lambda
        elif "RTKD" == self.cfg.model.KD_type:
            loss_KD = torch.zeros(task).cuda()
            all_ouput_for_dis = all_outputs
            all_pre_model_output_for_dis = FCTM_outputs
            for task_item in range(task):
                task_id_ouput = all_ouput_for_dis[:, self.dataset_handler.classes_per_task * task_item:
                                                     self.dataset_handler.classes_per_task * (task_item + 1)]
                task_id_pre_model_output = all_pre_model_output_for_dis[:,
                                           self.dataset_handler.classes_per_task * task_item:
                                           self.dataset_handler.classes_per_task * (task_item + 1)]
                soft_target = Func.softmax(task_id_pre_model_output / self.cfg.model.T,
                                           dim=1)
                output_log = Func.log_softmax(task_id_ouput / self.cfg.model.T, dim=1)
                loss_KD[task_item] = Func.kl_div(output_log, soft_target, reduction='batchmean') * (
                        self.cfg.model.T ** 2)
            loss_kd = loss_KD.sum() * self.cfg.model.kd_lambda
        elif "BKD" == self.cfg.model.KD_type:
            all_ouput_for_dis = all_outputs
            all_pre_model_output_for_dis = FCTM_outputs
            loss_kd = BKD(pred=all_ouput_for_dis, soft=all_pre_model_output_for_dis,
                          per_cls_weights=per_class_weights, T=self.cfg.model.T)
        elif "null" == self.cfg.model.KD_type:
            loss_kd = 0.
        if self.cfg.model.use_featureKD:
            # loss_kd += self.cfg.model.fkd_lambda * (self.kd_criterion(all_features,
            #                                                           FCTM_features).mean(dim=1)).mean()
            # logits_KD = self.previous_teacher.generate_score_by_features(all_features,
            #                                                              active_classes_num=dpt_classes_num,
            #                                                              is_nograd=False)
            self.FCTM.zero_grad()
            logits_KD = self.FCTM(x=all_features, pre_model_feature=None, train_cls_use_features=True)[
                "all_logits"]
            logits_KD = logits_KD[:, :active_classes_num]
            logits_KD_past = FCTM_outputs
            # # loss_kd += self.cfg.FCTM.kd_lambda * (
            # #     self.kd_criterion(logits_KD, logits_KD_past).sum(dim=1)).mean() / (
            # #               logits_KD.size(1))
            loss_kd += self.cfg.model.fkd_lambda * (
                self.kd_criterion(logits_KD, logits_KD_past).sum(dim=1)).mean() / (
                           logits_KD.size(1))
            # loss_kd += 0.5 * (
            #     self.kd_criterion(logits_KD, logits_KD_past).sum(dim=1)).mean() / (
            #                logits_KD.size(1))

        if self.cfg.model.use_cls_loss:
            if "softTarget" == self.cfg.model.cls_type:
                loss_cls = reweight_KD(all_outputs, FCTM_outputs, all_y,
                                       class_weight_from_feature_norm, temp=self.cfg.model.T)
            elif "adjusted_logit" == self.cfg.model.cls_type:
                loss_cls = adjusted_KD(all_outputs, FCTM_outputs, 1. / class_weight_from_feature_norm,
                                       temp=self.cfg.model.T)
            elif "LwF" == self.cfg.model.cls_type:
                loss_cls = cls_criterion(all_outputs[:self.cfg.model.TRAIN.BATCH_SIZE,
                                         -self.dataset_handler.classes_per_task:],
                                         all_y[:self.cfg.model.TRAIN.BATCH_SIZE] %
                                         self.dataset_handler.classes_per_task) * (1 - rnt)

                FCTM_output_for_pre_distill = FCTM_outputs[:, 0:dpt_classes_num]
                loss_cls += compute_distill_loss(all_outputs[:, 0:dpt_classes_num], FCTM_output_for_pre_distill,
                                                 temp=self.cfg.model.T) * rnt

                pass
            elif "iCaRL" == self.cfg.model.cls_type:
                FCTM_output_for_pre_distill = FCTM_outputs[:, 0:dpt_classes_num]
                loss_cls = compute_cls_distill_binary_loss(all_y, all_outputs,
                                                           self.dataset_handler.classes_per_task,
                                                           pre_model_output_for_distill=FCTM_output_for_pre_distill)

                pass

            elif "FTCE" == self.cfg.model.cls_type:
                # loss_cls = self.ft_criterion(all_outputs, all_y, dw_cls)
                # feat_class = all_features.detach()
                # ft_outputs = self.model(feat_class, train_cls_use_features=True)
                # ft_logits = ft_outputs[:, 0:active_classes_num]
                # loss_cls = self.ft_criterion(ft_logits, all_y, dw_cls)
                ft_outputs = self.model(all_x, train_classifier=True)
                ft_logits = ft_outputs[:, 0:active_classes_num]
                loss_cls = self.ft_criterion(ft_logits, all_y, dw_cls)


            else:
                raise ValueError(f"self.cfg.model.cls_type is illegal.")
        else:
            loss_cls = 0.

        total_loss = loss_cls + loss_kd
        if self.cfg.model.use_ABD_loss:
            # ft_outputs = self.model(all_x, train_classifier=True)
            # ft_logits = ft_outputs[:, 0:active_classes_num]
            # total_loss += self.ft_criterion(ft_logits, all_y, dw_cls)

            logits_KD = self.previous_teacher.generate_score_by_features(all_features,
                                                                         active_classes_num=dpt_classes_num,
                                                                         is_nograd=False)
            logits_KD_past = pre_model_output_for_distill
            ABD_loss = self.cfg.FCTM.kd_lambda * (
                self.kd_criterion(logits_KD, logits_KD_past).sum(dim=1)).mean() / (
                           logits_KD.size(1))
            total_loss += ABD_loss

    def ABD_loss(self, all_y, all_outputs, logits_for_frozen_features, fctm_logits_for_all_features, FCTM_outputs,
                 cls_criterion, active_classes_num, dpt_classes_num, dw_cls):
        loss_class = cls_criterion(all_outputs[:all_y.shape[0], dpt_classes_num:active_classes_num],
                                   all_y % self.dataset_handler.classes_per_task)

        ft_logits = logits_for_frozen_features
        loss_class += self.ft_criterion(ft_logits, all_y, dw_cls)

        logits_KD = fctm_logits_for_all_features
        logits_KD_past = FCTM_outputs
        loss_kd = self.cfg.model.kd_lambda * (
            self.kd_criterion(logits_KD, logits_KD_past).sum(dim=1)).mean() / (
                      logits_KD.size(1))

        total_loss = loss_class + loss_kd
        return total_loss

    def LwF_loss(self, all_y, all_outputs, FCTM_outputs,
                 cls_criterion, active_classes_num, dpt_classes_num):
        rnt = dpt_classes_num / active_classes_num
        loss_class = cls_criterion(all_outputs[:all_y.shape[0], dpt_classes_num:active_classes_num],
                                   all_y % self.dataset_handler.classes_per_task) * (1 - rnt)

        logits_KD = all_outputs[:, 0:dpt_classes_num]
        logits_KD_past = FCTM_outputs[:, 0:dpt_classes_num]
        loss_kd = compute_distill_loss(output_for_distill=logits_KD, previous_task_model_output=logits_KD_past,
                                       temp=self.cfg.model.T) * rnt

        total_loss = loss_class + loss_kd
        return total_loss

    def RDFCIL_loss(self, y, logits_com, pre_logits_replay, features_new_classes, pre_features_new_classes,
                    cls_criterion, dp_classes_num, active_classes_num, alpha, beta):

        loss_class = cls_criterion(logits_com[:y.shape[0], dp_classes_num:active_classes_num],
                                   y % self.dataset_handler.classes_per_task) * (
                             self.cfg.model.ce_lambda * (1 + 1 / alpha) / beta)
        logits_old_classes = logits_com[y.shape[0]:, :]
        logits_old_classes = logits_old_classes[:, :dp_classes_num]
        pre_logits_replay = pre_logits_replay[y.shape[0]:, :]
        pre_logits_replay = pre_logits_replay[:, :dp_classes_num]
        loss_hkd = (self.kd_criterion(logits_old_classes, pre_logits_replay).sum(dim=1)).mean() / (
            logits_old_classes.size(1)) * self.cfg.model.hkd_lambda

        loss_rkd = self.rkd(student=features_new_classes,
                            teacher=pre_features_new_classes) * self.cfg.model.rkd_lambda
        loss_kd = (loss_hkd + loss_rkd) * (alpha * beta)

        total_loss = loss_class + loss_kd
        return total_loss

    def FCTM_train_a_batch_datafree(self, optimizer, all_cls_criterion, add_cls_criterion, imgs, labels,
                                    active_classes_num):
        dpt_classes_num = active_classes_num - self.dataset_handler.classes_per_task

        mappings = torch.ones(labels.size(), dtype=torch.float32).cuda()
        rnt = 1.0 * dpt_classes_num / active_classes_num
        mappings[:self.cfg.FCTM.TRAIN.BATCH_SIZE] = 1 - rnt
        mappings[self.cfg.FCTM.TRAIN.BATCH_SIZE:] = rnt
        dw_cls = mappings
        assert labels.shape[0] == 2 * self.cfg.FCTM.TRAIN.BATCH_SIZE
        self.FCTM.train()
        '''获取imgs, examplar_imgs在pre_model的输出'''
        pre_model_output, pre_model_imgs_2_features = self.previous_teacher.generate_features_scores(imgs)
        '''获取imgs在要训练的模型FM'''
        outputs = self.FCTM(imgs, pre_model_imgs_2_features)
        FCTM_output = outputs["fctm_logits"][:, 0:active_classes_num]
        all_output = outputs["all_logits"][:, 0:active_classes_num]
        all_output_for_distill = all_output[:, 0:dpt_classes_num]

        pre_model_output_for_distill = pre_model_output[:, 0:dpt_classes_num]

        all_cls_loss = all_cls_criterion(all_output[:self.cfg.FCTM.TRAIN.BATCH_SIZE,
                                         -self.dataset_handler.classes_per_task:],
                                         labels[:self.cfg.FCTM.TRAIN.BATCH_SIZE] %
                                         self.dataset_handler.classes_per_task)
        #
        feat_class = self.FCTM(imgs, pre_model_imgs_2_features, is_nograd=True,
                               calibrated_features_flag=True)  # calibrated_features_flag outputs["features"].detach()
        ft_outputs = self.FCTM(feat_class, pre_model_feature=None, train_cls_use_features=True)
        ft_logits = ft_outputs["all_logits"][:, 0:active_classes_num]
        all_cls_loss += self.ft_criterion(ft_logits, labels, dw_cls)
        # all_cls_loss = self.ft_criterion(ft_logits, labels, dw_cls)

        logits_KD = self.previous_teacher.generate_score_by_features(outputs["features"],
                                                                     active_classes_num=dpt_classes_num,
                                                                     is_nograd=False)
        logits_KD_past = pre_model_output_for_distill
        loss_kd = self.cfg.FCTM.kd_lambda * (
            self.kd_criterion(logits_KD, logits_KD_past).sum(dim=1)).mean() / (
                      logits_KD.size(1))

        add_cls_loss = add_cls_criterion(FCTM_output, labels) * self.cfg.FCTM.addCls_lambda
        # add_cls_loss = 0.

        loss = add_cls_loss + self.cfg.FCTM.allcls_lambda * all_cls_loss + loss_kd
        # if self.cfg.FCTM.use_SSCE:
        #     all_cls_loss = all_cls_criterion(
        #         all_output[:self.cfg.FCTM.TRAIN.BATCH_SIZE, -self.dataset_handler.classes_per_task:],
        #         labels[:self.cfg.FCTM.TRAIN.BATCH_SIZE] % self.dataset_handler.classes_per_task) *
        # else:
        #     all_cls_loss = self.ft_criterion(all_output, labels, dw_cls)
        # add_cls_loss = add_cls_criterion(FCTM_output, labels)
        #
        # if self.cfg.FCTM.use_KD:
        #     if self.cfg.FCTM.use_binary_KD:
        #         distill_loss = compute_distill_binary_loss(all_output_for_distill, pre_model_output_for_distill)
        #     else:
        #         # distill_loss = compute_distill_loss(all_output_for_distill, pre_model_output_for_distill,
        #         #                                     temp=self.cfg.FCTM.T)
        #         loss_KD = torch.zeros(task).cuda()
        #         all_ouput_for_dis = all_output_for_distill
        #         all_pre_model_output_for_dis = pre_model_output_for_distill
        #         for task_id in range(task - 1):
        #             task_id_ouput = all_ouput_for_dis[:, self.dataset_handler.classes_per_task * task_id:
        #                                                  self.dataset_handler.classes_per_task * (task_id + 1)]
        #             task_id_pre_model_output = all_pre_model_output_for_dis[:,
        #                                        self.dataset_handler.classes_per_task * task_id:
        #                                        self.dataset_handler.classes_per_task * (task_id + 1)]
        #             soft_target = Func.softmax(task_id_pre_model_output / self.cfg.model.T,
        #                                        dim=1)
        #             output_log = Func.log_softmax(task_id_ouput / self.cfg.model.T, dim=1)
        #             loss_KD[task_id] = Func.kl_div(output_log, soft_target, reduction='batchmean') * (
        #                     self.cfg.model.T ** 2)
        #         distill_loss = loss_KD.sum()
        #     if self.cfg.FCTM.use_add_cls:
        #         loss = add_cls_loss + all_cls_loss + distill_loss
        #     else:
        #         loss = all_cls_loss + distill_loss
        # else:
        #     loss = add_cls_loss + all_cls_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        current_task_data_result = torch.argmax(all_output, 1)
        # current_task_acc, current_task_cnt = cuda_accuracy(current_task_data_result.cpu().numpy(), labels.cpu().numpy())
        current_task_acc, current_task_cnt = cuda_accuracy(current_task_data_result, labels)
        now_acc, now_cnt = current_task_acc, current_task_cnt
        return loss, now_acc, now_cnt
        pass

    def ABD_train_model(self, train_dataset, active_classes_num, task_id):
        if hasattr(self.FCTM, "module"):
            global_classifier_weights, global_classifier_bias = \
                get_classifier_weight_bias(self.FCTM.module.global_fc)
        else:
            global_classifier_weights, global_classifier_bias = \
                get_classifier_weight_bias(self.FCTM.global_fc)
        # global_classifier_paras = torch.cat((global_classifier_weights, global_classifier_bias), 1)
        global_classifier_weights_norm = torch.norm(global_classifier_weights.detach(), 2, 1)
        global_classifier_weights_norm = global_classifier_weights_norm[0: active_classes_num]
        # global_classifier_paras_norm = torch.norm(global_classifier_paras.detach(), 2, 1)
        dpt_classes_num = active_classes_num - self.dataset_handler.classes_per_task
        self.logger.info(f"global_classifier_weights_norm: {global_classifier_weights_norm}")
        self.logger.info(f"global_classifier_weights_norm mean of old classes: "
                         f"{torch.mean(global_classifier_weights_norm[0:dpt_classes_num])}")
        self.logger.info(f"global_classifier_weights_norm mean of new classes: "
                         f"{torch.mean(global_classifier_weights_norm[dpt_classes_num: active_classes_num])}")
        class_weight_from_feature_norm = global_classifier_weights_norm.sum() / (
                global_classifier_weights_norm * active_classes_num)
        dpt_classes_num = active_classes_num - self.dataset_handler.classes_per_task
        sample_num_grade_classes = torch.from_numpy(
            np.array([active_classes_num - self.dataset_handler.classes_per_task,
                      self.dataset_handler.classes_per_task])).long().to(self.device)

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
        best_acc = 0.
        cls_criterion = CrossEntropy().cuda()
        dw_kd = torch.ones(sample_num_grade_classes.size(), dtype=torch.float32).cuda()
        rnt = 1.0 * dpt_classes_num / active_classes_num
        dw_kd[:-1] = rnt
        dw_kd[-1] = 1 - rnt
        mappings = torch.ones(self.cfg.model.TRAIN.BATCH_SIZE * 2, dtype=torch.float32).cuda()
        mappings[:self.cfg.model.TRAIN.BATCH_SIZE] = 1 - rnt
        mappings[self.cfg.model.TRAIN.BATCH_SIZE:] = rnt
        dw_cls = mappings
        sample_num_per_class = torch.ones(active_classes_num, dtype=torch.float32).cuda()
        sample_num_per_class[:dpt_classes_num] = self.dataset_handler.classes_per_task
        sample_num_per_class[dpt_classes_num:] = dpt_classes_num
        for epoch in range(1, self.cfg.model.TRAIN.MAX_EPOCH + 1):
            all_loss = AverageMeter()
            if float(torch.__version__[:3]) < 1.3:
                scheduler.step()
            iter_index = 0
            for x, y in train_loader:
                self.model.train()
                self.FCTM.eval()
                self.FCTM.zero_grad()
                x, y = x.to(self.device), y.to(self.device)  # --> transfer them to correct device
                # ---> Train MAIN MODEL
                # data replay
                x_replay, y_replay, pre_logits_replay = self.sample(self.previous_teacher, len(x), dpt_classes_num)

                all_x = torch.cat([x, x_replay], dim=0)
                all_y = torch.cat([y, y_replay], dim=0)
                assert all_y.shape[0] == 2 * self.cfg.model.TRAIN.BATCH_SIZE

                logits_com, features_com = self.model(all_x)
                logits_com = logits_com[:, 0:active_classes_num]

                loss_class = cls_criterion(logits_com[:y.shape[0], dpt_classes_num:active_classes_num],
                                           y % self.dataset_handler.classes_per_task) * (1 - rnt)

                feat_class = features_com.detach()
                ft_outputs = self.model(feat_class, train_cls_use_features=True)
                ft_logits = ft_outputs[:, :active_classes_num]
                loss_class += self.ft_criterion(ft_logits, all_y, dw_cls)

                feature_com_past = self.previous_teacher.generate_features(all_x)
                with torch.no_grad():
                    FCTM_outputs_past = self.FCTM(all_x, feature_com_past)
                FCTM_logits_past = FCTM_outputs_past["all_logits"][:, 0:active_classes_num]

                FCTM_outputs = self.FCTM(features_com, pre_model_feature=None, train_cls_use_features=True)
                FCTM_logtis = FCTM_outputs["all_logits"][:, 0:active_classes_num]

                logits_KD = FCTM_logtis
                logits_KD_past = FCTM_logits_past

                loss_kd = compute_ReKD_loss(logits_KD,
                                            logits_KD_past,
                                            sample_num_grade_classes,
                                            temp=self.cfg.model.T,
                                            dw_kd=dw_kd) * self.cfg.model.kd_lambda  # todo Done

                if self.cfg.model.use_featureKD:
                    loss_kd += self.cfg.model.fkd_lambda * (self.kd_criterion(features_com,
                                                                              FCTM_outputs_past["features"]).mean(
                        dim=1)).mean()
                # loss_kd += self.cfg.model.kd_lambda * (
                #     self.kd_criterion(logits_KD, logits_KD_past).sum(dim=1)).mean() / (
                #               logits_KD.size(1))

                total_loss = loss_class + loss_kd
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                # measure accuracy and record loss
                all_loss.update(total_loss.data.item(), y.shape[0] + y_replay.shape[0])
                if iter_index % self.cfg.SHOW_STEP == 0:
                    pbar_str = "Approach: {}| | Epoch: {} || Batch:{:>3d}/{}|| lr: {} " \
                               "|| Batch_Loss:{:>5.3f}".format("EARS_DFCL", epoch, iter_index,
                                                               iter_num,
                                                               optimizer.param_groups[
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
                scheduler.step()

        # self.fine_tune(train_dataset, active_classes_num, task_id)

    def RDFCL_train_model(self, train_dataset, active_classes_num, task_id):
        if hasattr(self.FCTM, "module"):
            global_classifier_weights, global_classifier_bias = \
                get_classifier_weight_bias(self.FCTM.module.global_fc)
        else:
            global_classifier_weights, global_classifier_bias = \
                get_classifier_weight_bias(self.FCTM.global_fc)
        # global_classifier_paras = torch.cat((global_classifier_weights, global_classifier_bias), 1)
        global_classifier_weights_norm = torch.norm(global_classifier_weights.detach(), 2, 1)
        global_classifier_weights_norm = global_classifier_weights_norm[0: active_classes_num]
        # global_classifier_paras_norm = torch.norm(global_classifier_paras.detach(), 2, 1)
        dpt_classes_num = active_classes_num - self.dataset_handler.classes_per_task
        self.logger.info(f"global_classifier_weights_norm: {global_classifier_weights_norm}")
        self.logger.info(f"global_classifier_weights_norm mean of old classes: "
                         f"{torch.mean(global_classifier_weights_norm[0:dpt_classes_num])}")
        self.logger.info(f"global_classifier_weights_norm mean of new classes: "
                         f"{torch.mean(global_classifier_weights_norm[dpt_classes_num: active_classes_num])}")
        class_weight_from_feature_norm = global_classifier_weights_norm.sum() / (
                global_classifier_weights_norm * active_classes_num)
        dpt_classes_num = active_classes_num - self.dataset_handler.classes_per_task
        sample_num_grade_classes = torch.from_numpy(
            np.array([active_classes_num - self.dataset_handler.classes_per_task,
                      self.dataset_handler.classes_per_task])).long().to(self.device)

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
        best_acc = 0.
        cls_criterion = CrossEntropy().cuda()
        dw_kd = torch.ones(sample_num_grade_classes.size(), dtype=torch.float32).cuda()
        rnt = 1.0 * dpt_classes_num / active_classes_num
        dw_kd[:-1] = rnt
        dw_kd[-1] = 1 - rnt
        # rnt = 1.0 * dpt_classes_num / active_classes_num
        # mappings[:dpt_classes_num] = rnt
        # mappings[dpt_classes_num:] = 1 - rnt
        # dw_cls = mappings[all_y.long()]
        sample_num_per_class = torch.ones(active_classes_num, dtype=torch.float32).cuda()
        sample_num_per_class[:dpt_classes_num] = self.dataset_handler.classes_per_task
        sample_num_per_class[dpt_classes_num:] = dpt_classes_num
        for epoch in range(1, self.cfg.model.TRAIN.MAX_EPOCH + 1):
            all_loss = AverageMeter()
            if float(torch.__version__[:3]) < 1.3:
                scheduler.step()
            iter_index = 0
            for x, y in train_loader:
                self.model.train()
                self.FCTM.eval()
                self.FCTM.zero_grad()
                x, y = x.to(self.device), y.to(self.device)  # --> transfer them to correct device
                # ---> Train MAIN MODEL
                # data replay
                x_replay, y_replay, pre_logits_replay = self.sample(self.previous_teacher, len(x), dpt_classes_num)

                all_x = torch.cat([x, x_replay], dim=0)
                all_y = torch.cat([y, y_replay], dim=0)
                assert all_y.shape[0] == 2 * self.cfg.model.TRAIN.BATCH_SIZE

                # mappings = torch.ones(all_y.size(), dtype=torch.float32).cuda()
                # rnt = 1.0 * dpt_classes_num / active_classes_num
                # mappings[:dpt_classes_num] = rnt
                # mappings[dpt_classes_num:] = 1 - rnt
                # dw_cls = mappings[all_y.long()]
                feature_com_past = self.previous_teacher.generate_features(all_x)
                with torch.no_grad():
                    FCTM_outputs_past = self.FCTM(all_x, feature_com_past)
                FCTM_logits_past = FCTM_outputs_past["all_logits"][:, 0:active_classes_num]

                logits_com, features_com = self.model(all_x)
                logits_com = logits_com[:, 0:active_classes_num]

                loss_softTarget = self.cfg.model.softTarget_lambda * reweight_KD(logits_com, FCTM_logits_past, all_y,
                                                                                 class_weight_from_feature_norm)

                logits_KD = logits_com
                logits_KD_past = FCTM_logits_past

                loss_kd = compute_ReKD_loss(logits_KD,
                                            logits_KD_past,
                                            sample_num_grade_classes,
                                            temp=self.cfg.model.T,
                                            dw_kd=dw_kd) * self.cfg.model.kd_lambda  # todo Done

                if self.cfg.model.use_featureKD:
                    loss_kd += self.cfg.model.fkd_lambda * (self.kd_criterion(features_com,
                                                                              FCTM_outputs_past["features"]).mean(
                        dim=1)).mean()

                total_loss = loss_softTarget + loss_kd
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                # measure accuracy and record loss
                all_loss.update(total_loss.data.item(), y.shape[0] + y_replay.shape[0])
                if iter_index % self.cfg.SHOW_STEP == 0:
                    pbar_str = "Approach: {}| | Epoch: {} || Batch:{:>3d}/{}|| lr: {} " \
                               "|| Batch_Loss:{:>5.3f}".format("EARS_DFCL", epoch, iter_index,
                                                               iter_num,
                                                               optimizer.param_groups[
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
                scheduler.step()

        # self.fine_tune(train_dataset, active_classes_num, task_id)

    def validate_with_FCTM(self, task, is_test=False):
        acc = []
        fcn_mode = self.FCTM.training
        self.FCTM.eval()
        for task_id in range(task):  # 这里的task 从0 开始
            if self.dataset_handler.val_datasets and (not is_test):
                predict_result = self.validate_with_FCTM_per_task(self.dataset_handler.val_datasets[task_id], task)
            else:
                predict_result = self.validate_with_FCTM_per_task(self.dataset_handler.test_datasets[task_id],
                                                                  task)
            acc.append(predict_result)
            self.logger.info(
                f"task: {task} || per task {task_id}, validate_with_FC acc:{predict_result}"
            )
        acc = np.array(acc)
        self.FCTM.train(mode=fcn_mode)
        # print(
        #     f"task {task} validate_with_exemplars, acc_avg:{acc.mean()}")
        # self.logger.info(
        #     f"per task {task}, validate_with_exemplars, avg acc:{acc.mean()}"
        #     f"-------------------------------------------------------------"
        # )
        return acc

    def validate_with_FCTM_per_task(self, val_dataset, task):
        # todo
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.cfg.model.TRAIN.BATCH_SIZE,
                                num_workers=self.cfg.model.TRAIN.NUM_WORKERS, shuffle=False, drop_last=False,
                                persistent_workers=True)
        top1 = AverageMeter()
        correct = 0
        active_classes_num = self.dataset_handler.classes_per_task * task
        for inputs, labels in val_loader:
            correct_temp = 0
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            pre_model_features = self.previous_teacher.generate_features(inputs)
            with torch.no_grad():
                out = self.FCTM(inputs, pre_model_features, is_nograd=True)
            _, balance_fc_y_hat = torch.max(out[:, 0:active_classes_num], 1)
            correct_temp += balance_fc_y_hat.eq(labels.data).cpu().sum()
            correct += correct_temp
            top1.update((correct_temp / inputs.size(0)).item(), inputs.size(0))
        del val_loader
        return top1.avg
        pass

    def fine_tune(self, train_dataset, active_classes_num, task_id):
        dp_classes_num = active_classes_num - self.dataset_handler.classes_per_task
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

        dw_cls_temp = torch.ones(self.cfg.model.finetune.BATCH_SIZE, dtype=torch.float32).cuda()

        # bsce_criterion = BalancedSoftmax(sample_num_per_class)
        for epoch in range(1, self.cfg.model.finetune.MAX_EPOCH + 1):
            all_loss = AverageMeter()
            if float(torch.__version__[:3]) < 1.3:
                ft_scheduler.step()
            iter_index = 0
            for iter_index in range(1, iter_num + 1):
                self.model.train()
                # ---> Train MAIN MODEL
                # data replay
                x_replay, y_replay = self.fctm_teacher.sample(batch_size=self.cfg.model.finetune.BATCH_SIZE,
                                                              classes_num=active_classes_num,
                                                              return_scores=False)
                sample_num_per_class = count_sample_num_per_class(y_replay, active_classes_num)
                dw_cls = 1. / sample_num_per_class
                dw_sum = dw_cls.sum()
                dw_cls /= dw_sum
                data_weight = contruct_data_weight(dw_cls, self.cfg.model.finetune.BATCH_SIZE, y_replay)
                logits_com = self.model(x_replay, train_classifier=True)
                logits_com = logits_com[:, 0:active_classes_num]

                loss_class = self.ft_criterion(logits_com, y_replay, data_weight)
                # loss_class = bsce_criterion(logits_com, y_com)

                # total_loss = loss_class + loss_hkd
                total_loss = loss_class
                ft_optimizer.zero_grad()
                total_loss.backward()
                ft_optimizer.step()
                # measure accuracy and record loss
                all_loss.update(total_loss.data.item(), y_replay.shape[0])
                if iter_index % self.cfg.SHOW_STEP == 0:
                    pbar_str = "Approach: {}-Finetune || Epoch: {} || Batch:{:>3d}/{}|| lr: {} " \
                               "|| Batch_Loss:{:>5.3f}".format("EARS_DFCL", epoch, iter_index,
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

    def fine_tune_old_classes_generator(self, train_dataset, active_classes_num, task_id):
        dp_classes_num = active_classes_num - self.dataset_handler.classes_per_task
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
        dw_cls = torch.ones(2 * self.cfg.model.finetune.BATCH_SIZE, dtype=torch.float32).cuda()
        dw_sum = dp_classes_num ** 2 + self.dataset_handler.classes_per_task ** 2
        dw_cls[:self.cfg.model.finetune.BATCH_SIZE] = self.dataset_handler.classes_per_task / dw_sum
        dw_cls[self.cfg.model.finetune.BATCH_SIZE:] = dp_classes_num / dw_sum
        sample_num_per_class = torch.ones(active_classes_num, dtype=torch.float32).cuda()
        sample_num_per_class[0:dp_classes_num] = self.cfg.model.finetune.BATCH_SIZE / dp_classes_num
        sample_num_per_class[
        dp_classes_num:active_classes_num] = self.cfg.model.finetune.BATCH_SIZE / self.dataset_handler.classes_per_task
        bsce_criterion = BalancedSoftmax(sample_num_per_class)

        condition_old_labels = construct_condition_labels(self.cfg.model.finetune.BATCH_SIZE, dp_classes_num)
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
                x_replay, y_replay, pre_logits_replay = self.condition_sample(self.fctm_teacher,
                                                                              self.cfg.model.finetune.BATCH_SIZE,
                                                                              condition_old_labels)

                assert y.shape[0] == y_replay.shape[0] == self.cfg.model.finetune.BATCH_SIZE
                x_com = torch.cat([x, x_replay], dim=0)
                y_com = torch.cat([y, y_replay], dim=0)

                logits_com = self.model(x_com, train_classifier=True)
                logits_com = logits_com[:, 0:active_classes_num]

                loss_class = self.ft_criterion(logits_com, y_com, dw_cls)
                # loss_class = bsce_criterion(logits_com, y_com)

                # total_loss = loss_class + loss_hkd
                total_loss = loss_class
                ft_optimizer.zero_grad()
                total_loss.backward()
                ft_optimizer.step()
                # measure accuracy and record loss
                all_loss.update(total_loss.data.item(), y.shape[0] + y_replay.shape[0])
                if iter_index % self.cfg.SHOW_STEP == 0:
                    pbar_str = "Approach: {}-Finetune || Epoch: {} || Batch:{:>3d}/{}|| lr: {} " \
                               "|| Batch_Loss:{:>5.3f}".format("EARS_DFCL", epoch, iter_index,
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

    def save_best_latest_model_data(self, model_dir, task_id, acc, epoch):
        if self.best_model is None:
            self.best_model = self.model
        self.latest_model = self.model
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

            torch.save({
                'state_dict': self.FCTM.state_dict(),
                'task_id': task_id,
            }, os.path.join(model_dir, "latest_FCTM.pth")
            )
        pass

    def compute_weight_per_class(self, sample_per_class, beta):
        assert 0 < beta < 1
        effective_num = 1.0 - np.power(beta, sample_per_class)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)
        per_cls_weights = per_cls_weights / \
                          np.sum(per_cls_weights) * len(sample_per_class)

        self.logger.info("per cls weights : {}".format(per_cls_weights))
        per_cls_weights = torch.FloatTensor(per_cls_weights).to(self.device)
        return per_cls_weights

    def pre_steps_before_train_model(self, classes_num, old_classes_num):
        # for eval
        if self.fctm_teacher is not None:
            self.previous_fctm_teacher = copy.deepcopy(self.fctm_teacher)

        # new teacher: solver, generator, generator_iters, class_idx, deep_inv_params
        self.fctm_teacher = FCTM_Teacher(model_solver=copy.deepcopy(self.model),
                                         FCTM=copy.deepcopy(self.FCTM),
                                         generator=self.generator,
                                         generator_iters=self.cfg.generator.generator_iter,
                                         deep_inv_params=self.cfg.generator.deep_inv_params,
                                         class_idx=np.arange(classes_num),
                                         old_classes_num=old_classes_num)
        condition_old_labels = construct_condition_labels(self.cfg.generator.batch_size, old_classes_num)
        self.fctm_teacher.train_generator_for_old_classes(self.cfg.generator.batch_size, condition_old_labels)
        pass

    # def pre_steps_before_train_model(self, classes_num):
    #     # for eval
    #     if self.fctm_teacher is not None:
    #         self.previous_fctm_teacher = copy.deepcopy(self.fctm_teacher)
    #
    #     # new teacher: solver, generator, generator_iters, class_idx, deep_inv_params
    #     self.fctm_teacher = FCTM_Teacher(model_solver=copy.deepcopy(self.model),
    #                                      FCTM=copy.deepcopy(self.FCTM),
    #                                      generator=self.generator,
    #                                      generator_iters=self.cfg.generator.generator_iter,
    #                                      deep_inv_params=self.cfg.generator.deep_inv_params,
    #                                      class_idx=np.arange(classes_num))
    #     self.fctm_teacher.train_generator(self.cfg.generator.batch_size)
    #     pass
