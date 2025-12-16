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
from .deepInversionGenBN import DeepInversionGenBN

from lib.continual.datafree_helper import Teacher
from lib.model import resnet_model, CrossEntropy, compute_distill_loss, FCTM_model, compute_distill_binary_loss, \
    compute_ReKD_loss, reweight_KD, BalancedSoftmax, adjusted_KD, BKD, cam_resnet18, compute_cls_distill_binary_loss, \
    compute_BSCEKD_loss, NcmClassifier, compute_posthoc_LAKDLoss, FCN_model
from lib.model.distill_relation import RKDAngleLoss
from lib.utils import AverageMeter, get_optimizer, get_scheduler, cuda_accuracy, get_classifier_weight_bias, \
    contruct_imgs_paths
from lib.utils.util_class import FCTM_GradCAM, GradCAM

'''Code for CKDF_DFCL'''


class CKDF_DFCL(DeepInversionGenBN):

    def __init__(self, cfg, dataset_handler, logger):
        super(CKDF_DFCL, self).__init__(cfg, dataset_handler, logger)
        self.FCN = None
        self.criterion_fn = nn.CrossEntropyLoss(reduction="none")

    def learn_new_task(self, train_dataset, active_classes_num, task):
        dp_classes_num = active_classes_num - self.dataset_handler.classes_per_task
        self.pre_steps(dp_classes_num)  # 训练生成器
        self.train_FCN(train_dataset, active_classes_num, task)

        oldmodel_val_newclass = self.validate_with_FC(test_model=self.model, task=task)
        val_acc_FC_str = f'task: {task} classififer:{"FC"} oldmodel_val_newclass——val_acc: ' \
                         f'{oldmodel_val_newclass}, avg: {oldmodel_val_newclass.mean()} '
        self.logger.info(val_acc_FC_str)
        fctm_val_previous_tasks = self.validate_with_FCN(task - 1, is_test=False)
        val_acc_FC_str = f'task: {task} classififer:{"FC"} fctm_val_previous_tasks_acc: {fctm_val_previous_tasks}, ' \
                         f'avg: {fctm_val_previous_tasks.mean()} '
        self.logger.info(val_acc_FC_str)
        fctm_val = self.validate_with_FCN(task, is_test=False)
        val_acc_FC_str = f'task: {task} classififer:{"FC"} fctm_val_acc: {fctm_val}, avg: {fctm_val.mean()}, ' \
                         f'previous_avg: {fctm_val[0:-1].mean()} '
        self.logger.info(val_acc_FC_str)

        self.train_model(train_dataset, active_classes_num, task)
        self.fine_tune(train_dataset, active_classes_num, task)
        # trains

    def train_FCN(self, train_dataset, active_classes_num, task):
        dpt_classes_num = active_classes_num - self.dataset_handler.classes_per_task
        if self.FCN is None:
            self.FCN = FCN_model(self.cfg)  # todo
            self.FCN = self.transfer_to_cuda(self.FCN)
            print("FCN:", self.FCN)
        optimizer = self.build_optimize(model=self.FCN,
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

        all_cls_criterion = CrossEntropy().to(self.device)

        train_loader = DataLoader(dataset=train_dataset, batch_size=self.cfg.FCTM.TRAIN.BATCH_SIZE,
                                  num_workers=self.cfg.FCTM.TRAIN.NUM_WORKERS, shuffle=True, drop_last=True,
                                  persistent_workers=True)
        iter_num = len(train_loader)
        sample_num_per_class = torch.ones(active_classes_num, dtype=torch.float32)
        sample_num_per_class[
        :dpt_classes_num] = self.cfg.FCTM.TRAIN.BATCH_SIZE / dpt_classes_num
        sample_num_per_class[dpt_classes_num:] = self.cfg.FCTM.TRAIN.BATCH_SIZE / self.dataset_handler.classes_per_task

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
                mappings = torch.ones(all_y.size(), dtype=torch.float32).to(self.device)
                rnt = 1.0 * dpt_classes_num / active_classes_num
                mappings[:dpt_classes_num] = rnt
                mappings[dpt_classes_num:] = 1 - rnt
                dw_cls = mappings[all_y.long()]
                loss, now_acc, now_cnt = self.FCN_train_a_batch_datafree(optimizer, all_cls_criterion,
                                                                         all_x, all_y,
                                                                         active_classes_num,
                                                                         dw_cls=dw_cls)
                all_loss.update(loss.data.item(), cnt)
                acc.update(now_acc, y.shape[0])
                if iter_index % self.cfg.SHOW_STEP == 0:
                    pbar_str = "FCN train, Epoch: {} || Batch:{:>3d}/{} || lr : {} || Loss:{:>5.3f} || " \
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

                val_acc = self.validate_with_FCN(task)  # task_id 从1开始
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

    def ABD_loss(self, labels, all_output, logits_for_frozen_features, pre_logits_for_all_features,
                 pre_model_output, cls_criterion, dpt_classes_num, active_classes_num, dw_cls):
        loss_class = cls_criterion(all_output[:labels.shape[0], dpt_classes_num:active_classes_num],
                                   labels % self.dataset_handler.classes_per_task)

        ft_logits = logits_for_frozen_features
        loss_class += self.ft_criterion(ft_logits, labels, dw_cls)

        logits_KD = pre_logits_for_all_features
        logits_KD_past = pre_model_output
        loss_kd = self.cfg.FCTM.kd_lambda * (
            self.kd_criterion(logits_KD, logits_KD_past).sum(dim=1)).mean() / (
                      logits_KD.size(1))

        total_loss = loss_class + loss_kd
        return total_loss

    def ft_criterion(self, logits, targets, data_weights):
        loss_supervised = (self.criterion_fn(logits, targets.long()) * data_weights).mean()
        return loss_supervised

    def train_model(self, train_dataset, active_classes_num, task):
        dpt_classes_num = active_classes_num - self.dataset_handler.classes_per_task
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
        cls_criterion = CrossEntropy().to(self.device)
        iter_num = len(train_loader)
        best_acc = 0.
        sample_num_per_class = torch.ones(active_classes_num, dtype=torch.float32)
        sample_num_per_class[
        :dpt_classes_num] = self.cfg.model.TRAIN.BATCH_SIZE / dpt_classes_num
        sample_num_per_class[dpt_classes_num:] = self.cfg.model.TRAIN.BATCH_SIZE / self.dataset_handler.classes_per_task

        rnt = 1.0 * dpt_classes_num / active_classes_num
        for epoch in range(1, self.cfg.model.TRAIN.MAX_EPOCH + 1):
            all_loss = AverageMeter()
            if float(torch.__version__[:3]) < 1.3:
                scheduler.step()
            iter_index = 0
            for x, y in train_loader:
                self.model.train()
                self.FCN.eval()
                x, y = x.to(self.device), y.to(self.device)  # --> transfer them to correct device
                # ---> Train MAIN MODEL
                # data replay
                x_replay, y_replay, pre_logits_replay = self.sample(self.previous_teacher, len(x), dpt_classes_num)

                all_x = torch.cat([x, x_replay], dim=0)
                all_y = torch.cat([y, y_replay], dim=0)
                assert all_y.shape[0] == 2 * self.cfg.model.TRAIN.BATCH_SIZE

                all_outputs, all_features = self.model(all_x)
                all_outputs_for_distill = all_outputs[:, 0:active_classes_num]
                pre_all_output, pre_all_features = self.previous_teacher.generate_features_scores(
                    all_x)  # 获取classifier_output
                with torch.no_grad():
                    FCN_outs = self.FCN(pre_all_features)
                FCN_outputs = FCN_outs["all_logits"][:, 0:active_classes_num]
                calibrated_features = FCN_outs["calibrated_features"]

                loss_KD = torch.zeros(task).to(self.device)
                all_ouput_for_dis = all_outputs_for_distill
                all_pre_model_output_for_dis = FCN_outputs
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

                self.FCN.zero_grad()
                logits_KD = self.FCN(pre_model_feature=all_features, calibrate_features_2_logits=True)["all_logits"]
                # logits_KD = logits_KD[:, :dpt_classes_num]
                # logits_KD_past = FCN_outputs[:, :dpt_classes_num]
                logits_KD = logits_KD[:, :active_classes_num]
                logits_KD_past = FCN_outputs[:, :active_classes_num]
                # # loss_kd += self.cfg.FCTM.kd_lambda * (
                # #     self.kd_criterion(logits_KD, logits_KD_past).sum(dim=1)).mean() / (
                # #               logits_KD.size(1))
                loss_kd += self.cfg.model.fkd_lambda * (
                    self.kd_criterion(logits_KD, logits_KD_past).sum(dim=1)).mean() / (
                               logits_KD.size(1))
                # loss_kd += 0.5 * (
                #     self.kd_criterion(logits_KD, logits_KD_past).sum(dim=1)).mean() / (
                #                logits_KD.size(1))

                loss_cls = cls_criterion(all_outputs[:self.cfg.model.TRAIN.BATCH_SIZE,
                                         -self.dataset_handler.classes_per_task:],
                                         all_y[:self.cfg.model.TRAIN.BATCH_SIZE] %
                                         self.dataset_handler.classes_per_task) * (1 - rnt)

                FCN_output_for_pre_distill = FCN_outputs[:, 0:dpt_classes_num]
                loss_cls += compute_distill_loss(all_outputs[:, 0:dpt_classes_num], FCN_output_for_pre_distill,
                                                 temp=self.cfg.model.T) * rnt

                total_loss = loss_kd + loss_cls
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                # measure accuracy and record loss
                all_loss.update(total_loss.data.item(), y.shape[0] + y_replay.shape[0])
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

    def FCN_train_a_batch_datafree(self, optimizer, cls_criterion, imgs, labels,
                                   active_classes_num, dw_cls):
        dpt_classes_num = active_classes_num - self.dataset_handler.classes_per_task
        assert labels.shape[0] == 2 * self.cfg.FCTM.TRAIN.BATCH_SIZE
        self.FCN.train()
        '''获取imgs, examplar_imgs在pre_model的输出'''
        pre_model_output, pre_model_imgs_2_features = self.previous_teacher.generate_features_scores(imgs)
        pre_model_output = pre_model_output[:, 0:dpt_classes_num]
        '''获取imgs在要训练的模型FM'''
        outputs = self.FCN(pre_model_imgs_2_features)
        all_output = outputs["all_logits"][:, 0:active_classes_num]
        all_features = outputs["calibrated_features"]

        frozen_features = self.FCN(pre_model_imgs_2_features, is_nograd=True, feature_flag=True)
        logits_for_frozen_features = self.FCN(frozen_features, calibrate_features_2_logits=True)["all_logits"][:,
                                     0:active_classes_num]

        pre_logits_for_all_features = self.previous_teacher.generate_score_by_features(all_features,
                                                                                       active_classes_num=dpt_classes_num)

        "train_classifier"

        loss = self.ABD_loss(labels, all_output, logits_for_frozen_features, pre_logits_for_all_features,
                             pre_model_output, cls_criterion, dpt_classes_num, active_classes_num,
                             dw_cls)  # compute_distill_loss(all_output_for_distill, pre_model_output_for_distill, temp=self.cfg.FCTM.T)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        current_task_data_result = torch.argmax(all_output, 1)
        # current_task_acc, current_task_cnt = cuda_accuracy(current_task_data_result.cpu().numpy(), labels.cpu().numpy())
        current_task_acc, current_task_cnt = cuda_accuracy(current_task_data_result, labels)
        now_acc, now_cnt = current_task_acc, current_task_cnt
        return loss, now_acc, now_cnt
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

    def validate_with_FCN(self, task, is_test=False):
        acc = []
        fcn_mode = self.FCN.training
        self.FCN.eval()
        for task_id in range(task):  # 这里的task 从0 开始
            if self.dataset_handler.val_datasets and (not is_test):
                predict_result = self.validate_with_FCN_per_task(self.dataset_handler.val_datasets[task_id], task)
            else:
                predict_result = self.validate_with_FCN_per_task(self.dataset_handler.test_datasets[task_id],
                                                                 task)
            acc.append(predict_result)
            self.logger.info(
                f"task: {task} || per task {task_id}, validate_with_FC acc:{predict_result}"
            )
        acc = np.array(acc)
        self.FCN.train(mode=fcn_mode)
        # print(
        #     f"task {task} validate_with_exemplars, acc_avg:{acc.mean()}")
        # self.logger.info(
        #     f"per task {task}, validate_with_exemplars, avg acc:{acc.mean()}"
        #     f"-------------------------------------------------------------"
        # )
        return acc

    def validate_with_FCN_per_task(self, val_dataset, task):
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
                out = self.FCN(pre_model_features, is_nograd=True)
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
        dw_cls = torch.ones(2 * self.cfg.model.finetune.BATCH_SIZE, dtype=torch.float32).to(self.device)
        dw_sum = dp_classes_num ** 2 + self.dataset_handler.classes_per_task ** 2
        dw_cls[:self.cfg.model.finetune.BATCH_SIZE] = self.dataset_handler.classes_per_task / dw_sum
        dw_cls[self.cfg.model.finetune.BATCH_SIZE:] = dp_classes_num / dw_sum
        sample_num_per_class = torch.ones(active_classes_num, dtype=torch.float32).to(self.device)
        sample_num_per_class[0:dp_classes_num] = self.cfg.model.finetune.BATCH_SIZE / dp_classes_num
        sample_num_per_class[
        dp_classes_num:active_classes_num] = self.cfg.model.finetune.BATCH_SIZE / self.dataset_handler.classes_per_task
        bsce_criterion = BalancedSoftmax(sample_num_per_class)
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
                'state_dict': self.FCN.state_dict(),
                'task_id': task_id,
            }, os.path.join(model_dir, "latest_FCN.pth")
            )
        pass
