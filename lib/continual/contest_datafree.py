from __future__ import print_function

import os
from math import log, sqrt
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import copy
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import functional as Func
from lib import generator_models
from lib.continual.datafree_helper import Teacher
from lib.model import resnet_model, CrossEntropy, compute_distill_loss, FCTM_model, compute_distill_binary_loss, \
    compute_ReKD_loss, reweight_KD, BalancedSoftmax
from lib.model.distill_relation import RKDAngleLoss
from lib.utils import AverageMeter, get_optimizer, get_scheduler, cuda_accuracy, get_classifier_weight_bias


class DeepInversionGenBN:

    def __init__(self, cfg, dataset_handler, logger):
        self.cfg = cfg
        self.dataset_handler = dataset_handler
        self.logger = logger
        self.model = None
        self.kd_criterion = None

        # gen parameters
        self.generator = None
        self.previous_teacher = None  # a class object including generator and the model

        self.gpus = torch.cuda.device_count()
        self.device_ids = [] if cfg.availabel_cudas == "" else \
            [i for i in range(len(self.cfg.availabel_cudas.strip().split(',')))]
        self.device = torch.device("cuda")

        self.train_init()
        self.acc_result = None
        self.start_task_id = None
        # data weighting
        self.sample_num_per_class = None

        self.best_model = None
        self.latest_model = None
        self.best_epoch = None
        self.best_acc = None

    def construct_model(self):
        model = resnet_model(self.cfg)
        return model
        pass

    def build_optimize(self, model=None, base_lr=None, optimizer_type=None, momentum=None, weight_decay=None):
        # todo Done
        MODEL = model if model else self.model
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
        self.kd_criterion = nn.MSELoss(reduction="none")
        # self.generator = self.create_generator()
        # self.generator_optimizer = Adam(params=self.generator.parameters(), lr=self.deep_inv_params[0])
        self.model = self.transfer_to_cuda(self.model)
        # self.generator = self.transfer_to_cuda(self.generator)

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
        model = self.construct_model()
        model.load_model(filename)
        self.model = self.transfer_to_cuda(model)

    def create_generator(self):
        # Define the backbone (MLP, LeNet, VGG, ResNet ... etc) of model
        generator = generator_models.generator.__dict__[self.cfg.generator.gen_model_name]()
        return generator

    def resume(self, resumed_model_path, checkpoint):
        acc_result = checkpoint['acc_result']
        self.start_task_id = checkpoint['task_id']
        self.load_model(resumed_model_path)
        self.is_resume_legal(acc_result)
        pass

    def is_resume_legal(self, acc_result):
        self.logger.info(f"Resumed acc_result: {acc_result}")
        FC_acc = self.validate_with_FC(task=self.start_task_id)
        self.logger.info(f"validate resumed model: {FC_acc} || {FC_acc.mean()}")

    def print_model(self):
        print(f"self.model: {self.model}")
        print(f"self.generator: {self.generator}")
        pass

    ##########################################
    #           MODEL TRAINING               #
    ##########################################

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
        criterion = CrossEntropy().cuda()
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

    def learn_new_task(self, train_dataset, active_classes_num, task_id):
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
                x_replay, y_replay, pre_logits_replay = self.sample(self.previous_teacher, len(x), dp_classes_num)
                pre_logits = self.previous_teacher.generate_scores(x, allowed_predictions=dp_classes_num)
                x_com = torch.cat([x, x_replay], dim=0)
                y_com = torch.cat([y, y_replay], dim=0)
                pre_logits_com = torch.cat([pre_logits, pre_logits_replay], dim=0)

                logits, _ = self.model(x_com)
                logits = logits[:, 0:active_classes_num]

                loss_class = criterion(logits, y_com)

                logits_for_distill = logits[:, 0:dp_classes_num]

                loss_kd = self.cfg.model.mu * compute_distill_loss(logits_for_distill, pre_logits_com,
                                                                   temp=self.cfg.model.T)
                total_loss = loss_class + loss_kd
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                # measure accuracy and record loss
                all_loss.update(total_loss.data.item(), y_com.shape[0])
                if iter_index % self.cfg.SHOW_STEP == 0:
                    pbar_str = "Approach: {}| | Epoch: {} || Batch:{:>3d}/{}|| lr: {} " \
                               "|| Batch_Loss:{:>5.3f}".format("LwF-UE", epoch, iter_index,
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

    def pre_steps(self, dp_classes_num):
        # for eval
        if self.previous_teacher is not None:
            self.previous_previous_teacher = self.previous_teacher

        # new teacher: solver, generator, generator_iters, class_idx, deep_inv_params
        self.previous_teacher = Teacher(solver=copy.deepcopy(self.model), generator=self.generator,
                                        generator_iters=self.cfg.generator.generator_iter,
                                        deep_inv_params=self.cfg.generator.deep_inv_params,
                                        class_idx=np.arange(dp_classes_num))
        self.previous_teacher.train_generator(self.cfg.generator.batch_size)
        pass

    ##########################################
    #             MODEL UTILS                #
    ##########################################

    def count_parameter_gen(self):
        return sum(p.numel() for p in self.generator.parameters())

    def sample(self, teacher, batch_size, dp_classes_num, return_scores=True):
        return teacher.sample(batch_size, dp_classes_num, return_scores=return_scores)

    def validate_with_FC(self, task):
        acc = []
        mode = self.model.training
        self.model.eval()
        for task_id in range(task):  # 这里的task 从0 开始
            predict_result = self.validate_with_FC_per_task(self.dataset_handler.val_datasets[task_id], task)
            acc.append(predict_result)
            self.logger.info(
                f"task: {task} || per task {task_id}, validate_with_FC acc:{predict_result}"
            )
        acc = np.array(acc)
        self.model.train(mode=mode)
        return acc
        pass

    def validate_with_FC_per_task(self, val_dataset, task):
        # todo
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.cfg.model.TRAIN.BATCH_SIZE,
                                num_workers=self.cfg.model.TRAIN.NUM_WORKERS, shuffle=False, drop_last=False)
        top1 = AverageMeter()
        correct = 0
        active_classes_num = self.dataset_handler.classes_per_task * task
        for inputs, labels in val_loader:
            correct_temp = 0
            inputs, labels = inputs.to("cuda"), labels.to("cuda")
            out = self.model(x=inputs, is_nograd=True, get_classifier=True)
            out = out[:, 0:active_classes_num]
            _, balance_fc_y_hat = torch.max(out, 1)
            correct_temp += balance_fc_y_hat.eq(labels.data).cpu().sum()
            correct += correct_temp
            top1.update((correct_temp / inputs.size(0)).item(), inputs.size(0))
        return top1.avg
        pass

    def validate_with_FC_taskIL(self, task, is_test=False):
        acc = []
        mode = self.model.training
        self.model.eval()
        for task_id in range(task):  # 这里的task 从0 开始
            predict_result = self.validate_with_FC_per_task_taskIL(self.dataset_handler.val_datasets[task_id], task_id)
            acc.append(predict_result)
            self.logger.info(
                f"task: {task} || per task {task_id}, validate_with_FC acc:{predict_result}"
            )
        acc = np.array(acc)
        self.model.train(mode=mode)
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
            inputs, labels = inputs.to("cuda"), labels.to("cuda")
            out = self.model(x=inputs, is_nograd=True, get_classifier=True)
            _, balance_fc_y_hat = torch.max(out[:, allowed_classes], 1)
            balance_fc_y_hat += task_id * self.dataset_handler.classes_per_task
            correct_temp += balance_fc_y_hat.eq(labels.data).cpu().sum()
            correct += correct_temp
            top1.update((correct_temp / inputs.size(0)).item(), inputs.size(0))
        return top1.avg
        pass


class contest_DFCL(DeepInversionGenBN):

    def __init__(self, cfg, dataset_handler, logger):
        super(contest_DFCL, self).__init__(cfg, dataset_handler, logger)
        self.FCTM = None
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

    def learn_new_task(self, train_dataset, active_classes_num, task_id):
        dp_classes_num = active_classes_num - self.dataset_handler.classes_per_task
        self.pre_steps(dp_classes_num) # 训练生成器
        self.train_FCTM(train_dataset, active_classes_num, task_id)
        # self.train_model(train_dataset, active_classes_num, task_id)
        self.ABD_train_model(train_dataset, active_classes_num, task_id)
        # self.RDFCL_train_FCTM(train_dataset, active_classes_num, task_id) #训练扩展模型
        # self.RDFCL_train_model(train_dataset, active_classes_num, task_id) #压缩扩展模型

        # trains

    def train_FCTM(self, train_dataset, active_classes_num, task_id):
        dpt_classes_num = active_classes_num - self.dataset_handler.classes_per_task
        if self.FCTM is None:
            self.FCTM = FCTM_model(self.cfg)
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

    def RDFCL_train_FCTM(self, train_dataset, active_classes_num, task_id):
        dpt_classes_num = active_classes_num - self.dataset_handler.classes_per_task
        if self.FCTM is None:
            self.FCTM = FCTM_model(self.cfg)
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

    def train_model(self, train_dataset, active_classes_num, task_id):
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
                    loss_KD = torch.zeros(task_id).cuda()
                    all_ouput_for_dis = all_outputs[:, 0:dpt_classes_num]
                    all_pre_model_output_for_dis = FCTM_outputs[:, 0:dpt_classes_num]
                    for task_id in range(task_id - 1):
                        task_id_ouput = all_ouput_for_dis[:, self.dataset_handler.classes_per_task * task_id:
                                                             self.dataset_handler.classes_per_task * (task_id + 1)]
                        task_id_pre_model_output = all_pre_model_output_for_dis[:,
                                                   self.dataset_handler.classes_per_task * task_id:
                                                   self.dataset_handler.classes_per_task * (task_id + 1)]
                        soft_target = Func.softmax(task_id_pre_model_output / self.cfg.model.T,
                                                   dim=1)
                        output_log = Func.log_softmax(task_id_ouput / self.cfg.model.T, dim=1)
                        loss_KD[task_id] = Func.kl_div(output_log, soft_target, reduction='batchmean') * (
                                self.cfg.model.T ** 2)
                    loss_kd = loss_KD.sum() * self.cfg.model.kd_lambda
                elif "RTKD" == self.cfg.model.KD_type:
                    loss_KD = torch.zeros(task_id).cuda()
                    all_ouput_for_dis = all_outputs
                    all_pre_model_output_for_dis = FCTM_outputs
                    for task_id in range(task_id):
                        task_id_ouput = all_ouput_for_dis[:, self.dataset_handler.classes_per_task * task_id:
                                                             self.dataset_handler.classes_per_task * (task_id + 1)]
                        task_id_pre_model_output = all_pre_model_output_for_dis[:,
                                                   self.dataset_handler.classes_per_task * task_id:
                                                   self.dataset_handler.classes_per_task * (task_id + 1)]
                        soft_target = Func.softmax(task_id_pre_model_output / self.cfg.model.T,
                                                   dim=1)
                        output_log = Func.log_softmax(task_id_ouput / self.cfg.model.T, dim=1)
                        loss_KD[task_id] = Func.kl_div(output_log, soft_target, reduction='batchmean') * (
                                self.cfg.model.T ** 2)
                    loss_kd = loss_KD.sum() * self.cfg.model.kd_lambda

                if self.cfg.model.use_featureKD:
                    loss_kd += self.cfg.model.fkd_lambda * (self.kd_criterion(all_features,
                                                                              FCTM_features).mean(dim=1)).mean()

                if "softTarget" == self.cfg.model.cls_type:
                    loss_cls = reweight_KD(all_outputs, FCTM_outputs, all_y,
                                           class_weight_from_feature_norm, temp=self.cfg.model.T)
                elif "LwF" == self.cfg.model.cls_type:
                    loss_cls = cls_criterion(all_outputs[:self.cfg.FCTM.TRAIN.BATCH_SIZE,
                                             -self.dataset_handler.classes_per_task:],
                                             all_y[:self.cfg.FCTM.TRAIN.BATCH_SIZE] %
                                             self.dataset_handler.classes_per_task) * (1 - rnt)
                    loss_cls += compute_distill_loss(all_outputs[:, 0:dpt_classes_num], pre_model_output_for_distill,
                                                     temp=self.cfg.model.T) * rnt
                    pass

                elif "FTCE" == self.cfg.model.cls_type:
                    mappings = torch.ones(all_y.size(), dtype=torch.float32).cuda()
                    mappings[:dpt_classes_num] = rnt
                    mappings[dpt_classes_num:] = 1 - rnt
                    dw_cls = mappings[all_y.long()]
                    loss_cls = self.ft_criterion(all_outputs, all_y, dw_cls)

                else:
                    raise ValueError(f"self.cfg.model.cls_type is illegal.")

                total_loss = loss_cls + loss_kd
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

    def FCTM_train_a_batch_datafree(self, optimizer, all_cls_criterion, add_cls_criterion, imgs, labels,
                                    active_classes_num):
        dpt_classes_num = active_classes_num - self.dataset_handler.classes_per_task

        mappings = torch.ones(labels.size(), dtype=torch.float32).cuda()
        rnt = 1.0 * dpt_classes_num / active_classes_num
        mappings[:dpt_classes_num] = rnt
        mappings[dpt_classes_num:] = 1 - rnt
        dw_cls = mappings[labels.long()]
        # rnt = 1.0 * dpt_classes_num / active_classes_num
        # sample_num_per_class = torch.ones(active_classes_num, dtype=torch.float32).cuda()
        # sample_num_per_class[:dpt_classes_num] = self.dataset_handler.classes_per_task
        # sample_num_per_class[dpt_classes_num:] = dpt_classes_num
        assert labels.shape[0] == 2 * self.cfg.FCTM.TRAIN.BATCH_SIZE
        self.FCTM.train()
        '''获取imgs, examplar_imgs在pre_model的输出'''
        pre_model_output, pre_model_imgs_2_features = self.previous_teacher.generate_features_scores(imgs)
        '''获取imgs在要训练的模型FM'''
        outputs = self.FCTM(imgs, pre_model_imgs_2_features)
        FCTM_output = outputs["fctm_logits"][:, 0:active_classes_num]
        all_output = outputs["all_logits"][:, 0:active_classes_num]

        pre_model_output_for_distill = pre_model_output[:, 0:dpt_classes_num]

        all_cls_loss = all_cls_criterion(all_output[:self.cfg.FCTM.TRAIN.BATCH_SIZE,
                                         -self.dataset_handler.classes_per_task:],
                                         labels[:self.cfg.FCTM.TRAIN.BATCH_SIZE] %
                                         self.dataset_handler.classes_per_task) * (1 - rnt)

        feat_class = outputs["features"].detach()
        ft_outputs = self.FCTM(feat_class, pre_model_feature=None, train_cls_use_features=True)
        ft_logits = ft_outputs["all_logits"][:, 0:active_classes_num]
        all_cls_loss += self.ft_criterion(ft_logits, labels, dw_cls)

        logits_KD = self.previous_teacher.generate_score_by_features(outputs["features"],
                                                                     active_classes_num=dpt_classes_num,
                                                                     is_nograd=False)
        logits_KD_past = pre_model_output_for_distill
        loss_kd = self.cfg.FCTM.kd_lambda * (
            self.kd_criterion(logits_KD, logits_KD_past).sum(dim=1)).mean() / (
                      logits_KD.size(1))

        add_cls_loss = add_cls_criterion(FCTM_output, labels) * self.cfg.FCTM.addCls_lambda

        loss = add_cls_loss + all_cls_loss + loss_kd
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

                mappings = torch.ones(all_y.size(), dtype=torch.float32).cuda()
                rnt = 1.0 * dpt_classes_num / active_classes_num
                mappings[:dpt_classes_num] = rnt
                mappings[dpt_classes_num:] = 1 - rnt
                dw_cls = mappings[all_y.long()]
                # rnt = 1.0 * dpt_classes_num / active_classes_num
                # mappings[:dpt_classes_num] = rnt
                # mappings[dpt_classes_num:] = 1 - rnt
                # dw_cls = mappings[all_y.long()]

                logits_com, features_com = self.model(all_x)
                logits_com = logits_com[:, 0:active_classes_num]

                loss_class = cls_criterion(logits_com[:y.shape[0], dpt_classes_num:active_classes_num],
                                           y % self.dataset_handler.classes_per_task) * (1 - rnt)

                feat_class = features_com.detach()
                ft_outputs = self.model(feat_class, train_cls_use_features=True)
                ft_logits = ft_outputs[:, :active_classes_num]
                loss_class += self.ft_criterion(ft_logits, all_y, dw_cls)
                # loss_class = self.bsce_criterion(ft_logits, all_y, sample_num_per_class)

                feature_com_past = self.previous_teacher.generate_features(all_x)
                with torch.no_grad():
                    FCTM_outputs_past = self.FCTM(all_x, feature_com_past)
                FCTM_logits_past = FCTM_outputs_past["all_logits"][:, 0:active_classes_num]

                FCTM_outputs = self.FCTM(all_x, features_com)
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
