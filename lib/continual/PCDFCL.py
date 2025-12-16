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


'''Code for PCDFCL'''

class PCDFCL(DeepInversionGenBN):

    def __init__(self, cfg, dataset_handler, logger):
        super(PCDFCL, self).__init__(cfg, dataset_handler, logger)
        self.new_model = None
        self.criterion_fn = nn.CrossEntropyLoss(reduction="none")
        self.centroid = None
        self.stddevs = None
        self.ranges = None
        self.sample_num_per_class = None
        self.classifier = None
        self.model_optim_params_list = []
        self.train_dataset_for_all = None
        self.rkd = RKDAngleLoss(
            self.cfg.extractor.output_feature_dim,
            proj_dim=2 * self.cfg.extractor.output_feature_dim,
        )
        self.rkd = self.transfer_to_cuda(self.rkd)

    def construct_datasets_for_all(self, active_classes_num, train_dataset):
        dpt_classes_num = active_classes_num - self.dataset_handler.classes_per_task
        current_task_classes_imgs_num = int(len(train_dataset) / self.dataset_handler.classes_per_task)
        old_class_sample_num = 0
        old_classes_exemplars = [[] for i in range(dpt_classes_num)]
        while old_class_sample_num < current_task_classes_imgs_num:
            x_replay, y_replay, pre_logits_replay = self.sample(self.previous_teacher, self.cfg.model.TRAIN.BATCH_SIZE,
                                                                dpt_classes_num)
            for index in range(len(y_replay)):
                if len(old_classes_exemplars[y_replay[index]]) < current_task_classes_imgs_num:
                    old_classes_exemplars[y_replay[index]].append(x_replay[index].cpu())
            old_class_sample_num = len(old_classes_exemplars[0])
            for samples_per_class in old_classes_exemplars:
                if old_class_sample_num > len(samples_per_class):
                    old_class_sample_num = len(samples_per_class)
            print(old_class_sample_num)
        old_classes_datasets = SynthesizedDataset(old_classes_exemplars)
        self.train_dataset_for_all = None
        self.train_dataset_for_all = ConcatDataset([old_classes_datasets, train_dataset])

    def extract_features(self):
        feats_all, labels_all = [], []
        train_loader = DataLoader(dataset=self.train_dataset_for_all, batch_size=self.cfg.new_model.TRAIN.BATCH_SIZE,
                                  num_workers=self.cfg.new_model.TRAIN.NUM_WORKERS, shuffle=True, drop_last=True,
                                  persistent_workers=True)
        for x, y in train_loader:
            x, y = x.to(self.device), y.to(self.device)  # --> transfer them to correct device
            _, features = self.new_model(x, is_nograd=True)
            feats_all.append(features.cpu().numpy())
            labels_all.append(y.cpu().numpy())
        total_features = np.concatenate(feats_all)
        total_labels = np.concatenate(labels_all)
        return total_features, total_labels

    def compute_centroids(self, active_classes_num):
        total_features, total_labels = self.extract_features()
        centroids, stddevs, ranges = [], [], []
        epsilon = 1e-1
        for cls in range(active_classes_num):
            cls_idx = np.where(total_labels == cls)[0]
            cls_feats = total_features[cls_idx]
            cls_mean = np.mean(cls_feats, axis=0)
            centroids.append(cls_mean)
            cls_stddev = np.mean(np.linalg.norm(cls_feats - cls_mean[None, :], axis=1))
            stddevs.append(cls_stddev)
            print(f"cls_feats.shape: {cls_feats.shape}")
            cls_range = np.max(np.linalg.norm(cls_feats[None, :, :] - cls_feats[:, None, :], axis=2))
            ranges.append(cls_range)
        assert len(centroids) == active_classes_num
        self.centroids = torch.from_numpy(np.vstack(centroids)).to(self.device)
        self.stddevs = torch.from_numpy(np.array(stddevs)).to(self.device)
        self.ranges = torch.from_numpy(np.array(ranges)).to(self.device)

    def get_prototype(self, active_classes_num):
        if hasattr(self.new_model, "module"):
            global_classifier_weights, global_classifier_bias = \
                get_classifier_weight_bias(self.new_model.module.linear_classifier)
        else:
            global_classifier_weights, global_classifier_bias = \
                get_classifier_weight_bias(self.new_model.linear_classifier)
        centroids = global_classifier_weights[0:active_classes_num, :]
        self.centroids = centroids.to(self.device)
        self.stddevs = None
        self.ranges = None
        assert len(self.centroids) == active_classes_num
        pass

    def ft_criterion(self, logits, targets, data_weights):
        loss_supervised = (self.criterion_fn(logits, targets.long()) * data_weights).mean()
        return loss_supervised

    def learn_new_task(self, train_dataset, active_classes_num, task):
        dp_classes_num = active_classes_num - self.dataset_handler.classes_per_task
        if self.new_model is None:
            self.pre_steps(dp_classes_num)  # 训练生成器
        else:
            if "ABD" == self.cfg.new_model.loss_type:
                self.pre_steps(dp_classes_num)  # 训练生成器
            else:
                self.pre_steps(dp_classes_num, model=self.new_model)  # 训练生成器
            # self.pre_steps(dp_classes_num)  # 训练生成器
        self.train_new_model(train_dataset, active_classes_num, task)

        if not "CFIL" == self.cfg.new_model.loss_type and not "SSIL" == self.cfg.new_model.loss_type:

            self.ERA_train_model(train_dataset, active_classes_num, task)

            self.fine_tune(train_dataset, active_classes_num, task, model=self.model,
                           discription="No use feature compress",
                           scale_factor=None)
            # if "R-DFCIL" == self.cfg.new_model.loss_type or "CFIL" == self.cfg.new_model.loss_type:
            #     self.fine_tune(train_dataset, active_classes_num, task, model=self.new_model)
            # self.evaluate_model(task, model=self.new_model, discritption="New model Evaluate")
            #
            scale_factor = self.FCC_train_model(train_dataset, active_classes_num, task)
            self.post_hoc_evaluate_model(task, model=self.new_model, scale_factor=scale_factor,
                                         discritption="New model Post-hoc-FCC-Evaluate")
            self.evaluate_model(task, model=self.model, discritption="Model Evaluate")
            if "R-DFCIL" == self.cfg.new_model.loss_type or "CFIL" == self.cfg.new_model.loss_type:
                if "NoFCC" in self.cfg.model.loss_type or (not self.cfg.model.use_finetune_FCC):
                    self.fine_tune(train_dataset, active_classes_num, task, model=self.model,
                                   discription="No use feature compress",
                                   scale_factor=scale_factor)
                else:
                    self.fine_tune(train_dataset, active_classes_num, task, model=self.model, discription="use FCC",
                                   scale_factor=scale_factor)
                # else:
                #     raise ValueError(f"self.cfg.model.loss_type is illegal.")
                # self.fine_tune(train_dataset, active_classes_num, task, model=self.model,
                #                discription="No use feature compress",
                #                scale_factor=scale_factor)
        self.idendity_model()
        # self.train_model(train_dataset, active_classes_num, task)

    def fine_tune(self, train_dataset, active_classes_num, task_id, model=None, discription="", scale_factor=None):
        dp_classes_num = active_classes_num - self.dataset_handler.classes_per_task
        # trains
        ft_optimizer = self.build_optimize(model=model,
                                           base_lr=self.cfg.new_model.finetune.BASE_LR,
                                           optimizer_type=self.cfg.new_model.finetune.TYPE,
                                           momentum=self.cfg.new_model.finetune.MOMENTUM,
                                           weight_decay=self.cfg.new_model.finetune.WEIGHT_DECAY)
        ft_scheduler = self.build_scheduler(ft_optimizer, lr_type=self.cfg.new_model.finetune.LR_TYPE,
                                            lr_step=self.cfg.new_model.finetune.LR_STEP,
                                            lr_factor=self.cfg.new_model.finetune.LR_FACTOR,
                                            warmup_epochs=self.cfg.new_model.finetune.WARM_EPOCH,
                                            MAX_EPOCH=self.cfg.new_model.finetune.MAX_EPOCH)

        train_loader = DataLoader(dataset=train_dataset, batch_size=self.cfg.new_model.finetune.BATCH_SIZE,
                                  num_workers=self.cfg.new_model.finetune.NUM_WORKERS, shuffle=True, drop_last=True,
                                  persistent_workers=True)
        iter_num = len(train_loader)
        best_acc = 0.
        dw_cls = torch.ones(2 * self.cfg.new_model.finetune.BATCH_SIZE, dtype=torch.float32).to(self.device)
        dw_sum = dp_classes_num ** 2 + self.dataset_handler.classes_per_task ** 2
        dw_cls[:self.cfg.new_model.finetune.BATCH_SIZE] = self.dataset_handler.classes_per_task / dw_sum
        dw_cls[self.cfg.new_model.finetune.BATCH_SIZE:] = dp_classes_num / dw_sum
        sample_num_per_class = torch.ones(active_classes_num, dtype=torch.float32).to(self.device)
        sample_num_per_class[0:dp_classes_num] = self.cfg.new_model.finetune.BATCH_SIZE / dp_classes_num
        sample_num_per_class[
        dp_classes_num:active_classes_num] = self.cfg.new_model.finetune.BATCH_SIZE / self.dataset_handler.classes_per_task
        bsce_criterion = BalancedSoftmax(sample_num_per_class)
        for epoch in range(1, self.cfg.new_model.finetune.MAX_EPOCH + 1):
            all_loss = AverageMeter()
            if float(torch.__version__[:3]) < 1.3:
                ft_scheduler.step()
            iter_index = 0
            for x, y in train_loader:
                model.train()
                x, y = x.to(self.device), y.to(self.device)  # --> transfer them to correct device
                # ---> Train MAIN MODEL
                # data replay
                x_replay, y_replay, pre_logits_replay = self.sample(self.previous_teacher, len(x), dp_classes_num)
                assert y.shape[0] == y_replay.shape[0] == self.cfg.new_model.finetune.BATCH_SIZE
                x_com = torch.cat([x, x_replay], dim=0)
                y_com = torch.cat([y, y_replay], dim=0)
                if "FCC" in discription:
                    scale_factor_batch = []
                    for j in range(y_com.shape[0]):
                        scale_factor_batch.append(scale_factor[y_com[j]])
                    scale_factor_batch = torch.tensor(scale_factor_batch).to(self.device)

                    features_com = model(x_com, is_nograd=True, feature_flag=True)
                    raw_shape = features_com.shape
                    scale_factor_batch = scale_factor_batch.view(y_com.shape[0], 1)
                    features_com = features_com.view(y_com.shape[0], -1)
                    new_features = torch.mul(features_com, scale_factor_batch)
                    new_features = new_features.view(raw_shape)

                    logits_com = model(new_features, train_cls_use_features=True)
                    logits_com = logits_com[:, 0:active_classes_num]
                else:
                    logits_com = model(x_com, train_classifier=True)
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
                               "|| Batch_Loss:{:>5.3f} || discription: {}".format("EARS_DFCL", epoch,
                                                                                  iter_index,
                                                                                  iter_num,
                                                                                  ft_optimizer.param_groups[
                                                                                      0]['lr'],
                                                                                  all_loss.val,
                                                                                  discription)
                    self.logger.info(pbar_str)
                iter_index += 1
            if self.cfg.VALID_STEP != -1 and epoch % self.cfg.VALID_STEP == 0:

                val_acc = self.validate_with_FC(task=task_id, test_model=model)  # task_id 从1开始

                if val_acc.mean() > best_acc:
                    best_acc, best_epoch = val_acc.mean(), epoch
                    self.logger.info(
                        "--------------Best_Epoch:{:>3d}    Best_Acc:{:>5.2f}%--------------".format(
                            best_epoch, best_acc * 100
                        )
                    )

            if float(torch.__version__[:3]) >= 1.3:
                ft_scheduler.step()
        pass

    def evaluate_model(self, task, model, discritption=""):
        '''Evaluation.'''
        val_acc = self.validate_with_FC(task, test_model=model)
        taskIL_FC_val_acc = self.validate_with_FC_taskIL(task, test_model=model)
        # if "PCDFCL" == self.cfg.trainer_name and task > 1:
        #     val_acc = self.trainer.validate_with_NCM(task)
        #     taskIL_FC_val_acc = self.trainer.validate_with_NCM_taskIL(task)
        # else:
        #     val_acc = self.trainer.validate_with_FC(task)
        #     taskIL_FC_val_acc = self.trainer.validate_with_FC_taskIL(task)
        test_acc = None
        self.logger.info(f'{discritption}, #############task: {task:0>3d} is finished Test begin. ##############')
        if self.dataset_handler.val_datasets:
            test_acc = self.validate_with_FC(task, test_model=model, is_test=True)
            taskIL_FC_test_acc = self.validate_with_FC_taskIL(task, test_model=model, is_test=True)

            val_acc_FC_str = f'{discritption}, task: {task} classififer:{"FC"} val_acc: {val_acc}, avg: {val_acc.mean()} '
            test_acc_FC_str = f'{discritption}, task: {task} classififer:{"FC"} || test_acc: {test_acc}, avg: {test_acc.mean()} '
            self.logger.info(val_acc_FC_str)
            self.logger.info(test_acc_FC_str)
            self.logger.info(
                f"{discritption}, validate taskIL: val FC: {taskIL_FC_val_acc} || {taskIL_FC_val_acc.mean()}")
            self.logger.info(
                f"{discritption}, validate taskIL: test FC: {taskIL_FC_test_acc} || {taskIL_FC_test_acc.mean()}")
        else:
            test_acc_FC_str = f'{discritption}, task: {task} classififer:{"FC"} || test_acc: {val_acc}, avg: {val_acc.mean()} '
            self.logger.info(test_acc_FC_str)
            self.logger.info(f"{discritption}, validate taskIL: FC: {taskIL_FC_val_acc} || {taskIL_FC_val_acc.mean()}")

    def post_hoc_evaluate_model(self, task, model, scale_factor, discritption=""):
        '''Evaluation.'''
        val_acc = self.post_hoc_validate_with_FC(task, test_model=model, scale_factor=scale_factor)
        taskIL_FC_val_acc = self.post_hoc_validate_with_FC_taskIL(task, test_model=model, scale_factor=scale_factor)
        # if "PCDFCL" == self.cfg.trainer_name and task > 1:
        #     val_acc = self.trainer.validate_with_NCM(task)
        #     taskIL_FC_val_acc = self.trainer.validate_with_NCM_taskIL(task)
        # else:
        #     val_acc = self.trainer.validate_with_FC(task)
        #     taskIL_FC_val_acc = self.trainer.validate_with_FC_taskIL(task)
        test_acc = None
        self.logger.info(f'{discritption}, #############task: {task:0>3d} is finished Test begin. ##############')
        if self.dataset_handler.val_datasets:
            test_acc = self.post_hoc_validate_with_FC(task, test_model=model, scale_factor=scale_factor, is_test=True)
            taskIL_FC_test_acc = self.post_hoc_validate_with_FC_taskIL(task, test_model=model,
                                                                       scale_factor=scale_factor,
                                                                       is_test=True)

            val_acc_FC_str = f'{discritption}, task: {task} classififer:{"FC"} val_acc: {val_acc}, avg: {val_acc.mean()} '
            test_acc_FC_str = f'{discritption}, task: {task} classififer:{"FC"} || test_acc: {test_acc}, avg: {test_acc.mean()} '
            self.logger.info(val_acc_FC_str)
            self.logger.info(test_acc_FC_str)
            self.logger.info(
                f"{discritption}, validate taskIL: val FC: {taskIL_FC_val_acc} || {taskIL_FC_val_acc.mean()}")
            self.logger.info(
                f"{discritption}, validate taskIL: test FC: {taskIL_FC_test_acc} || {taskIL_FC_test_acc.mean()}")
        else:
            test_acc_FC_str = f'{discritption}, task: {task} classififer:{"FC"} || test_acc: {val_acc}, avg: {val_acc.mean()} '
            self.logger.info(test_acc_FC_str)
            self.logger.info(f"{discritption}, validate taskIL: FC: {taskIL_FC_val_acc} || {taskIL_FC_val_acc.mean()}")

    def post_hoc_validate_with_FC(self, task, test_model=None, scale_factor=None, is_test=False):
        acc = []
        model = test_model if test_model is not None else self.model
        mode = model.training
        model.eval()
        for task_id in range(task):  # 这里的task 从0 开始
            if self.dataset_handler.val_datasets and (not is_test):
                predict_result = self.post_hoc_validate_with_FC_per_task(model,
                                                                         self.dataset_handler.val_datasets[task_id],
                                                                         task, scale_factor)
            else:
                predict_result = self.post_hoc_validate_with_FC_per_task(model,
                                                                         self.dataset_handler.test_datasets[task_id],
                                                                         task, scale_factor)
            acc.append(predict_result)
            self.logger.info(
                f"task: {task} || per task {task_id}, validate_with_FC acc:{predict_result}"
            )
        acc = np.array(acc)
        model.train(mode=mode)
        return acc
        pass

    def post_hoc_validate_with_FC_per_task(self, model, val_dataset, task, scale_factor):
        # todo
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.cfg.model.TRAIN.BATCH_SIZE,
                                num_workers=self.cfg.model.TRAIN.NUM_WORKERS, shuffle=False, drop_last=False)
        top1 = AverageMeter()
        correct = 0
        active_classes_num = self.dataset_handler.classes_per_task * task
        for inputs, labels in val_loader:
            correct_temp = 0
            inputs, labels = inputs.to("cuda"), labels.to("cuda")
            scale_factor_batch = []
            for j in range(labels.shape[0]):
                scale_factor_batch.append(1 / scale_factor[labels[j]])
            scale_factor_batch = torch.tensor(scale_factor_batch).to(self.device)
            features_com = model(x=inputs, is_nograd=True, feature_flag=True)
            raw_shape = features_com.shape
            scale_factor_batch = scale_factor_batch.view(labels.shape[0], 1)
            features_com = features_com.view(labels.shape[0], -1)
            new_features = torch.mul(features_com, scale_factor_batch)
            new_features = new_features.view(raw_shape)
            out = model(x=new_features, is_nograd=True, get_out_use_features=True)
            out = out[:, 0:active_classes_num]
            _, balance_fc_y_hat = torch.max(out, 1)
            correct_temp += balance_fc_y_hat.eq(labels.data).cpu().sum()
            correct += correct_temp
            top1.update((correct_temp / inputs.size(0)).item(), inputs.size(0))
        return top1.avg
        pass

    def post_hoc_validate_with_FC_taskIL(self, task, test_model=None, scale_factor=None, is_test=False):
        acc = []
        model = test_model if test_model is not None else self.model
        mode = model.training
        model.eval()
        for task_id in range(task):  # 这里的task 从0 开始
            if self.dataset_handler.val_datasets and (not is_test):
                predict_result = self.post_hoc_validate_with_FC_per_task_taskIL(
                    self.dataset_handler.val_datasets[task_id],
                    task_id, model, scale_factor)
            else:
                predict_result = self.post_hoc_validate_with_FC_per_task_taskIL(
                    self.dataset_handler.test_datasets[task_id],
                    task_id, model, scale_factor)
            acc.append(predict_result)
            self.logger.info(
                f"task: {task} || per task {task_id}, validate_with_FC acc:{predict_result}"
            )
        acc = np.array(acc)
        self.model.train(mode=mode)
        return acc
        pass

    def post_hoc_validate_with_FC_per_task_taskIL(self, val_dataset, task_id, model, scale_factor):
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
            scale_factor_batch = []
            for j in range(labels.shape[0]):
                scale_factor_batch.append(1 / scale_factor[labels[j]])
            scale_factor_batch = torch.tensor(scale_factor_batch).to(self.device)
            features_com = model(x=inputs, is_nograd=True, feature_flag=True)
            raw_shape = features_com.shape
            scale_factor_batch = scale_factor_batch.view(labels.shape[0], 1)
            features_com = features_com.view(labels.shape[0], -1)
            new_features = torch.mul(features_com, scale_factor_batch)
            new_features = new_features.view(raw_shape)
            out = model(x=new_features, is_nograd=True, get_out_use_features=True)
            _, balance_fc_y_hat = torch.max(out[:, allowed_classes], 1)
            balance_fc_y_hat += task_id * self.dataset_handler.classes_per_task
            correct_temp += balance_fc_y_hat.eq(labels.data).cpu().sum()
            correct += correct_temp
            top1.update((correct_temp / inputs.size(0)).item(), inputs.size(0))
        return top1.avg
        pass

    def train_new_model(self, train_dataset, active_classes_num, task):
        dp_classes_num = active_classes_num - self.dataset_handler.classes_per_task
        if self.new_model is None:
            self.new_model = copy.deepcopy(self.model)
            # self.new_model = self.transfer_to_cuda(self.new_model)
            print("new_model:", self.new_model)
        alpha = log(self.dataset_handler.classes_per_task / 2 + 1, 2)
        beta2 = 1.0 * dp_classes_num / self.dataset_handler.classes_per_task
        beta = sqrt(beta2)
        # trains
        if "R-DFCIL" == self.cfg.new_model.loss_type:
            model_plus_rkd = nn.ModuleList([self.new_model, self.rkd])

            optimizer = self.build_optimize(model=model_plus_rkd,
                                            base_lr=self.cfg.new_model.TRAIN.OPTIMIZER.BASE_LR,
                                            optimizer_type=self.cfg.new_model.TRAIN.OPTIMIZER.TYPE,
                                            momentum=self.cfg.new_model.TRAIN.OPTIMIZER.MOMENTUM,
                                            weight_decay=self.cfg.new_model.TRAIN.OPTIMIZER.WEIGHT_DECAY)
            scheduler = self.build_scheduler(optimizer, lr_type=self.cfg.new_model.TRAIN.LR_SCHEDULER.TYPE,
                                             lr_step=self.cfg.new_model.TRAIN.LR_SCHEDULER.LR_STEP,
                                             lr_factor=self.cfg.new_model.TRAIN.LR_SCHEDULER.LR_FACTOR,
                                             warmup_epochs=self.cfg.new_model.TRAIN.LR_SCHEDULER.WARM_EPOCH,
                                             MAX_EPOCH=self.cfg.new_model.TRAIN.MAX_EPOCH)
        else:
            optimizer = self.build_optimize(model=self.new_model,
                                            base_lr=self.cfg.new_model.TRAIN.OPTIMIZER.BASE_LR,
                                            optimizer_type=self.cfg.new_model.TRAIN.OPTIMIZER.TYPE,
                                            momentum=self.cfg.new_model.TRAIN.OPTIMIZER.MOMENTUM,
                                            weight_decay=self.cfg.new_model.TRAIN.OPTIMIZER.WEIGHT_DECAY)
            scheduler = self.build_scheduler(optimizer, lr_type=self.cfg.new_model.TRAIN.LR_SCHEDULER.TYPE,
                                             lr_step=self.cfg.new_model.TRAIN.LR_SCHEDULER.LR_STEP,
                                             lr_factor=self.cfg.new_model.TRAIN.LR_SCHEDULER.LR_FACTOR,
                                             warmup_epochs=self.cfg.new_model.TRAIN.LR_SCHEDULER.WARM_EPOCH,
                                             MAX_EPOCH=self.cfg.new_model.TRAIN.MAX_EPOCH)
        MAX_EPOCH = self.cfg.new_model.TRAIN.MAX_EPOCH
        best_acc = 0

        cls_criterion = CrossEntropy().to(self.device)

        train_loader = DataLoader(dataset=train_dataset, batch_size=self.cfg.new_model.TRAIN.BATCH_SIZE,
                                  num_workers=self.cfg.new_model.TRAIN.NUM_WORKERS, shuffle=True, drop_last=True,
                                  persistent_workers=True)
        iter_num = len(train_loader)
        sample_num_per_class = torch.ones(active_classes_num, dtype=torch.float32).to(self.device)
        sample_num_per_class[:dp_classes_num] = self.cfg.new_model.TRAIN.BATCH_SIZE / dp_classes_num
        sample_num_per_class[
        dp_classes_num:] = self.cfg.new_model.TRAIN.BATCH_SIZE / self.dataset_handler.classes_per_task

        bsce_criterion = BalancedSoftmax(sample_per_class=sample_num_per_class).to(self.device)
        for epoch in range(1, MAX_EPOCH + 1):
            all_loss = AverageMeter()
            if float(torch.__version__[:3]) < 1.3:
                scheduler.step()
            iter_index = 0
            for x, y in train_loader:
                self.new_model.train()
                x, y = x.to(self.device), y.to(self.device)  # --> transfer them to correct device
                # ---> Train MAIN MODEL
                # data replay
                x_replay, y_replay, pre_logits_replay = self.sample(self.previous_teacher, len(x), dp_classes_num)
                pre_logits = self.previous_teacher.generate_scores(x, active_classes_num=dp_classes_num)
                x_com = torch.cat([x, x_replay], dim=0)
                y_com = torch.cat([y, y_replay], dim=0)
                pre_logits_com = torch.cat([pre_logits, pre_logits_replay], dim=0)

                mappings = torch.ones(y_com.size(), dtype=torch.float32).to(self.device)
                rnt = 1.0 * dp_classes_num / active_classes_num
                mappings[:dp_classes_num] = rnt
                mappings[dp_classes_num:] = 1 - rnt
                dw_cls = mappings[y_com.long()]

                logits_com, features_com = self.new_model(x_com)
                logits_com = logits_com[:, 0:active_classes_num]

                if "ABD" == self.cfg.new_model.loss_type:
                    total_loss = self.ABD_loss(x_com=x_com, y_com=y_com, y=y, logits_com=logits_com,
                                               features_com=features_com, pre_logits_com=pre_logits_com,
                                               cls_criterion=cls_criterion, dp_classes_num=dp_classes_num,
                                               active_classes_num=active_classes_num, dw_cls=dw_cls)
                elif "BSCE" == self.cfg.new_model.loss_type:
                    total_loss = self.BSCE_loss(x_com=x_com, y_com=y_com, y=y, logits_com=logits_com,
                                                features_com=features_com, pre_logits_com=pre_logits_com,
                                                cls_criterion=cls_criterion, bsce_criterion=bsce_criterion,
                                                dp_classes_num=dp_classes_num, active_classes_num=active_classes_num,
                                                dw_cls=dw_cls)
                    pass
                elif "LwF" == self.cfg.new_model.loss_type:
                    # total_loss = self.LwF_loss(y_com=y_com, logits_com=logits_com, pre_logits_com=pre_logits_com,
                    #                            cls_criterion=cls_criterion, dp_classes_num=dp_classes_num, rnt=rnt)
                    total_loss = self.LwF_loss(y_com=y_com, logits_com=logits_com, pre_logits_com=pre_logits_com,
                                               cls_criterion=bsce_criterion, dp_classes_num=dp_classes_num, rnt=rnt)
                    pass
                elif "R-DFCIL" == self.cfg.new_model.loss_type:
                    pre_features_new_classes = self.previous_teacher.generate_features(x)
                    # y, logits_com, pre_logits_replay, features_new_classes,
                    # pre_features_new_classes, cls_criterion, dp_classes_num, active_classes_num, alpha, beta
                    total_loss = self.RDFCIL_loss(y=y, logits_com=logits_com,
                                                  pre_logits_replay=pre_logits_replay,
                                                  features_new_classes=features_com[:y.shape[0], :],
                                                  pre_features_new_classes=pre_features_new_classes,
                                                  cls_criterion=cls_criterion, dp_classes_num=dp_classes_num,
                                                  active_classes_num=active_classes_num, alpha=alpha, beta=beta)
                elif "CFIL" == self.cfg.new_model.loss_type:
                    total_loss = self.CFIL_loss(y=y, logits_com=logits_com, pre_logits=pre_logits_replay,
                                                cls_criterion=cls_criterion, dp_classes_num=dp_classes_num,
                                                active_classes_num=active_classes_num)

                elif "SSIL" == self.cfg.new_model.loss_type:
                    total_loss = self.SSIL_loss(y=y, y_replay=y_replay, logits_com=logits_com,
                                                premodel_logits=pre_logits_com, cls_criterion=cls_criterion,
                                                dp_classes_num=dp_classes_num,
                                                active_classes_num=active_classes_num,
                                                task=task)
                else:
                    raise ValueError("self.cfg.model.loss_type is illegal.")

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                # measure accuracy and record loss
                all_loss.update(total_loss.data.item(), y_com.shape[0])
                if iter_index % self.cfg.SHOW_STEP == 0:
                    pbar_str = "Approach: {} || {} || Epoch: {} || Batch:{:>3d}/{}|| lr: {} " \
                               "|| Batch_Loss:{:>5.3f}".format("New model Train", self.cfg.new_model.loss_type,
                                                               epoch, iter_index,
                                                               iter_num,
                                                               optimizer.param_groups[
                                                                   0]['lr'],
                                                               all_loss.val
                                                               )
                    self.logger.info(pbar_str)
                iter_index += 1
            if self.cfg.VALID_STEP != -1 and epoch % self.cfg.VALID_STEP == 0:

                val_acc = self.validate_with_FC(task=task, test_model=self.new_model)  # task_id 从1开始

                if val_acc.mean() > best_acc:
                    best_acc, best_epoch = val_acc.mean(), epoch
                    self.logger.info(
                        "--------------Best_Epoch:{:>3d}    Best_Acc:{:>5.2f}%--------------".format(
                            best_epoch, best_acc * 100
                        )
                    )

            if float(torch.__version__[:3]) >= 1.3:
                scheduler.step()
        pass

    def ERA_train_model(self, train_dataset, active_classes_num, task):

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
        cls_criterion = CrossEntropy().to(self.device)
        dw_kd = torch.ones(sample_num_grade_classes.size(), dtype=torch.float32).to(self.device)
        rnt = 1.0 * dpt_classes_num / active_classes_num
        dw_kd[:-1] = 1.
        dw_kd[-1] = self.dataset_handler.classes_per_task / dpt_classes_num
        mappings = torch.ones(int(2 * self.cfg.model.TRAIN.BATCH_SIZE), dtype=torch.float32).to(self.device)
        mappings[:self.cfg.model.TRAIN.BATCH_SIZE] = 1 - rnt
        mappings[self.cfg.model.TRAIN.BATCH_SIZE:] = rnt
        dw_cls = mappings

        sample_num_per_class = torch.ones(active_classes_num, dtype=torch.float32)
        sample_num_per_class[
        :dpt_classes_num] = self.cfg.model.TRAIN.BATCH_SIZE / dpt_classes_num
        sample_num_per_class[dpt_classes_num:] = self.cfg.model.TRAIN.BATCH_SIZE / self.dataset_handler.classes_per_task

        for epoch in range(1, self.cfg.model.TRAIN.MAX_EPOCH + 1):
            all_loss = AverageMeter()
            if float(torch.__version__[:3]) < 1.3:
                scheduler.step()
            iter_index = 0
            for x, y in train_loader:
                self.model.train()
                self.new_model.eval()
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
                    FCTM_outs, FCTM_features = self.new_model(all_x)
                FCTM_outputs = FCTM_outs[:, 0:active_classes_num]

                loss_KD = torch.zeros(task).to(self.device)
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

                self.new_model.zero_grad()
                logits_KD = self.new_model(x=all_features, train_cls_use_features=True)
                logits_KD = logits_KD[:, :active_classes_num]
                logits_KD_past = FCTM_outputs
                # # loss_kd += self.cfg.FCTM.kd_lambda * (
                # #     self.kd_criterion(logits_KD, logits_KD_past).sum(dim=1)).mean() / (
                # #               logits_KD.size(1))
                loss_kd += 0.1 * (
                    self.kd_criterion(logits_KD, logits_KD_past).sum(dim=1)).mean() / (
                               logits_KD.size(1))
                # loss_kd += 0.5 * (
                #     self.kd_criterion(logits_KD, logits_KD_past).sum(dim=1)).mean() / (
                #                logits_KD.size(1))

                loss_cls = cls_criterion(all_outputs[:self.cfg.model.TRAIN.BATCH_SIZE,
                                         -self.dataset_handler.classes_per_task:],
                                         all_y[:self.cfg.model.TRAIN.BATCH_SIZE] %
                                         self.dataset_handler.classes_per_task) * (1 - rnt)

                FCTM_output_for_pre_distill = FCTM_outputs[:, 0:dpt_classes_num]
                loss_cls += compute_distill_loss(all_outputs[:, 0:dpt_classes_num], FCTM_output_for_pre_distill,
                                                 temp=self.cfg.model.T) * rnt

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
        pass

    def ABD_loss(self, x_com, y_com, y, logits_com, features_com, pre_logits_com, cls_criterion, dp_classes_num,
                 active_classes_num, dw_cls):
        loss_class = cls_criterion(logits_com[:y.shape[0], dp_classes_num:active_classes_num],
                                   y % self.dataset_handler.classes_per_task)

        with torch.no_grad():
            feat_class = self.new_model(x_com, is_nograd=True, feature_flag=True).detach()
        ft_logits = self.new_model(feat_class, train_cls_use_features=True)
        ft_logits = ft_logits[:, 0:active_classes_num]
        loss_class += self.ft_criterion(ft_logits, y_com, dw_cls)

        logits_KD = self.previous_teacher.generate_score_by_features(features_com,
                                                                     active_classes_num=dp_classes_num,
                                                                     is_nograd=False)
        logits_KD_past = pre_logits_com
        loss_kd = self.cfg.new_model.kd_lambda * (
            self.kd_criterion(logits_KD, logits_KD_past).sum(dim=1)).mean() / (
                      logits_KD.size(1))

        total_loss = loss_class + loss_kd
        return total_loss

    # mappings = torch.ones(y_com.size(), dtype=torch.float32).to(self.device)
    # rnt = 1.0 * dp_classes_num / active_classes_num
    # mappings[:dp_classes_num] = rnt
    # mappings[dp_classes_num:] = 1 - rnt
    # dw_cls = mappings[y_com.long()]
    #
    # logits_com, features_com = self.model(x_com)
    # logits_com = logits_com[:, 0:active_classes_num]
    #
    # loss_class = criterion(logits_com[:y.shape[0], dp_classes_num:active_classes_num],
    #                        y % self.dataset_handler.classes_per_task) * (1 - rnt)
    #
    # with torch.no_grad():
    #     feat_class = self.model(x_com, is_nograd=True, feature_flag=True).detach()
    # ft_logits = self.model(feat_class, train_cls_use_features=True)
    # ft_logits = ft_logits[:, 0:active_classes_num]
    # loss_class += self.ft_criterion(ft_logits, y_com, dw_cls)
    #
    # logits_KD = self.previous_teacher.generate_score_by_features(features_com,
    #                                                              active_classes_num=dp_classes_num,
    #                                                              is_nograd=False)
    # logits_KD_past = pre_logits_com
    # loss_kd = self.cfg.model.kd_lambda * (
    #     self.kd_criterion(logits_KD, logits_KD_past).sum(dim=1)).mean() / (
    #               logits_KD.size(1))

    def RDFCIL_loss(self, y, logits_com, pre_logits_replay, features_new_classes, pre_features_new_classes,
                    cls_criterion, dp_classes_num, active_classes_num, alpha, beta):

        loss_class = cls_criterion(logits_com[:y.shape[0], dp_classes_num:active_classes_num],
                                   y % self.dataset_handler.classes_per_task) * (
                             self.cfg.new_model.ce_lambda * (1 + 1 / alpha) / beta)
        logits_old_classes = logits_com[y.shape[0]:, :]
        logits_old_classes = logits_old_classes[:, :dp_classes_num]
        loss_hkd = (self.kd_criterion(logits_old_classes, pre_logits_replay).sum(dim=1)).mean() / (
            logits_old_classes.size(1)) * self.cfg.new_model.hkd_lambda

        loss_rkd = self.rkd(student=features_new_classes,
                            teacher=pre_features_new_classes) * self.cfg.new_model.rkd_lambda
        loss_kd = (loss_hkd + loss_rkd) * (alpha * beta)

        total_loss = loss_class + loss_kd
        return total_loss

    # loss_class = criterion(logits_new_classes, y % self.dataset_handler.classes_per_task) \
    #              * (self.cfg.model.ce_lambda * (1 + 1 / alpha) / beta)
    #
    # logits_old_classes, features_old_classes = self.model(x_replay)
    # logits_old_classes = logits_old_classes[:, :dp_classes_num]
    # loss_hkd = (self.kd_criterion(logits_old_classes, pre_logits_replay).sum(dim=1)).mean() / (
    #     logits_old_classes.size(1)) * self.cfg.model.hkd_lambda
    #
    # loss_rkd = self.rkd(student=features_new_classes,
    #                     teacher=pre_features_new_classes) * self.cfg.model.rkd_lambda
    # loss_kd = (loss_hkd + loss_rkd) * (alpha * beta)
    #
    # total_loss = loss_class + loss_kd

    def BSCE_loss(self, x_com, y_com, y, logits_com, features_com, pre_logits_com, cls_criterion,
                  bsce_criterion, dp_classes_num, active_classes_num, dw_cls):
        loss_class = cls_criterion(logits_com[:y.shape[0], dp_classes_num:active_classes_num],
                                   y % self.dataset_handler.classes_per_task)
        with torch.no_grad():
            feat_class = self.new_model(x_com, is_nograd=True, feature_flag=True).detach()
        ft_logits = self.new_model(feat_class, train_cls_use_features=True)
        ft_logits = ft_logits[:, 0:active_classes_num]
        # loss_class += self.ft_criterion(ft_logits, y_com, dw_cls)
        loss_class += bsce_criterion(ft_logits, y_com)

        # logits_KD = self.previous_teacher.generate_score_by_features(features_com,
        #                                                              active_classes_num=dp_classes_num,
        #                                                              is_nograd=False)
        # logits_KD_past = pre_logits_com
        # loss_kd = self.cfg.new_model.kd_lambda * (
        #     self.kd_criterion(logits_KD, logits_KD_past).sum(dim=1)).mean() / (
        #               logits_KD.size(1))
        logits_KD = logits_com[:, 0:dp_classes_num]
        logits_KD_past = pre_logits_com
        loss_kd = compute_distill_loss(output_for_distill=logits_KD, previous_task_model_output=logits_KD_past,
                                       temp=self.cfg.new_model.T)

        total_loss = loss_class + loss_kd
        return total_loss

    def LwF_loss(self, y_com, logits_com, pre_logits_com, cls_criterion, dp_classes_num, rnt):
        loss_class = cls_criterion(logits_com, y_com) * (1 - rnt)
        logits_KD = logits_com[:, 0:dp_classes_num]
        logits_KD_past = pre_logits_com
        loss_kd = compute_distill_loss(output_for_distill=logits_KD, previous_task_model_output=logits_KD_past,
                                       temp=self.cfg.new_model.T) * rnt

        total_loss = loss_class + loss_kd
        return total_loss

    def CFIL_loss(self, y, logits_com, pre_logits, cls_criterion, dp_classes_num, active_classes_num):
        loss_class = cls_criterion(logits_com[:y.shape[0], dp_classes_num:active_classes_num],
                                   y % self.dataset_handler.classes_per_task)
        logits_KD = logits_com[y.shape[0]:, 0:dp_classes_num]
        logits_KD_past = pre_logits
        loss_kd = self.cfg.new_model.kd_lambda * (
            self.kd_criterion(logits_KD, logits_KD_past).sum(dim=1)).mean() / (
                      logits_KD.size(1))

        total_loss = loss_class + loss_kd
        return total_loss

    def SSIL_loss(self, y, y_replay, logits_com, premodel_logits, cls_criterion, dp_classes_num, active_classes_num, task):
        curr_cls_loss = cls_criterion(logits_com[:y.shape[0], dp_classes_num:active_classes_num],
                                   y % self.dataset_handler.classes_per_task)
        exemplar_cls_loss = cls_criterion(logits_com[y.shape[0]:, :dp_classes_num], y_replay)
        cls_loss = curr_cls_loss + exemplar_cls_loss

        all_ouput_for_dis = logits_com[:, :dp_classes_num]

        all_pre_model_output_for_dis = premodel_logits

        loss_KD = torch.zeros(task).cuda()
        for task_id in range(task - 1):
            task_id_ouput = all_ouput_for_dis[:, self.dataset_handler.classes_per_task * task_id:
                                                 self.dataset_handler.classes_per_task * (task_id + 1)]
            task_id_pre_model_output = all_pre_model_output_for_dis[:,
                                       self.dataset_handler.classes_per_task * task_id:
                                       self.dataset_handler.classes_per_task * (task_id + 1)]
            soft_target = Func.softmax(task_id_pre_model_output / self.cfg.new_model.T, dim=1)
            output_log = Func.log_softmax(task_id_ouput / self.cfg.new_model.T, dim=1)
            loss_KD[task_id] = Func.kl_div(output_log, soft_target, reduction='batchmean') * (self.cfg.new_model.T ** 2)
        loss_KD = loss_KD.sum()
        total_loss = cls_loss + loss_KD
        return total_loss

    def idendity_model(self):
        self.model = copy.deepcopy(self.new_model)
        return

    def train_model(self, train_dataset, active_classes_num, task):
        # self.construct_datasets_for_all(active_classes_num, train_dataset)
        self.model = copy.deepcopy(self.new_model)
        if self.classifier is None:
            self.classifier = NcmClassifier(self.cfg.extractor.output_feature_dim)
            self.classifier = self.transfer_to_cuda(self.classifier)
        self.model_optim_params_list = []
        self.model_optim_params_list.append(
            {'params': self.classifier.parameters(), 'lr': self.cfg.model.cdt.lr,
             'momentum': self.cfg.model.cdt.momentum,
             'weight_decay': self.cfg.model.cdt.weight_decay})

        # self.compute_centroids(active_classes_num=active_classes_num)
        self.get_prototype(active_classes_num=active_classes_num)
        self.centroids.requires_grad = True

        self.logger.info(f"self.centroids: {self.centroids.detach()}")
        self.logger.info(f"self.classifier.T: {self.classifier.T.detach()}")
        # print(self.classifier.T)
        self.model_optim_params_list.append({'params': self.centroids, 'lr': self.cfg.model.centroids.lr,
                                             'momentum': self.cfg.model.centroids.momentum})

        model_optimizer = torch.optim.SGD(self.model_optim_params_list)
        model_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optimizer, self.cfg.model.TRAIN.MAX_EPOCH,
                                                                     eta_min=0)

        dpt_classes_num = active_classes_num - self.dataset_handler.classes_per_task
        sample_num_grade_classes = torch.from_numpy(
            np.array([active_classes_num - self.dataset_handler.classes_per_task,
                      self.dataset_handler.classes_per_task])).long().to(self.device)

        train_loader = DataLoader(dataset=train_dataset, batch_size=self.cfg.model.TRAIN.BATCH_SIZE,
                                  num_workers=self.cfg.model.TRAIN.NUM_WORKERS, shuffle=True, drop_last=True,
                                  persistent_workers=True)
        iter_num = len(train_loader)
        best_acc = 0.
        dw_kd = torch.ones(sample_num_grade_classes.size(), dtype=torch.float32).to(self.device)
        rnt = 1.0 * dpt_classes_num / active_classes_num
        dw_kd[:-1] = 1.
        dw_kd[-1] = self.dataset_handler.classes_per_task / dpt_classes_num
        mappings = torch.ones(2 * self.cfg.model.TRAIN.BATCH_SIZE, dtype=torch.float32).to(self.device)
        mappings[:self.cfg.model.TRAIN.BATCH_SIZE] = 1 - rnt
        mappings[self.cfg.model.TRAIN.BATCH_SIZE:] = rnt
        dw_cls = mappings

        data_weights = torch.ones(2 * self.cfg.model.TRAIN.BATCH_SIZE, dtype=torch.float32).to(self.device)
        dw_sum = dpt_classes_num ** 2 + self.dataset_handler.classes_per_task ** 2
        data_weights[:self.cfg.model.TRAIN.BATCH_SIZE] = self.dataset_handler.classes_per_task / dw_sum
        data_weights[self.cfg.model.TRAIN.BATCH_SIZE:] = dpt_classes_num / dw_sum

        sample_num_per_class = torch.ones(active_classes_num, dtype=torch.float32).to(self.device)
        sample_num_per_class[:dpt_classes_num] = self.cfg.model.TRAIN.BATCH_SIZE / dpt_classes_num
        sample_num_per_class[dpt_classes_num:] = self.cfg.model.TRAIN.BATCH_SIZE / self.dataset_handler.classes_per_task
        bsce_criterion = BalancedSoftmax(sample_per_class=sample_num_per_class, tau=self.cfg.model.tau).to(self.device)
        cls_criterion = CrossEntropy().to(self.device)

        for epoch in range(1, self.cfg.model.TRAIN.MAX_EPOCH + 1):
            all_loss = AverageMeter()
            if float(torch.__version__[:3]) < 1.3:
                model_scheduler.step()
            iter_index = 0
            for x, y in train_loader:
                self.model.train()
                self.new_model.eval()
                x, y = x.to(self.device), y.to(self.device)  # --> transfer them to correct device
                # ---> Train MAIN MODEL
                # data replay
                x_replay, y_replay, pre_logits_replay = self.sample(self.previous_teacher, len(x), dpt_classes_num)

                all_x = torch.cat([x, x_replay], dim=0)
                all_y = torch.cat([y, y_replay], dim=0)
                assert all_y.shape[0] == 2 * self.cfg.model.TRAIN.BATCH_SIZE

                _, all_features = self.model(all_x, is_nograd=True)
                all_logits = self.classifier(all_features, self.centroids)

                # loss_cls = bsce_criterion(input=all_logits, label=all_y)

                # loss_cls = cls_criterion(input=all_logits, label=all_y)

                loss_cls = self.ft_criterion(all_logits, all_y, data_weights)

                total_loss = loss_cls
                model_optimizer.zero_grad()
                total_loss.backward()
                model_optimizer.step()

                all_loss.update(total_loss.data.item(), y.shape[0])
                if iter_index % self.cfg.SHOW_STEP == 0:
                    pbar_str = "Approach: {}| | Epoch: {} || Batch:{:>3d}/{}|| lr: {} " \
                               "|| Batch_Loss:{:>5.3f}".format("EARS_DFCL", epoch, iter_index,
                                                               iter_num,
                                                               model_scheduler.get_last_lr()[-1],
                                                               all_loss.val
                                                               )
                    self.logger.info(pbar_str)
                iter_index += 1
            if self.cfg.VALID_STEP != -1 and epoch % self.cfg.VALID_STEP == 0:
                val_acc = self.validate_with_NCM(task=task)  # task_id 从1开始
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
                model_scheduler.step()
        self.logger.info(f"train end, self.centroids: {self.centroids.detach()}")
        self.logger.info(f"train end, self.classifier.T: {self.classifier.T.detach()}")

    def FCC_train_model(self, train_dataset, active_classes_num, task):
        dpt_classes_num = active_classes_num - self.dataset_handler.classes_per_task

        self.model = copy.deepcopy(self.new_model)
        self.model.train()

        if hasattr(self.new_model, "module"):
            global_classifier_weights, global_classifier_bias = \
                get_classifier_weight_bias(self.new_model.module.linear_classifier)
        else:
            global_classifier_weights, global_classifier_bias = \
                get_classifier_weight_bias(self.new_model.linear_classifier)
        global_classifier_weights_norm = torch.norm(global_classifier_weights.detach(), 2, 1)
        global_classifier_weights_norm = global_classifier_weights_norm[0: active_classes_num]
        global_classifier_weights_norm_max = global_classifier_weights_norm.max()

        self.logger.info(f"global_classifier_weights_norm: {global_classifier_weights_norm}")
        self.logger.info(f"global_classifier_weights_norm mean of old classes: "
                         f"{torch.mean(global_classifier_weights_norm[0:dpt_classes_num])}")
        self.logger.info(f"global_classifier_weights_norm mean of new classes: "
                         f"{torch.mean(global_classifier_weights_norm[dpt_classes_num: active_classes_num])}")
        self.logger.info(f"global_classifier_weights_norm max: "
                         f"{global_classifier_weights_norm_max}")

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
        cls_criterion = CrossEntropy().to(self.device)
        rnt = 1.0 * dpt_classes_num / active_classes_num
        mappings = torch.ones(int(2 * self.cfg.model.TRAIN.BATCH_SIZE), dtype=torch.float32).to(self.device)
        mappings[:self.cfg.model.TRAIN.BATCH_SIZE] = 1 - rnt
        mappings[self.cfg.model.TRAIN.BATCH_SIZE:] = rnt
        dw_cls = mappings
        scale_factor = torch.ones(active_classes_num, dtype=torch.float32)
        if self.cfg.model.use_feature_norm:
            temp = global_classifier_weights_norm / global_classifier_weights_norm_max
            for class_index in range(active_classes_num):
                scale_factor[class_index] = 1 + self.cfg.model.gamma * temp[class_index]
        else:
            for class_index in range(active_classes_num):
                in_task = (class_index // self.dataset_handler.classes_per_task) + 1
                scale_factor[class_index] = 1 + self.cfg.model.gamma * (in_task / task)

        for epoch in range(1, self.cfg.model.TRAIN.MAX_EPOCH + 1):
            all_loss = AverageMeter()
            if float(torch.__version__[:3]) < 1.3:
                scheduler.step()
            iter_index = 0
            for x, y in train_loader:
                self.new_model.train()
                x, y = x.to(self.device), y.to(self.device)  # --> transfer them to correct device
                # ---> Train MAIN MODEL
                # data replay
                x_replay, y_replay, pre_logits_replay = self.sample(self.previous_teacher, len(x), dpt_classes_num)
                x_com = torch.cat([x, x_replay], dim=0)
                y_com = torch.cat([y, y_replay], dim=0)

                scale_factor_batch = []
                for j in range(y_com.shape[0]):
                    scale_factor_batch.append(scale_factor[y_com[j]])
                scale_factor_batch = torch.tensor(scale_factor_batch).to(self.device)

                logit_com, features_com = self.model(x_com)
                logit_com = logit_com[:, 0:active_classes_num]

                raw_shape = features_com.shape
                scale_factor_batch = scale_factor_batch.view(y_com.shape[0], 1)
                features_com = features_com.view(y_com.shape[0], -1)
                FCC_features = torch.mul(features_com, scale_factor_batch)
                FCC_features = FCC_features.view(raw_shape)
                features_com = features_com.view(raw_shape)

                FCC_logits_com = self.model(FCC_features, train_cls_use_features=True)

                FCC_logits_com = FCC_logits_com[:, 0:active_classes_num]

                if "ABD" == self.cfg.model.loss_type:
                    pre_logits = self.previous_teacher.generate_scores(x, active_classes_num=dpt_classes_num)
                    pre_logits_com = torch.cat([pre_logits, pre_logits_replay], dim=0)

                    loss_class = cls_criterion(FCC_logits_com[:y.shape[0], dpt_classes_num:active_classes_num],
                                               y % self.dataset_handler.classes_per_task)

                    with torch.no_grad():
                        feat_class = self.model(x_com, is_nograd=True, feature_flag=True).detach()
                        feat_class = feat_class.view(y_com.shape[0], -1)
                        new_feat_class = torch.mul(feat_class, scale_factor_batch)
                        new_feat_class = new_feat_class.view(raw_shape)
                    ft_logits = self.model(new_feat_class, train_cls_use_features=True)
                    ft_logits = ft_logits[:, 0:active_classes_num]
                    loss_class += self.ft_criterion(ft_logits, y_com, dw_cls)

                    if self.cfg.model.use_FCCKD:
                        logits_KD = self.previous_teacher.generate_score_by_features(FCC_features,
                                                                                     active_classes_num=dpt_classes_num,
                                                                                     is_nograd=False)
                    else:

                        logits_KD = self.previous_teacher.generate_score_by_features(features_com,
                                                                                     active_classes_num=dpt_classes_num,
                                                                                     is_nograd=False)
                    logits_KD_past = pre_logits_com
                    loss_kd = self.cfg.model.kd_lambda * (
                        self.kd_criterion(logits_KD, logits_KD_past).sum(dim=1)).mean() / (
                                  logits_KD.size(1))

                    total_loss = loss_class + loss_kd
                elif "calibration" == self.cfg.model.loss_type:
                    new_model_logits, _ = self.new_model(x_com, is_nograd=True)
                    new_model_logits = new_model_logits[:, 0:active_classes_num]
                    if self.cfg.model.use_localCE:
                        if self.cfg.model.use_localKD:
                            loss_class = cls_criterion(FCC_logits_com[:y.shape[0], dpt_classes_num:active_classes_num],
                                                       y % self.dataset_handler.classes_per_task) * (1 - rnt)
                            loss_kd = compute_distill_loss(output_for_distill=FCC_logits_com[:, 0:dpt_classes_num],
                                                           previous_task_model_output=new_model_logits[:,
                                                                                      0:dpt_classes_num],
                                                           temp=self.cfg.model.T) * rnt
                        else:
                            loss_class = cls_criterion(FCC_logits_com[:y.shape[0], dpt_classes_num:active_classes_num],
                                                       y % self.dataset_handler.classes_per_task)
                            loss_kd = 0.
                    else:
                        if self.cfg.model.use_localKD:
                            loss_class = 0.
                            loss_kd = compute_distill_loss(output_for_distill=FCC_logits_com[:, 0:dpt_classes_num],
                                                           previous_task_model_output=new_model_logits[:,
                                                                                      0:dpt_classes_num],
                                                           temp=self.cfg.model.T)
                        else:
                            loss_class = 0.
                            loss_kd = 0.
                    # loss_kd += self.cfg.model.kd_lambda * (
                    #     self.kd_criterion(FCC_logits_com, new_model_logits).sum(dim=1)).mean() / (
                    #                FCC_logits_com.size(1))
                    if self.cfg.model.use_post_LAKD:
                        loss_kd += self.cfg.model.kd_lambda * compute_posthoc_LAKDLoss(
                            output_for_distill=FCC_logits_com,
                            global_model_output_for_distill=new_model_logits,
                            sample_num_per_class=global_classifier_weights_norm,
                            temp=self.cfg.model.T)
                    else:
                        # loss_kd += self.cfg.model.kd_lambda * (
                        #     self.kd_criterion(FCC_logits_com, new_model_logits).sum(dim=1)).mean() / (
                        #                FCC_logits_com.size(1))
                        loss_kd += self.cfg.model.kd_lambda * \
                                   compute_distill_loss(output_for_distill=FCC_logits_com,
                                                        previous_task_model_output=new_model_logits,
                                                        temp=self.cfg.model.T)

                    # loss_kd += self.cfg.model.kd_lambda * ((self.kd_criterion(FCC_logits_com,
                    #                                                               new_model_logits).sum(dim=1)) *
                    #                                            dw_cls).mean() / (FCC_logits_com.size(1))

                    # loss_kd += self.cfg.model.kd_lambda * \
                    #            compute_distill_loss(output_for_distill=new_logits_com,
                    #                                 previous_task_model_output=new_model_logits,
                    #                                 temp=self.cfg.model.T)
                    total_loss = loss_class + loss_kd
                elif "NoFCC-calibration" == self.cfg.model.loss_type:
                    new_model_logits, _ = self.new_model(x_com, is_nograd=True)
                    new_model_logits = new_model_logits[:, 0:active_classes_num]
                    if self.cfg.model.use_localCE:
                        if self.cfg.model.use_localKD:
                            loss_class = cls_criterion(logit_com[:y.shape[0], dpt_classes_num:active_classes_num],
                                                       y % self.dataset_handler.classes_per_task) * (1 - rnt)
                            loss_kd = compute_distill_loss(output_for_distill=logit_com[:, 0:dpt_classes_num],
                                                           previous_task_model_output=new_model_logits[:,
                                                                                      0:dpt_classes_num],
                                                           temp=self.cfg.model.T) * rnt
                        else:
                            loss_class = cls_criterion(logit_com[:y.shape[0], dpt_classes_num:active_classes_num],
                                                       y % self.dataset_handler.classes_per_task)
                            loss_kd = 0.
                    else:
                        if self.cfg.model.use_localKD:
                            loss_class = 0.
                            loss_kd = compute_distill_loss(output_for_distill=logit_com[:, 0:dpt_classes_num],
                                                           previous_task_model_output=new_model_logits[:,
                                                                                      0:dpt_classes_num],
                                                           temp=self.cfg.model.T)
                        else:
                            loss_class = 0.
                            loss_kd = 0.

                    # loss_kd += self.cfg.model.kd_lambda * (
                    #     self.kd_criterion(logit_com, new_model_logits).sum(dim=1)).mean() / (
                    #                logit_com.size(1))
                    if self.cfg.model.use_post_LAKD:
                        loss_kd += self.cfg.model.kd_lambda * compute_posthoc_LAKDLoss(output_for_distill=logit_com,
                                                                                       global_model_output_for_distill=new_model_logits,
                                                                                       sample_num_per_class=global_classifier_weights_norm,
                                                                                       temp=self.cfg.model.T)
                    else:
                        # loss_kd += self.cfg.model.kd_lambda * (
                        #     self.kd_criterion(logit_com, new_model_logits).sum(dim=1)).mean() / (
                        #                logit_com.size(1))

                        loss_kd += self.cfg.model.kd_lambda * \
                                   compute_distill_loss(output_for_distill=logit_com,
                                                        previous_task_model_output=new_model_logits,
                                                        temp=self.cfg.model.T)

                    # loss_kd += self.cfg.model.kd_lambda * ((self.kd_criterion(logit_com,
                    #                                                               new_model_logits).sum(dim=1)) *
                    #                                            dw_cls).mean() / (logit_com.size(1))

                    # loss_kd += self.cfg.model.kd_lambda * \
                    #            compute_distill_loss(output_for_distill=new_logits_com,
                    #                                 previous_task_model_output=new_model_logits,
                    #                                 temp=self.cfg.model.T)
                    total_loss = loss_class + loss_kd
                elif "com-calibration" == self.cfg.model.loss_type:

                    loss_class = cls_criterion(FCC_logits_com[:y.shape[0], dpt_classes_num:active_classes_num],
                                               y % self.dataset_handler.classes_per_task) * (1 - rnt)

                    pre_logits = self.previous_teacher.generate_scores(x, active_classes_num=dpt_classes_num)
                    pre_logits_com = torch.cat([pre_logits, pre_logits_replay], dim=0)

                    new_model_logits, _ = self.new_model(x_com, is_nograd=True)
                    new_model_logits = new_model_logits[:, 0:active_classes_num]

                    loss_kd = compute_distill_loss(output_for_distill=FCC_logits_com[:, 0:dpt_classes_num],
                                                   previous_task_model_output=pre_logits_com,
                                                   temp=self.cfg.model.T) * rnt
                    loss_kd += (self.kd_criterion(FCC_logits_com, new_model_logits).sum(dim=1)).mean() / (
                        FCC_logits_com.size(1))

                    # loss_kd += self.cfg.new_model.kd_lambda * \
                    #            compute_distill_loss(output_for_distill=new_logits_com,
                    #                                 previous_task_model_output=new_model_logits,
                    #                                 temp=self.cfg.model.T)
                    total_loss = loss_kd + loss_class
                else:
                    raise ValueError(f"self.cfg.model.loss_type is illegal.")

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                # measure accuracy and record loss
                all_loss.update(total_loss.data.item(), y_com.shape[0])
                if iter_index % self.cfg.SHOW_STEP == 0:
                    pbar_str = "Approach: {}| | Epoch: {} || Batch:{:>3d}/{}|| lr: {} " \
                               "|| Batch_Loss:{:>5.3f}".format("Model Train", epoch, iter_index,
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
        return scale_factor

    def validate_with_NCM(self, task, test_model=None, is_test=False):
        acc = []
        self.model.eval()
        self.classifier.eval()
        for task_id in range(task):  # 这里的task 从0 开始
            if self.dataset_handler.val_datasets and (not is_test):
                predict_result = self.validate_with_NCM_per_task(self.model, self.dataset_handler.val_datasets[task_id],
                                                                 task)
            else:
                predict_result = self.validate_with_NCM_per_task(self.model,
                                                                 self.dataset_handler.test_datasets[task_id],
                                                                 task)
            acc.append(predict_result)
            self.logger.info(
                f"task: {task} || per task {task_id}, validate_with_FC acc:{predict_result}"
            )
        acc = np.array(acc)
        return acc
        pass

    def validate_with_NCM_per_task(self, model, val_dataset, task):
        # todo
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.cfg.model.TRAIN.BATCH_SIZE,
                                num_workers=self.cfg.model.TRAIN.NUM_WORKERS, shuffle=False, drop_last=False)
        top1 = AverageMeter()
        correct = 0
        for inputs, labels in val_loader:
            correct_temp = 0
            inputs, labels = inputs.to("cuda"), labels.to("cuda")
            with torch.set_grad_enabled(False):
                features = model(x=inputs, is_nograd=True, feature_flag=True)
                logits = self.classifier(features, self.centroids)
            _, balance_fc_y_hat = torch.max(logits, 1)
            correct_temp += balance_fc_y_hat.eq(labels.data).cpu().sum()
            correct += correct_temp
            top1.update((correct_temp / inputs.size(0)).item(), inputs.size(0))
        return top1.avg
        pass

    def validate_with_NCM_taskIL(self, task, is_test=False):
        acc = []
        mode = self.model.training
        self.model.eval()
        for task_id in range(task):  # 这里的task 从0 开始
            if self.dataset_handler.val_datasets and (not is_test):
                predict_result = self.validate_with_NCM_per_task_taskIL(self.dataset_handler.val_datasets[task_id],
                                                                        task_id)
            else:
                predict_result = self.validate_with_NCM_per_task_taskIL(self.dataset_handler.test_datasets[task_id],
                                                                        task_id)
            acc.append(predict_result)
            self.logger.info(
                f"task: {task} || per task {task_id}, validate_with_FC acc:{predict_result}"
            )
        acc = np.array(acc)
        self.model.train(mode=mode)
        return acc
        pass

    def validate_with_NCM_per_task_taskIL(self, val_dataset, task_id):
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
            with torch.set_grad_enabled(False):
                features = self.model(x=inputs, is_nograd=True, feature_flag=True)
                out = self.classifier(features, self.centroids)
            _, balance_fc_y_hat = torch.max(out[:, allowed_classes], 1)
            balance_fc_y_hat += task_id * self.dataset_handler.classes_per_task
            correct_temp += balance_fc_y_hat.eq(labels.data).cpu().sum()
            correct += correct_temp
            top1.update((correct_temp / inputs.size(0)).item(), inputs.size(0))
        return top1.avg
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
            }, os.path.join(model_dir, "{}_best_model.pth".format(task_id))
            )
            torch.save({
                'state_dict': self.latest_model.state_dict(),
                'acc_result': acc,
                'latest_epoch': epoch,
                'task_id': task_id,
                'split_selected_data': split_selected_data
            }, os.path.join(model_dir, "{}_latest_model.pth".format(task_id))
            )

            torch.save({
                'state_dict': self.new_model.state_dict(),
                'task_id': task_id,
            }, os.path.join(model_dir, "{}_latest_new_model.pth".format(task_id))
            )
        pass