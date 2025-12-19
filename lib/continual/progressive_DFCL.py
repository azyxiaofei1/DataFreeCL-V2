import torch
import torch.nn as nn
import copy

from torch.utils.data import DataLoader
from torch.nn import functional as Func
import numpy as np

from .deepInversionGenBN import DeepInversionGenBN
from lib.model import (
    BalancedSoftmax,
    CrossEntropy,
    combine_loss,
    compute_BSCEKD_loss,
    compute_distill_loss,
    curvature_balanced_loss,
    log_tau_epoch_weight,
)
from lib.model.distill_relation import RKDAngleLoss
from lib.utils import AverageMeter, get_classifier_weight_bias
from lib.utils.feature_pool import FeaturePool

'''Code for progressive_DFCL'''


class progressive_DFCL(DeepInversionGenBN):

    def __init__(self, cfg, dataset_handler, logger):
        super(progressive_DFCL, self).__init__(cfg, dataset_handler, logger)
        self.new_model = None
        self.criterion_fn = nn.CrossEntropyLoss(reduction="none")
        self.rkd = RKDAngleLoss(
            self.cfg.extractor.output_feature_dim,
            proj_dim=2 * self.cfg.extractor.output_feature_dim,
        )
        self.rkd = self.transfer_to_cuda(self.rkd)

        # >>> 新增：曲率正则的权重，从 cfg 里读（你可以在 YAML 里加一个 CR_lambda）
        self.cr_lambda = getattr(self.cfg.model, "CR_LAMBDA", 0.0)
        self.cr_mode = getattr(self.cfg.model, "CR_MODE", "proxy")
        self.cr_tau = getattr(self.cfg.model, "CR_TAU", 2.0)
        self.cr_warmup_epoch = getattr(self.cfg.model, "CR_WARMUP_EPOCH", 0)
        self.cr_pool_size = getattr(self.cfg.model, "CR_POOL_SIZE", 0)
        self.cr_interval = getattr(self.cfg.model, "CR_INTERVAL", 10)
        self.cr_pool_sample_max = getattr(self.cfg.model, "CR_POOL_SAMPLE_MAX", 2048)
        self.cr_k_neighbors = getattr(self.cfg.model, "CR_K_NEIGHBORS", 12)
        self.cr_num_samples_per_class = getattr(self.cfg.model, "CR_NUM_SAMPLES_PER_CLASS", 32)
        # <<<

    def curvature_regularization(self, features, labels, num_samples_per_class=32, k_neighbors=16):
        """
        features: [N, D] 当前 batch 的特征 all_features
        labels:   [N]    当前 batch 的标签 all_y
        简化版思路：
          1. 对每个类采样若干点（num_samples_per_class）。
          2. 对这些点找同类的邻居（k_neighbors）。
          3. 在局部邻域上拟合一个二次函数 f(o_j; θ) ~ z_j（或者直接用邻域点对局部“弯曲度”的 proxy）。
          4. 用拟合残差或 Hessian 的 Frobenius 范数作为“曲率度量”，在所有类上求平均。
        """
        device = features.device
        labels = labels.view(-1)

        # 1. 按类分组
        unique_classes = labels.unique()
        all_class_curv = []
        for c in unique_classes:
            idx = (labels == c).nonzero(as_tuple=False).view(-1)
            if idx.numel() < k_neighbors + 1:
                # 样本太少，跳过
                continue

            # 随机采样部分点，避免算太慢
            perm = torch.randperm(idx.numel(), device=device)
            idx = idx[perm[:min(num_samples_per_class, idx.numel())]]

            feats_c = features[idx]   # [Nc, D]
            # 2. 在类内计算 pairwise 距离
            with torch.no_grad():
                # [Nc, Nc]
                dist2 = torch.cdist(feats_c, feats_c, p=2.0)

            # 3. 对每个点取最近的 k_neighbors 个邻居，拟合“局部平面/二次曲面”的弯曲度 proxy
            curv_list = []
            for i in range(feats_c.size(0)):
                # 最近邻（排除自身）
                d_i = dist2[i]              # [Nc]
                nn_idx = torch.topk(d_i, k_neighbors + 1, largest=False).indices[1:]  # 去掉自己
                nbr = feats_c[nn_idx]       # [k, D]
                center = feats_c[i:i+1]     # [1, D]



                # 简化版曲率度量：
                #   用邻居点到其局部 PCA 平面的残差作为“弯曲度”
                #   1) 对 diff 做 SVD，取前 2~3 个主方向作为切空间；
                #   2) 其余方向上的分量（法向残差）的平方和即为 curvature proxy。
                # 实现一个很粗糙但能跑的版本：
                # [D, k]
                # 局部坐标系 (去中心)
                diff = nbr - center  # [k, D]
                diff_T = diff.t()           # [D, k]
                # SVD: diff_T = U S V^T
                # U: [D, D] 主方向
                U, S, Vh = torch.linalg.svd(diff_T, full_matrices=False)

                # 取前 d_tan 个方向作为切空间，比如 2
                d_tan = min(2, U.size(1))
                U_tan = U[:, :d_tan]        # [D, d_tan]
                U_norm = U[:, d_tan:]       # [D, D-d_tan]

                # 投影到法向子空间
                # diff_norm = U_norm U_norm^T diff^T
                diff_norm = U_norm @ (U_norm.t() @ diff_T)  # [D, k]
                # 残差平方（法向方向能量）
                curv_i = (diff_norm ** 2).sum() / (diff_norm.numel() + 1e-8)
                curv_list.append(curv_i)

            if len(curv_list) > 0:
                class_curv = torch.stack(curv_list).mean()
                all_class_curv.append(class_curv)

        if len(all_class_curv) == 0:
            return torch.tensor(0.0, device=device)

        # 所有类的平均“曲率”
        curv_loss = torch.stack(all_class_curv).mean()
        return curv_loss

    def curvature_proxy_per_class(self, features, labels, num_samples_per_class=32, k_neighbors=16):
        """
        Return per-class curvature proxy Gi, shape [K]. K is number of valid classes.
        features: [N, D], labels: [N]
        """
        device = features.device
        labels = labels.view(-1)

        unique_classes = labels.unique()
        all_class_curv = []

        for c in unique_classes:
            idx = (labels == c).nonzero(as_tuple=False).view(-1)
            Nc_total = idx.numel()
            if Nc_total < (k_neighbors + 1):
                continue

            perm = torch.randperm(Nc_total, device=device)
            idx = idx[perm[:min(num_samples_per_class, Nc_total)]]
            feats_c = features[idx]  # [Nc, D]
            Nc = feats_c.size(0)
            if Nc < (k_neighbors + 1):
                continue

            with torch.no_grad():
                dist2 = torch.cdist(feats_c, feats_c, p=2.0)  # [Nc, Nc]

            curv_list = []
            for i in range(Nc):
                d_i = dist2[i]
                k_use = min(max(k_neighbors + 1, 4), Nc)
                nn_idx = torch.topk(d_i, k_use, largest=False).indices[1:]  # 去掉自己
                if nn_idx.numel() < 2:
                    continue

                nbr = feats_c[nn_idx]     # [k, D]
                center = feats_c[i:i+1]   # [1, D]

                diff = nbr - center       # [k, D]
                diff_T = diff.t()         # [D, k]
                U, S, Vh = torch.linalg.svd(diff_T, full_matrices=False)

                d_tan = min(2, max(1, U.size(1) - 1))
                U_norm = U[:, d_tan:]     # [D, D-d_tan]
                if U_norm.numel() == 0:
                    curv_i = torch.zeros((), device=device, dtype=features.dtype)
                else:
                    diff_norm = U_norm @ (U_norm.t() @ diff_T)  # [D, k]
                    curv_i = (diff_norm ** 2).sum() / (diff_norm.numel() + 1e-8)
                curv_list.append(curv_i)

            if len(curv_list) > 0:
                all_class_curv.append(torch.stack(curv_list).mean())

        if len(all_class_curv) == 0:
            return torch.empty(0, device=device)

        return torch.stack(all_class_curv)

    @staticmethod
    def curvature_proxy_per_class_grad(features, labels, num_samples_per_class=32, k_neighbors=16):
        device = features.device
        labels = labels.view(-1)
        unique_classes = labels.unique()
        all_class_curv = []
        for c in unique_classes:
            idx = (labels == c).nonzero(as_tuple=False).view(-1)
            Nc_total = idx.numel()
            if Nc_total < (k_neighbors + 1):
                continue
            perm = torch.randperm(Nc_total, device=device)
            idx = idx[perm[:min(num_samples_per_class, Nc_total)]]
            feats_c = features[idx]
            Nc = feats_c.size(0)
            if Nc < (k_neighbors + 1):
                continue

            with torch.no_grad():
                dist2 = torch.cdist(feats_c, feats_c, p=2.0)

            curv_list = []
            for i in range(Nc):
                d_i = dist2[i]
                k_use = min(max(k_neighbors + 1, 4), Nc)
                nn_idx = torch.topk(d_i, k_use, largest=False).indices[1:]
                if nn_idx.numel() < 2:
                    continue

                nbr = feats_c[nn_idx]
                center = feats_c[i:i+1]
                diff = nbr - center
                diff_T = diff.t()

                with torch.no_grad():
                    U, S, Vh = torch.linalg.svd(diff_T.detach(), full_matrices=False)
                    d_tan = min(2, max(1, U.size(1) - 1))
                    U_norm = U[:, d_tan:]

                if U_norm.numel() == 0:
                    curv_i = torch.zeros((), device=device, dtype=features.dtype)
                else:
                    diff_norm = U_norm @ (U_norm.t() @ diff_T)
                    curv_i = (diff_norm ** 2).mean()

                curv_list.append(curv_i)

            if len(curv_list) > 0:
                all_class_curv.append(torch.stack(curv_list).mean())

        if len(all_class_curv) == 0:
            return torch.empty(0, device=device, dtype=features.dtype)
        return torch.stack(all_class_curv)


    def ft_criterion(self, logits, targets, data_weights):
        loss_supervised = (self.criterion_fn(logits, targets.long()) * data_weights).mean()
        return loss_supervised

    def bsce_criterion(self, logits, targets, sample_num_per_class):
        bsce_criterion = BalancedSoftmax(sample_num_per_class).to(self.device)
        loss_supervised = bsce_criterion(logits, targets)
        return loss_supervised

    def learn_new_task(self, train_dataset, active_classes_num, task):
        dp_classes_num = active_classes_num - self.dataset_handler.classes_per_task
        self.pre_steps(dp_classes_num)  # 训练生成器
        self.train_new_model(train_dataset, active_classes_num, task)
        self.train_model(train_dataset, active_classes_num, task)

        self.fine_tune(train_dataset, active_classes_num, task)

    def train_new_model(self, train_dataset, active_classes_num, task):
        dp_classes_num = active_classes_num - self.dataset_handler.classes_per_task
        if self.new_model is None:
            self.new_model = copy.deepcopy(self.model)
            # self.new_model = self.transfer_to_cuda(self.new_model)
            print("new_model:", self.new_model)
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
                                  persistent_workers=True,pin_memory=True,) #加上 pin_memory=True
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
                    total_loss = self.LwF_loss(y_com=y_com, logits_com=logits_com, pre_logits_com=pre_logits_com,
                                               cls_criterion=cls_criterion, dp_classes_num=dp_classes_num, rnt=rnt)
                    # total_loss = self.LwF_loss(y_com=y_com, logits_com=logits_com, pre_logits_com=pre_logits_com,
                    #                            cls_criterion=bsce_criterion, dp_classes_num=dp_classes_num, rnt=rnt)
                    pass
                elif "cls" in self.cfg.new_model.loss_type:
                    total_loss = self.cls_loss(y_com=y_com, logits_com=logits_com, cls_criterion=cls_criterion)
                else:
                    raise ValueError("self.cfg.model.loss_type is illegal.")

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                # measure accuracy and record loss
                all_loss.update(total_loss.data.item(), y_com.shape[0])
                if iter_index % self.cfg.SHOW_STEP == 0:
                    pbar_str = "Approach: {}| | Epoch: {} || Batch:{:>3d}/{}|| lr: {} " \
                               "|| Batch_Loss:{:>5.3f}".format("progressive KD", epoch, iter_index,
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
                    self.best_model = copy.deepcopy(self.new_model)
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

    def cls_loss(self, y_com, logits_com, cls_criterion):
        loss_class = cls_criterion(logits_com, y_com)
        total_loss = loss_class
        return total_loss


    def train_model(self, train_dataset, active_classes_num, task):
        if hasattr(self.new_model, "module"):
            global_classifier_weights, global_classifier_bias = \
                get_classifier_weight_bias(self.new_model.module.linear_classifier)
        else:
            global_classifier_weights, global_classifier_bias = \
                get_classifier_weight_bias(self.new_model.linear_classifier)
        global_classifier_paras = torch.cat((global_classifier_weights, global_classifier_bias.unsqueeze(1)), 1)
        global_classifier_weights_norm = torch.norm(global_classifier_weights.detach(), 2, 1)
        global_classifier_weights_norm = global_classifier_weights_norm[0: active_classes_num]

        global_classifier_paras_norm = torch.norm(global_classifier_paras.detach(), 2, 1)
        global_classifier_paras_norm = global_classifier_paras_norm[0: active_classes_num]

        dpt_classes_num = active_classes_num - self.dataset_handler.classes_per_task
        self.logger.info(f"global_classifier_weights_norm: {global_classifier_weights_norm}")
        self.logger.info(f"global_classifier_weights_norm mean of old classes: "
                         f"{torch.mean(global_classifier_weights_norm[0:dpt_classes_num])}")
        self.logger.info(f"global_classifier_weights_norm mean of new classes: "
                         f"{torch.mean(global_classifier_weights_norm[dpt_classes_num: active_classes_num])}")

        self.logger.info(f"global_classifier_paras_norm: {global_classifier_paras_norm}")
        self.logger.info(f"global_classifier_paras_norm mean of old classes: "
                         f"{torch.mean(global_classifier_paras_norm[0:dpt_classes_num])}")
        self.logger.info(f"global_classifier_paras_norm mean of new classes: "
                         f"{torch.mean(global_classifier_paras_norm[dpt_classes_num: active_classes_num])}")

        class_weight_from_feature_norm = global_classifier_weights_norm.sum() / global_classifier_weights_norm
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

        bsce_criterion = BalancedSoftmax(sample_per_class=sample_num_per_class).to(self.device)
        feat_bsce_criterion = BalancedSoftmax(sample_per_class=global_classifier_weights_norm).to(self.device)
        feat_bias_bsce_criterion = BalancedSoftmax(sample_per_class=global_classifier_paras_norm).to(self.device)

        pool = None
        if self.cr_mode == "balanced":
            pool_size = int(self.cr_pool_size) if int(self.cr_pool_size) > 0 else 10 * self.cfg.model.TRAIN.BATCH_SIZE
            pool = FeaturePool(max_samples=pool_size, store_device="cpu")

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
                    new_model_outs, new_model_features = self.new_model(all_x)
                new_model_outs = new_model_outs[:, :active_classes_num]

                if "RTKD" == self.cfg.model.KD_type:
                    loss_KD = torch.zeros(task).to(self.device)
                    all_ouput_for_dis = all_outputs
                    all_pre_model_output_for_dis = new_model_outs
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
                elif "LAKD-class" == self.cfg.model.KD_type:
                    all_ouput_for_dis = all_outputs
                    all_pre_model_output_for_dis = new_model_outs
                    loss_kd = compute_BSCEKD_loss(all_ouput_for_dis, all_pre_model_output_for_dis,
                                                  sample_num_per_class, temp=self.cfg.model.T,
                                                  tau=self.cfg.model.tau) * self.cfg.model.kd_lambda

                    output_for_pre_distill = pre_model_output_for_distill
                    loss_kd += compute_distill_loss(all_outputs[:, 0:dpt_classes_num], output_for_pre_distill,
                                                    temp=self.cfg.model.T) * self.cfg.model.kd_lambda
                    pass
                elif "LAKD-feat" == self.cfg.model.KD_type:
                    all_ouput_for_dis = all_outputs
                    all_pre_model_output_for_dis = new_model_outs
                    loss_kd = compute_BSCEKD_loss(all_ouput_for_dis, all_pre_model_output_for_dis,
                                                  global_classifier_weights_norm, temp=self.cfg.model.T,
                                                  tau=self.cfg.model.tau) * self.cfg.model.kd_lambda
                    output_for_pre_distill = pre_model_output_for_distill
                    loss_kd += compute_distill_loss(all_outputs[:, 0:dpt_classes_num], output_for_pre_distill,
                                                    temp=self.cfg.model.T) * self.cfg.model.kd_lambda
                elif "LAKD-feat-bias" == self.cfg.model.KD_type:
                    all_ouput_for_dis = all_outputs
                    all_pre_model_output_for_dis = new_model_outs
                    loss_kd = compute_BSCEKD_loss(all_ouput_for_dis, all_pre_model_output_for_dis,
                                                  global_classifier_paras_norm, temp=self.cfg.model.T,
                                                  tau=self.cfg.model.tau) * self.cfg.model.kd_lambda
                    output_for_pre_distill = pre_model_output_for_distill
                    loss_kd += compute_distill_loss(all_outputs[:, 0:dpt_classes_num], output_for_pre_distill,
                                                    temp=self.cfg.model.T) * self.cfg.model.kd_lambda
                elif "vanillaKD" == self.cfg.model.KD_type:
                    loss_kd = compute_distill_loss(all_outputs, new_model_outs,
                                                   temp=self.cfg.model.T) * self.cfg.model.kd_lambda
                elif "null" == self.cfg.model.KD_type:
                    loss_kd = 0.
                else:
                    raise ValueError("self.cfg.model.KD_type is illegal.")

                if self.cfg.model.use_featureKD:
                    self.new_model.zero_grad()
                    logits_KD = self.new_model(x=all_features, train_cls_use_features=True)
                    logits_KD = logits_KD[:, :active_classes_num]
                    logits_KD_past = new_model_outs

                    # loss_kd += self.cfg.model.fkd_lambda * (
                    #     self.kd_criterion(logits_KD, logits_KD_past).sum(dim=1) * data_weights).mean() / (
                    #                logits_KD.size(1))

                    loss_kd += self.cfg.model.fkd_lambda * (
                        self.kd_criterion(logits_KD, logits_KD_past).sum(dim=1)).mean() / (
                                   logits_KD.size(1))

                    # logits_KD_2 = self.previous_teacher.generate_score_by_features(features=all_features,
                    #                                                              active_classes_num=active_classes_num,
                    #                                                              is_nograd=False)
                    # logits_KD_2 = logits_KD_2[:, :dpt_classes_num]
                    # logits_KD_past_2 = pre_model_output_for_distill
                    # loss_kd += self.cfg.model.fkd_lambda * (
                    #     self.kd_criterion(logits_KD_2, logits_KD_past_2).sum(dim=1)).mean() / (
                    #                logits_KD_2.size(1))

                if "BSCE-class" == self.cfg.model.cls_type:
                    loss_cls = bsce_criterion(input=all_outputs, label=all_y)
                elif "BSCE-feat" == self.cfg.model.cls_type:
                    loss_cls = feat_bsce_criterion(input=all_outputs, label=all_y)
                    pass
                elif "BSCE-feat-bias" == self.cfg.model.cls_type:
                    loss_cls = feat_bias_bsce_criterion(input=all_outputs, label=all_y)
                    pass
                elif "LwF" == self.cfg.model.cls_type:
                    loss_cls = cls_criterion(all_outputs[:self.cfg.model.TRAIN.BATCH_SIZE,
                                             -self.dataset_handler.classes_per_task:],
                                             all_y[:self.cfg.model.TRAIN.BATCH_SIZE] %
                                             self.dataset_handler.classes_per_task) * (1 - rnt)

                    output_for_pre_distill = pre_model_output_for_distill
                    # output_for_pre_distill = new_model_outs[:, 0:dpt_classes_num]
                    loss_cls += compute_distill_loss(all_outputs[:, 0:dpt_classes_num], output_for_pre_distill,
                                                     temp=self.cfg.model.T) * rnt

                    pass
                elif "null" == self.cfg.model.cls_type:
                    loss_cls = 0.
                else:
                    raise ValueError(f"self.cfg.model.cls_type is illegal.")

                # ……前面 loss_kd / loss_cls 都算好之后……

                # === 曲率正则 CR 部分（静态版/平衡版） ===
                L_original = loss_cls + loss_kd
                curv_loss = torch.tensor(0.0, device=self.device)
                gi_stats = None
                weight_val = 0.0

                if self.cr_mode == "proxy":
                    if getattr(self, "cr_lambda", 0.0) > 0:
                        curv_loss = self.curvature_regularization(all_features, all_y)
                    total_loss = L_original + getattr(self, "cr_lambda", 0.0) * curv_loss

                elif self.cr_mode == "balanced" and pool is not None:
                    pool.enqueue(all_features, all_y)
                    if epoch < self.cr_warmup_epoch:
                        total_loss = L_original
                    else:
                        if (iter_index % self.cr_interval) != 0:
                            total_loss = L_original
                        else:
                            feats_pool, labs_pool = pool.get_all(device=torch.device("cpu"))
                            if feats_pool.size(0) > int(self.cr_pool_sample_max) and int(self.cr_pool_sample_max) > 0:
                                M = int(self.cr_pool_sample_max)
                                perm = torch.randperm(
                                    feats_pool.size(0), device=feats_pool.device
                                )[:M]
                                feats_pool = feats_pool[perm]
                                labs_pool = labs_pool[perm]

                            Gi_cpu = self.curvature_proxy_per_class(
                                feats_pool, labs_pool,
                                num_samples_per_class=self.cr_num_samples_per_class,
                                k_neighbors=self.cr_k_neighbors
                            )
                            max_inv_const = None
                            if Gi_cpu.numel() > 0:
                                inv_cpu = 1.0 / (Gi_cpu + 1e-8)
                                max_inv_const = inv_cpu.max().to(self.device)

                            Gi_grad = progressive_DFCL.curvature_proxy_per_class_grad(
                                all_features, all_y,
                                num_samples_per_class=self.cr_num_samples_per_class,
                                k_neighbors=self.cr_k_neighbors
                            )
                            if Gi_grad.numel() > 0:
                                if max_inv_const is None:
                                    max_inv_const = (1.0 / (Gi_grad.detach() + 1e-8)).max()
                                curv_loss = curvature_balanced_loss(Gi_grad, max_inv_const=max_inv_const)
                                total_loss = combine_loss(L_original, curv_loss, epoch, self.cr_tau)
                                stats_src = Gi_cpu if Gi_cpu.numel() > 0 else Gi_grad.detach()
                                gi_stats = (stats_src.min().item(), stats_src.mean().item(), stats_src.max().item())
                                weight_val = log_tau_epoch_weight(epoch, self.cr_tau).item()
                            else:
                                total_loss = L_original
                else:
                    total_loss = L_original

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # ✅ 每个 batch 更新一次
                all_loss.update(total_loss.item(), all_y.size(0))

                if iter_index % self.cfg.SHOW_STEP == 0:
                    pbar_str = (
                        "Approach: {}| | Epoch: {} || Batch:{:>3d}/{}|| lr: {} || Batch_Loss:{:>5.3f}"
                    ).format(
                        self.cfg.trainer_name, epoch, iter_index, iter_num,
                        optimizer.param_groups[0]['lr'],
                        all_loss.val
                    )

                    if self.cr_mode == "proxy":
                        pbar_str += " || CR:{:>5.3f}".format(
                            curv_loss.item() if isinstance(curv_loss, torch.Tensor) else float(curv_loss)
                        )
                    elif self.cr_mode == "balanced":
                        pbar_str += " || L_orig:{:>5.3f} || L_curv:{:>5.3f} || weight:{:>5.3f} || pool:{:>4d}".format(
                            L_original.item() if isinstance(L_original, torch.Tensor) else float(L_original),
                            curv_loss.item() if isinstance(curv_loss, torch.Tensor) else float(curv_loss),
                            float(weight_val),
                            len(pool) if pool is not None else 0
                        )
                        if gi_stats is not None:
                            pbar_str += " || Gi[min/mean/max]:{:>5.3f}/{:>5.3f}/{:>5.3f}".format(
                                gi_stats[0], gi_stats[1], gi_stats[2]
                            )
                        pbar_str += " || samp:{:d}".format(int(self.cr_pool_sample_max))
                        pbar_str += " || intv:{:d}".format(self.cr_interval)
                        pbar_str += " || knn:{:d} || nsamp:{:d}".format(
                            int(self.cr_k_neighbors), int(self.cr_num_samples_per_class)
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
