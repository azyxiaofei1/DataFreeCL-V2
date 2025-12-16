import copy

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from lib.continual.dfcl_base_learner import base_learner
from lib.dataset import SubDataset, TransformedDataset_for_exemplars
from lib.utils import AverageMeter


class FeCAM_learner(base_learner):
    def __init__(self, cfg, dataset_handler, logger):
        super(FeCAM_learner, self).__init__(cfg, dataset_handler, logger)
        self.mean_feature_per_class = None
        self.features_per_current_class = None
        self._cov_mat = []
        self._norm_cov_mat = None
        self._cov_mat_shrink, self._norm_cov_mat = [], []

    # def _maha_dist(self, vectors, class_means, active_classes_num):
    #     vectors = torch.tensor(vectors).cuda()
    #     vectors = self._tukeys_transform(vectors)
    #     maha_dist = []
    #     for class_index in range(active_classes_num):
    #         if self.cfg.model.norm_cov:
    #             dist = self._mahalanobis(vectors, class_means[class_index], self._norm_cov_mat[class_index])
    #         elif self.cfg.model.shrink:
    #             dist = self._mahalanobis(vectors, class_means[class_index],
    #                                      self._cov_mat_shrink[class_index])
    #         else:
    #             dist = self._mahalanobis(vectors, class_means[class_index], self._cov_mat[class_index])
    #         maha_dist.append(dist)
    #     maha_dist = np.array(maha_dist)  # [nb_classes, N]
    #     return torch.from_numpy(maha_dist.T)

    def _maha_dist(self, vectors, class_means, active_classes_num, allowed_classes=None):
        vectors = torch.tensor(vectors).cuda()
        vectors = self._tukeys_transform(vectors)
        maha_dist = []
        if allowed_classes is not None:
            for class_item in allowed_classes:
                if self.cfg.model.norm_cov:
                    dist = self._mahalanobis(vectors, class_means[class_item], self._norm_cov_mat[class_item])
                elif self.cfg.model.shrink:
                    dist = self._mahalanobis(vectors, class_means[class_item],
                                             self._cov_mat_shrink[class_item])
                else:
                    dist = self._mahalanobis(vectors, class_means[class_item], self._cov_mat[class_item])
                maha_dist.append(dist)
        else:
            for class_index in range(active_classes_num):
                if self.cfg.model.norm_cov:
                    dist = self._mahalanobis(vectors, class_means[class_index], self._norm_cov_mat[class_index])
                elif self.cfg.model.shrink:
                    dist = self._mahalanobis(vectors, class_means[class_index],
                                             self._cov_mat_shrink[class_index])
                else:
                    dist = self._mahalanobis(vectors, class_means[class_index], self._cov_mat[class_index])
                maha_dist.append(dist)
        maha_dist = np.array(maha_dist)  # [nb_classes, N]
        return torch.from_numpy(maha_dist.T)

    def _mahalanobis(self, vectors, class_means, cov=None):
        class_means = self._tukeys_transform(class_means)
        class_means = class_means.cuda()
        x_minus_mu = F.normalize(vectors, p=2, dim=-1) - F.normalize(class_means, p=2, dim=-1)
        if cov is None:
            cov = torch.eye(self.cfg.extractor.output_feature_dim)  # identity covariance matrix for euclidean distance
        inv_covmat = torch.linalg.pinv(cov).float().cuda()
        left_term = torch.matmul(x_minus_mu, inv_covmat)
        mahal = torch.matmul(left_term, x_minus_mu.T)
        # print(f"mahal shape: {mahal.shape}")
        return torch.diagonal(mahal, 0).cpu().numpy()

    def diagonalization(self, cov):
        diag = cov.clone()
        cov_ = cov.clone()
        cov_.fill_diagonal_(0.0)
        diag = diag - cov_
        return diag

    def shrink_cov(self, cov):
        diag_mean = torch.mean(torch.diagonal(cov))
        off_diag = cov.clone()
        off_diag.fill_diagonal_(0.0)
        mask = off_diag != 0.0
        off_diag_mean = (off_diag * mask).sum() / mask.sum()
        iden = torch.eye(cov.shape[0])
        alpha1 = self.cfg.model.alpha1
        alpha2 = self.cfg.model.alpha2
        cov_ = cov + (alpha1 * diag_mean * iden) + (alpha2 * off_diag_mean * (1 - iden))
        return cov_

    def normalize_cov(self):
        if self.cfg.model.shrink:
            cov_mat = self._cov_mat_shrink
        else:
            cov_mat = self._cov_mat
        norm_cov_mat = []
        for cov in cov_mat:
            sd = torch.sqrt(torch.diagonal(cov))  # standard deviations of the variables
            cov = cov / (torch.matmul(sd.unsqueeze(1), sd.unsqueeze(0)))
            norm_cov_mat.append(cov)

        # print(len(norm_cov_mat))
        return norm_cov_mat

    def normalize_cov2(self, cov):
        diag = torch.diagonal(cov)
        norm = torch.linalg.norm(diag)
        cov = cov / norm
        return cov

    def _tukeys_transform(self, x):
        beta = self.cfg.model.beta
        x = torch.tensor(x)
        if beta == 0:
            return torch.log(x)
        else:
            return torch.pow(x, beta)

    def store_mean_feature_per_class(self, task):
        train_dataset = TransformedDataset_for_exemplars(self.dataset_handler.original_imgs_train_datasets[task - 1],
                                                         transform=self.dataset_handler.val_test_dataset_transform)

        new_classes = list(range(self.dataset_handler.classes_per_task * (task - 1),
                                 self.dataset_handler.classes_per_task * task))
        self.features_per_current_class = []
        for class_id in new_classes:
            # create new dataset containing only all examples of this class
            self.logger.info(f"construct_exemplar_set class_id: {class_id}")
            class_dataset = SubDataset(original_dataset=train_dataset,
                                       sub_labels=[class_id])
            # based on this dataset, construct new exemplar-set for this class
            class_mean_feature, features = self.compute_mean_feature(class_dataset, self.model)
            self.features_per_current_class.append(features)
            if self.mean_feature_per_class is None:
                self.mean_feature_per_class = class_mean_feature
            else:
                self.mean_feature_per_class = torch.cat([self.mean_feature_per_class, class_mean_feature], dim=0)
        assert len(self.features_per_current_class) == self.dataset_handler.classes_per_task

    def store_mean_feature_per_class_for_local_dataset(self, train_dataset_transform, task):
        train_dataset_temp = self.dataset_handler.original_imgs_train_datasets_per_class[task - 1]
        new_classes = list(range(self.dataset_handler.classes_per_task * (task - 1),
                                 self.dataset_handler.classes_per_task * task))
        self.features_per_current_class = []
        for class_id in range(len(new_classes)):
            # create new dataset containing only all examples of this class
            self.logger.info(
                f"construct_exemplar_set class_id: {class_id + self.dataset_handler.classes_per_task * (task - 1)}")
            class_dataset = TransformedDataset_for_exemplars(train_dataset_temp[class_id],
                                                             store_original_imgs=False,
                                                             transform=train_dataset_transform)

            # based on this dataset, construct new exemplar-set for this class
            class_mean_feature, features = self.compute_mean_feature(class_dataset, self.model)
            self.features_per_current_class.append(features)
            if self.mean_feature_per_class is None:
                self.mean_feature_per_class = class_mean_feature
            else:
                self.mean_feature_per_class = torch.cat([self.mean_feature_per_class, class_mean_feature], dim=0)
        assert len(self.features_per_current_class) == self.dataset_handler.classes_per_task

    def compute_mean_feature(self, train_dataset, model):
        first_entry = True
        dataloader = DataLoader(train_dataset, batch_size=128, shuffle=False, num_workers=1,
                                pin_memory=False, drop_last=False, persistent_workers=True)
        for (image_batch, _, _) in dataloader:
            image_batch = image_batch.to(self.device)
            feature_batch = self.get_extracted_features(model, image_batch).cpu()
            if first_entry:
                features = feature_batch
                first_entry = False
            else:
                features = torch.cat([features, feature_batch], dim=0)
        class_mean_feature = torch.mean(features, dim=0, keepdim=True)
        return class_mean_feature, features.numpy()
        pass

    @staticmethod
    def get_extracted_features(model, image_batch):
        features = model(x=image_batch, is_nograd=True, feature_flag=True)
        return features

    def compute_new_cov(self):
        for features_per_class in self.features_per_current_class:
            vectors = self._tukeys_transform(features_per_class)
            cov = torch.tensor(np.cov(vectors.T))
            if self.cfg.model.shrink:
                cov = self.shrink_cov(cov)
            self._cov_mat.append(cov)

    def learn_new_task(self, train_dataset, active_classes_num, task):
        self.pre_tasks_model = copy.deepcopy(self.model).eval()
        dp_class_num = self.dataset_handler.classes_per_task * (task - 1)
        if not self.cfg.use_base_half and task == 2:
            self.store_mean_feature_per_class(1)
            self.compute_new_cov()
        elif self.cfg.use_base_half and task == int(self.dataset_handler.all_tasks / 2) + 1:
            for task_id in range(1, task):
                self.store_mean_feature_per_class(task_id)
                self.compute_new_cov()
        self.store_mean_feature_per_class(task)
        self.compute_new_cov()
        assert len(self._cov_mat) == active_classes_num
        self._cov_mat_shrink, self._norm_cov_mat = [], []
        if self.cfg.model.shrink:
            for cov in self._cov_mat:
                self._cov_mat_shrink.append(self.shrink_cov(cov))
            if self.cfg.model.norm_cov:
                self._norm_cov_mat = self.normalize_cov()
        else:
            raise ValueError(f"shrink must be True.")

    def learn_new_task_for_local_dataset(self, train_dataset, active_classes_num, task, train_dataset_transform):
        self.pre_tasks_model = copy.deepcopy(self.model).eval()
        dp_class_num = self.dataset_handler.classes_per_task * (task - 1)
        if not self.cfg.use_base_half and task == 2:
            self.store_mean_feature_per_class_for_local_dataset(train_dataset_transform, 1)
            self.compute_new_cov()
        elif self.cfg.use_base_half and task == int(self.dataset_handler.all_tasks / 2) + 1:
            for task_id in range(1, task):
                self.store_mean_feature_per_class_for_local_dataset(train_dataset_transform, task_id)
                self.compute_new_cov()
        self.store_mean_feature_per_class_for_local_dataset(train_dataset_transform, task)
        self.compute_new_cov()
        assert len(self._cov_mat) == active_classes_num
        self._cov_mat_shrink, self._norm_cov_mat = [], []
        if self.cfg.model.shrink:
            for cov in self._cov_mat:
                self._cov_mat_shrink.append(self.shrink_cov(cov))
            if self.cfg.model.norm_cov:
                self._norm_cov_mat = self.normalize_cov()
        else:
            raise ValueError(f"shrink must be True.")

    def validate_with_maha(self, task, active_classes_num, is_test=False):
        acc = []
        assert active_classes_num == self.mean_feature_per_class.shape[0]
        for task_id in range(task):  # 这里的task 从0 开始
            if self.dataset_handler.val_datasets and (not is_test):
                predict_result = self.validate_with_maha_per_task(self.dataset_handler.val_datasets[task_id],
                                                                  active_classes_num)
            else:
                predict_result = self.validate_with_maha_per_task(self.dataset_handler.test_datasets[task_id],
                                                                  active_classes_num)
            acc.append(predict_result)
            self.logger.info(
                f"task: {task} || per task {task_id}, validate_with_FC acc:{predict_result}"
            )
        acc = np.array(acc)
        return acc
        pass

    def validate_with_maha_per_task(self, val_dataset, active_classes_num):
        # todo
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.cfg.model.eeil_finetune_train.BATCH_SIZE,
                                num_workers=self.cfg.model.eeil_finetune_train.NUM_WORKERS, shuffle=False,
                                drop_last=False)
        features = None
        labels = None
        top1 = AverageMeter()
        correct = 0
        for x, y in val_loader:
            x = x.to("cuda")
            y = y.to("cuda")
            feature_item = self.model(x, is_nograd=True, feature_flag=True)
            feature_item = feature_item.cpu()
            if features is None:
                features = feature_item
                labels = y
            else:
                features = torch.cat([features, feature_item], dim=0)
                labels = torch.cat([labels, y], dim=0)
        features = features.numpy()
        vectors = (features.T / (np.linalg.norm(features.T, axis=0) + self.cfg.model.EPSILON)).T
        dists = self._maha_dist(vectors, self.mean_feature_per_class, active_classes_num)
        out = -dists  # [N, nb_classes], choose the one with the smallest distance
        _, balance_fc_y_hat = torch.max(out, 1)
        assert balance_fc_y_hat.shape[0] == labels.shape[0]
        correct += balance_fc_y_hat.eq(labels.cpu().data).sum()
        top1.update((correct / labels.size(0)).item(), labels.size(0))
        return top1.avg
        pass

    def validate_with_maha_taskIL(self, task, active_classes_num, is_test=False):
        acc = []
        assert active_classes_num == self.mean_feature_per_class.shape[0]
        for task_id in range(task):  # 这里的task 从0 开始
            if self.dataset_handler.val_datasets and (not is_test):
                predict_result = self.validate_with_maha_per_task_taskIL(self.dataset_handler.val_datasets[task_id],
                                                                         task_id, active_classes_num)
            else:
                predict_result = self.validate_with_maha_per_task_taskIL(self.dataset_handler.test_datasets[task_id],
                                                                         task_id, active_classes_num)
            acc.append(predict_result)
            self.logger.info(
                f"task: {task} || per task {task_id}, validate_with_FC acc:{predict_result}"
            )
        acc = np.array(acc)
        return acc
        pass

    def validate_with_maha_per_task_taskIL(self, val_dataset, task_id, active_classes_num):
        # todo
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.cfg.model.eeil_finetune_train.BATCH_SIZE,
                                num_workers=self.cfg.model.eeil_finetune_train.NUM_WORKERS, shuffle=False,
                                drop_last=False)
        features = None
        labels = None
        top1 = AverageMeter()
        correct = 0
        allowed_classes = [i for i in range(task_id * self.dataset_handler.classes_per_task,
                                            (task_id + 1) * self.dataset_handler.classes_per_task)]
        for x, y in val_loader:
            x = x.to("cuda")
            y = y.to("cuda")
            feature_item = self.model(x, is_nograd=True, feature_flag=True)
            feature_item = feature_item.cpu()
            if features is None:
                features = feature_item
                labels = y
            else:
                features = torch.cat([features, feature_item], dim=0)
                labels = torch.cat([labels, y], dim=0)
        features = features.numpy()
        vectors = (features.T / (np.linalg.norm(features.T, axis=0) + self.cfg.model.EPSILON)).T
        dists = self._maha_dist(vectors, self.mean_feature_per_class, active_classes_num,
                                allowed_classes=allowed_classes)
        out = -dists  # [N, nb_classes], choose the one with the smallest distance
        _, balance_fc_y_hat = torch.max(out, 1)
        correct += balance_fc_y_hat.eq(labels.cpu().data % self.dataset_handler.classes_per_task).sum()
        top1.update((correct / labels.size(0)).item(), labels.size(0))
        return top1.avg
        pass
