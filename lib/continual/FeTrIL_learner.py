import copy

import torch
from sklearn.metrics import pairwise_distances
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader
import numpy as np
from lib.continual.dfcl_base_learner import base_learner
from lib.dataset import SubDataset, TransformedDataset_for_exemplars
from lib.dataset.FeatureSet import FeatureDataset
from lib.model import CrossEntropy
from lib.utils import AverageMeter, accuracy


class FeTrIL_learner(base_learner):
    def __init__(self, cfg, dataset_handler, logger):
        super(FeTrIL_learner, self).__init__(cfg, dataset_handler, logger)
        self.mean_feature_per_class = None
        self.features_per_current_class = []
        self.clf = None

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
            self.logger.info(f"construct_exemplar_set class_id: {class_id + self.dataset_handler.classes_per_task * (task - 1)}")
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
        dataloader = DataLoader(train_dataset, batch_size=128, shuffle=False, num_workers=2,
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

    def learn_new_task(self, train_dataset, active_classes_num, task):
        self.pre_tasks_model = copy.deepcopy(self.model).eval()
        dp_class_num = self.dataset_handler.classes_per_task * (task - 1)
        if not self.cfg.use_base_half and task == 2:
            self.store_mean_feature_per_class(1)
        elif self.cfg.use_base_half and task == int(self.dataset_handler.all_tasks / 2) + 1:
            for task_id in range(1, task):
                self.store_mean_feature_per_class(task_id)
        self.store_mean_feature_per_class(task)

        '''Contrust features Dataset'''
        current_task_mean_features = self.mean_feature_per_class[dp_class_num:, :]
        previous_task_mean_features = self.mean_feature_per_class[:dp_class_num, :]
        assert len(self.mean_feature_per_class) == active_classes_num, \
            f"len(self.mean_feature_per_class): {len(self.mean_feature_per_class)}"
        assert len(current_task_mean_features) == self.dataset_handler.classes_per_task, \
            f"len(current_task_mean_features): {len(current_task_mean_features)}"
        distances = pairwise_distances(previous_task_mean_features.numpy(), current_task_mean_features.numpy())
        c_distance_mini = np.argmin(distances, axis=1)
        if self.cfg.model.use_svm:
            self.logger.info(f"svm training")
            x = None
            y = None
            pseudo_features_per_old_class = []
            for class_id in range(dp_class_num):
                features_item = self.features_per_current_class[c_distance_mini[class_id]] - \
                                np.expand_dims(current_task_mean_features[c_distance_mini[class_id]], 0) + \
                                np.expand_dims(previous_task_mean_features[class_id], 0)
                if x is None:
                    x = torch.from_numpy(features_item)
                    y = torch.from_numpy(np.array([class_id] * len(features_item)))
                else:
                    x = torch.cat([x, torch.from_numpy(features_item)], dim=0)
                    y = torch.cat([y, torch.from_numpy(np.array([class_id] * len(features_item)))], dim=0)

            for class_id in range(self.dataset_handler.classes_per_task):
                x = torch.cat([x, torch.from_numpy(self.features_per_current_class[class_id])], dim=0)
                y = torch.cat([y, torch.from_numpy(np.array([class_id + dp_class_num] *
                                                            len(self.features_per_current_class[class_id])))], dim=0)
            self.clf = None
            self.clf = LinearSVC(penalty='l2', dual=False, tol=0.0001, C=1., multi_class='ovr',
                                 fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0)
            self.clf.fit(x, y)
            self.logger.info(f"svm training End.")
            pass
        else:
            pseudo_features_per_old_class = []
            for class_id in range(dp_class_num):
                features_item = self.features_per_current_class[c_distance_mini[class_id]] - \
                                np.expand_dims(current_task_mean_features[c_distance_mini[class_id]], 0) + \
                                np.expand_dims(previous_task_mean_features[class_id], 0)
                pseudo_features_per_old_class.append(features_item)
            training_features_per_class = pseudo_features_per_old_class
            for class_id in range(self.dataset_handler.classes_per_task):
                training_features_per_class.append(self.features_per_current_class[class_id])
            training_features_sets = FeatureDataset(training_features_per_class)
            active_classes_num = self.dataset_handler.classes_per_task * task
            self.train_classifier(training_features_sets, active_classes_num, task)

    def learn_new_task_for_local_dataset(self, train_dataset, active_classes_num, task, train_dataset_transform):
        self.pre_tasks_model = copy.deepcopy(self.model).eval()
        dp_class_num = self.dataset_handler.classes_per_task * (task - 1)
        if not self.cfg.use_base_half and task == 2:
            self.store_mean_feature_per_class_for_local_dataset(train_dataset_transform, 1)
        elif self.cfg.use_base_half and task == int(self.dataset_handler.all_tasks / 2) + 1:
            for task_id in range(1, task):
                self.store_mean_feature_per_class_for_local_dataset(train_dataset_transform, task_id)
        self.store_mean_feature_per_class_for_local_dataset(train_dataset_transform, task)

        '''Contrust features Dataset'''
        current_task_mean_features = self.mean_feature_per_class[dp_class_num:, :]
        previous_task_mean_features = self.mean_feature_per_class[:dp_class_num, :]
        assert len(self.mean_feature_per_class) == active_classes_num, \
            f"len(self.mean_feature_per_class): {len(self.mean_feature_per_class)}"
        assert len(current_task_mean_features) == self.dataset_handler.classes_per_task, \
            f"len(current_task_mean_features): {len(current_task_mean_features)}"
        distances = pairwise_distances(previous_task_mean_features.numpy(), current_task_mean_features.numpy())
        c_distance_mini = np.argmin(distances, axis=1)
        if self.cfg.model.use_svm:
            self.logger.info(f"svm training")
            x = None
            y = None
            pseudo_features_per_old_class = []
            for class_id in range(dp_class_num):
                features_item = self.features_per_current_class[c_distance_mini[class_id]] - \
                                np.expand_dims(current_task_mean_features[c_distance_mini[class_id]], 0) + \
                                np.expand_dims(previous_task_mean_features[class_id], 0)
                if x is None:
                    x = torch.from_numpy(features_item)
                    y = torch.from_numpy(np.array([class_id] * len(features_item)))
                else:
                    x = torch.cat([x, torch.from_numpy(features_item)], dim=0)
                    y = torch.cat([y, torch.from_numpy(np.array([class_id] * len(features_item)))], dim=0)

            for class_id in range(self.dataset_handler.classes_per_task):
                x = torch.cat([x, torch.from_numpy(self.features_per_current_class[class_id])], dim=0)
                y = torch.cat([y, torch.from_numpy(np.array([class_id + dp_class_num] *
                                                            len(self.features_per_current_class[class_id])))], dim=0)
            self.clf = None
            self.clf = LinearSVC(penalty='l2', dual=False, tol=0.0001, C=1., multi_class='ovr',
                                 fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0)
            self.clf.fit(x, y)
            self.logger.info(f"svm training End.")
            pass
        else:
            pseudo_features_per_old_class = []
            for class_id in range(dp_class_num):
                features_item = self.features_per_current_class[c_distance_mini[class_id]] - \
                                np.expand_dims(current_task_mean_features[c_distance_mini[class_id]], 0) + \
                                np.expand_dims(previous_task_mean_features[class_id], 0)
                pseudo_features_per_old_class.append(features_item)
            training_features_per_class = pseudo_features_per_old_class
            for class_id in range(self.dataset_handler.classes_per_task):
                training_features_per_class.append(self.features_per_current_class[class_id])
            training_features_sets = FeatureDataset(training_features_per_class)
            active_classes_num = self.dataset_handler.classes_per_task * task
            self.train_classifier(training_features_sets, active_classes_num, task)

    def train_classifier(self, training_features_sets, active_classes_num, task):
        optimizer = self.build_optimize(model=self.model,
                                        base_lr=self.cfg.model.eeil_finetune_train.OPTIMIZER.BASE_LR,
                                        optimizer_type=self.cfg.model.eeil_finetune_train.OPTIMIZER.TYPE,
                                        momentum=self.cfg.model.eeil_finetune_train.OPTIMIZER.MOMENTUM,
                                        weight_decay=self.cfg.model.eeil_finetune_train.OPTIMIZER.WEIGHT_DECAY)
        scheduler = self.build_scheduler(optimizer, lr_type=self.cfg.model.eeil_finetune_train.LR_SCHEDULER.TYPE,
                                         lr_step=self.cfg.model.eeil_finetune_train.LR_SCHEDULER.LR_STEP,
                                         lr_factor=self.cfg.model.eeil_finetune_train.LR_SCHEDULER.LR_FACTOR,
                                         warmup_epochs=self.cfg.model.eeil_finetune_train.LR_SCHEDULER.WARM_EPOCH)
        MAX_EPOCH = self.cfg.model.eeil_finetune_train.MAX_EPOCH
        criterion = CrossEntropy()
        dpt_active_classes_num = active_classes_num - self.dataset_handler.classes_per_task
        best_acc = 0
        for epoch in range(1, MAX_EPOCH + 1):
            all_loss = [AverageMeter(), AverageMeter()]
            distance_loss = AverageMeter()
            acc = AverageMeter()
            if float(torch.__version__[:3]) < 1.3:
                scheduler.step()
            is_first_ite = True
            iters_left = 1
            iter_index = 0
            iter_num = 0
            while iters_left > 0:
                self.model.train()
                # Update # iters left on current data-loader(s) and, if needed, create new one(s)
                iters_left -= 1
                if is_first_ite:
                    is_first_ite = False
                    data_loader = iter(
                        DataLoader(dataset=training_features_sets,
                                   batch_size=self.cfg.model.eeil_finetune_train.BATCH_SIZE,
                                   num_workers=self.cfg.model.eeil_finetune_train.NUM_WORKERS, shuffle=True,
                                   drop_last=True))
                    # NOTE:  [train_dataset]  is training-set of current task
                    #      [training_dataset] is training-set of current task with stored exemplars added (if requested)
                    iter_num = iters_left = len(data_loader)
                    continue

                #####-----CURRENT BATCH-----#####
                try:
                    feature_batch, y_batch = next(data_loader)  # --> sample training data of current task
                except StopIteration:
                    raise ValueError("next(data_loader) error while read data. ")
                feature_batch, y_batch = feature_batch.to(self.device), y_batch.to(
                    self.device)  # --> transfer them to correct device
                # ---> Train MAIN MODEL
                # Train the main model with this batch
                # image, label, meta, active_classes_num, classes_per_task, criterion, optimizer,
                # previous_task_model, all_loss, acc, epoch, batch_index, number_batch, ** kwargs
                cnt = y_batch.shape[0]
                output = self.model(feature_batch, train_cls_use_features=True)
                output = output[:, 0:active_classes_num]
                _, now_result = torch.max(output, 1)
                now_acc, now_cnt = accuracy(now_result.cpu().numpy(), y_batch.cpu().numpy())
                cls_loss = criterion(output, y_batch)

                optimizer.zero_grad()
                cls_loss.backward()
                optimizer.step()
                now_acc = [now_acc]
                acc.update(now_acc[0], cnt)
                if iter_index % self.cfg.SHOW_STEP == 0:
                    pbar_str = "eeil-fine-tun, Epoch: {} || Batch:{:>3d}/{} || lr : {} " \
                               "|| Batch_Accuracy:{:>5.2f}".format(epoch, iter_index,
                                                                   iter_num,
                                                                   optimizer.param_groups[
                                                                       0]['lr'],
                                                                   acc.val * 100
                                                                   )
                    self.logger.info(pbar_str)
                iter_index += 1

                # if epoch % self.cfg.epoch_show_step == 0:
                # train_acc, train_loss = acc.avg, all_loss.avg
                # loss_dict, acc_dict = {"train_loss": train_loss}, {"train_acc": train_acc}
            if self.cfg.VALID_STEP != -1 and epoch % self.cfg.VALID_STEP == 0:
                pbar_str = "Validate Epoch: {} || lr: {} || epoch_cls_Loss:{:>5.3f}  || epoch_distill_Loss:{:>5.3f}" \
                           "epoch_Accuracy:{:>5.2f}".format(epoch, optimizer.param_groups[0]['lr'],
                                                            all_loss[0].avg, all_loss[1].avg,
                                                            acc.val * 100)

                self.logger.info(pbar_str)

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
                # if writer:
                #     writer.add_scalars("scalar/acc", acc_dict, epoch)
                #     writer.add_scalars("scalar/loss", loss_dict, epoch)

            if float(torch.__version__[:3]) >= 1.3:
                scheduler.step()

    def validate_with_svm(self, task, test_model=None, is_test=False):
        acc = []
        for task_id in range(task):  # 这里的task 从0 开始
            if self.dataset_handler.val_datasets and (not is_test):
                predict_result = self.validate_with_svm_per_task(self.dataset_handler.val_datasets[task_id])
            else:
                predict_result = self.validate_with_svm_per_task(self.dataset_handler.test_datasets[task_id])
            acc.append(predict_result)
            self.logger.info(
                f"task: {task} || per task {task_id}, validate_with_FC acc:{predict_result}"
            )
        acc = np.array(acc)
        return acc
        pass

    def validate_with_svm_per_task(self, val_dataset):
        # todo
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.cfg.model.eeil_finetune_train.BATCH_SIZE,
                                num_workers=self.cfg.model.eeil_finetune_train.NUM_WORKERS, shuffle=False,
                                drop_last=False)
        features = None
        labels = None
        for x, y in val_loader:
            x = x.to("cuda")
            feature_item = self.model(x, is_nograd=True, feature_flag=True)
            feature_item = feature_item.cpu()
            if features is None:
                features = feature_item
                labels = y
            else:
                features = torch.cat([features, feature_item], dim=0)
                labels = torch.cat([labels, y], dim=0)
        pred_score = self.clf.score(features.numpy(), labels.numpy())
        return pred_score
        pass
