import os
import random
import shutil

import numpy as np
import torch
from torchvision.transforms import transforms
from torch.backends import cudnn
from torch.utils.data import ConcatDataset, DataLoader
from lib.continual import datafree, AverageMeter
from lib.dataset import TransformedDataset, AVAILABLE_TRANSFORMS, contest_transforms
from lib.model import FCTM_model, resnet_model, FCN_model
from lib.utils import FCTM_strore_features, FCN_strore_features


class MI_DFCL_handler:
    """Our approach DDC"""

    def __init__(self, dataset_handler, cfg, logger):
        self.device = torch.device("cpu" if cfg.CPU_MODE else "cuda")
        self.dataset_handler = dataset_handler
        self.cfg = cfg
        self.logger = logger
        self.gpus = torch.cuda.device_count()

    def construct_model(self):
        model = resnet_model(self.cfg)
        return model
        pass

    def visualize_train_main(self):
        self.logger.info(f"use {self.gpus} gpus")
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        torch.cuda.manual_seed(self.cfg.seed)
        # determinstic backend
        torch.backends.cudnn.deterministic = True

        '''mkdir direction for storing codes and checkpoint'''
        self.dataset_handler.get_dataset()
        cudnn.benchmark = True
        cudnn.enabled = True

        '''Construct dataset for each task.'''
        train_dataset_transform = transforms.Compose([
            *AVAILABLE_TRANSFORMS[self.dataset_handler.dataset_name]['test_transform'],
        ])

        # train_dataset_transform = transforms.Compose([
        #     *AVAILABLE_TRANSFORMS[self.dataset_handler.dataset_name]['train_transform'],
        # ])

        # 初始化 Network
        self.FCTM = FCTM_model(self.cfg)
        self.FCTM.load_model(self.cfg.fctm_model_path)
        # print(self.FCTM)
        self.model = self.construct_model()
        self.model.load_model(self.cfg.task1_MODEL)
        self.model = self.model.to("cuda")
        self.FCTM = self.FCTM.to("cuda")

        til_model_val_acc = self.validate_with_FC_taskIL(model=self.model, task=2)  # task_id 从1开始
        model_val_acc = self.validate_with_FC(model=self.model, task=2)  # task_id 从1开始
        fctm_val_acc = self.validate_with_fctm(old_model=self.model, fctm=self.FCTM, task=2)  # task_id 从1开始
        self.logger.info(
            f"--------------task-IL model_val_acc: {til_model_val_acc} ave_Acc:{til_model_val_acc.mean()}-------------"
        )
        self.logger.info(
            f"--------------model_val_acc: {model_val_acc} ave_Acc:{model_val_acc.mean()}-------------"
        )
        self.logger.info(
            f"--------------fctm_val_acc: {fctm_val_acc} ave_Acc:{fctm_val_acc.mean()}-------------"
        )
        file_dir = os.path.join(self.cfg.OUTPUT_DIR, "visualize_features")
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        val_datasets = None
        train_datasets = None
        for task in range(2):
            if val_datasets is None:
                val_datasets = self.dataset_handler.test_datasets[0]
                train_datasets = TransformedDataset(self.dataset_handler.original_imgs_train_datasets[0],
                                                    transform=train_dataset_transform)
            else:
                val_datasets = ConcatDataset([val_datasets, self.dataset_handler.test_datasets[task]])
                train_datasets = ConcatDataset(
                    [train_datasets, TransformedDataset(self.dataset_handler.original_imgs_train_datasets[task],
                                                        transform=train_dataset_transform)])
        print("length of val_datasets:", len(val_datasets))
        FCTM_strore_features(self.model, self.FCTM, val_datasets, file_dir,
                             features_file="features.npy",
                             calibrated_features_file="calibrated_features.npy", label_file='labels.npy')
        pass

    def CKDF_visualize_train_main(self):
        self.dataset_handler.get_dataset()
        gpus = torch.cuda.device_count()
        self.logger.info(f"use {gpus} gpus")
        cudnn.benchmark = True
        cudnn.enabled = True
        # 初始化 Network
        FCN = FCN_model(self.cfg)
        FCN.load_model(self.cfg.fctm_model_path)
        # print(self.FCTM)
        self.model = self.construct_model()
        self.model.load_model(self.cfg.task1_MODEL)

        self.model = self.model.to("cuda")
        FCN = FCN.to("cuda")

        '''Construct dataset for each task.'''
        # train_dataset_transform = transforms.Compose([
        #     *AVAILABLE_TRANSFORMS[self.dataset_handler.dataset_name]['test_transform'],
        # ])
        train_dataset_transform = transforms.Compose([
            *AVAILABLE_TRANSFORMS[self.dataset_handler.dataset_name]['train_transform'],
        ])

        model_val_acc = self.validate_with_FC_taskIL(model=self.model, task=2)  # task_id 从1开始
        fctm_val_acc = self.validate_with_fcn(old_model=self.model, fcn=FCN, task=2)  # task_id 从1开始
        self.logger.info(
            f"--------------model_val_acc: {model_val_acc} ave_Acc:{model_val_acc.mean()}-------------"
        )
        self.logger.info(
            f"--------------fctm_val_acc: {fctm_val_acc} ave_Acc:{fctm_val_acc.mean()}-------------"
        )
        file_dir = os.path.join(self.cfg.OUTPUT_DIR, "visualize_features")
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        val_datasets = None
        train_datasets = None
        for task in range(2):
            if val_datasets is None:
                val_datasets = self.dataset_handler.test_datasets[0]
                train_datasets = TransformedDataset(self.dataset_handler.original_imgs_train_datasets[0],
                                                    transform=train_dataset_transform)
            else:
                val_datasets = ConcatDataset([val_datasets, self.dataset_handler.test_datasets[task]])
                train_datasets = ConcatDataset(
                    [train_datasets, TransformedDataset(self.dataset_handler.original_imgs_train_datasets[task],
                                                        transform=train_dataset_transform)])
        print("length of val_datasets:", len(val_datasets))
        print("length of train_datasets:", len(train_datasets))
        FCN_strore_features(self.model, FCN, val_datasets, file_dir,
                            features_file="features.npy",
                            calibrated_features_file="calibrated_features.npy", label_file='labels.npy')
        pass

    def validate_with_FC(self, model=None, task=None, is_test=False):
        acc = []
        Model = model if model is not None else self.model
        mode = Model.training
        Model.eval()
        for task_id in range(task):  # 这里的task 从0 开始
            if self.dataset_handler.val_datasets and (not is_test):
                predict_result = self.validate_with_FC_per_task(Model, self.dataset_handler.val_datasets[task_id], task)
            else:
                predict_result = self.validate_with_FC_per_task(Model, self.dataset_handler.test_datasets[task_id],
                                                                task)
            acc.append(predict_result)
            self.logger.info(
                f"task: {task} || per task {task_id}, validate_with_FC acc:{predict_result}"
            )
        acc = np.array(acc)
        Model.train(mode=mode)
        # print(
        #     f"task {task} validate_with_exemplars, acc_avg:{acc.mean()}")
        # self.logger.info(
        #     f"per task {task}, validate_with_exemplars, avg acc:{acc.mean()}"
        #     f"-------------------------------------------------------------"
        # )
        return acc
        pass

    def validate_with_FC_per_task(self, Model, val_dataset, task):
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
            out = Model(x=inputs, is_nograd=True, get_classifier=True)
            _, balance_fc_y_hat = torch.max(out[:, 0:active_classes_num], 1)
            correct_temp += balance_fc_y_hat.eq(labels.data).cpu().sum()
            correct += correct_temp
            top1.update((correct_temp / inputs.size(0)).item(), inputs.size(0))
        del val_loader
        return top1.avg
        pass

    def validate_with_fctm(self, old_model, fctm, task):
        acc = []
        old_model.eval()
        fctm.eval()
        active_classes_num = self.dataset_handler.classes_per_task * task
        for task_id in range(task):  # 这里的task 从0 开始
            val_loader = DataLoader(dataset=self.dataset_handler.test_datasets[task_id],
                                    batch_size=self.cfg.model.TRAIN.BATCH_SIZE,
                                    num_workers=self.cfg.model.TRAIN.NUM_WORKERS, shuffle=False, drop_last=False,
                                    persistent_workers=True)
            top1 = AverageMeter()
            correct = 0
            for inputs, labels in val_loader:
                correct_temp = 0
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                features = old_model(x=inputs, is_nograd=True, feature_flag=True)
                fctm_outs = fctm(inputs, features, is_nograd=True)
                _, balance_fc_y_hat = torch.max(fctm_outs[:, 0:active_classes_num], 1)
                correct_temp += balance_fc_y_hat.eq(labels.data).cpu().sum()
                correct += correct_temp
                top1.update((correct_temp / inputs.size(0)).item(), inputs.size(0))
            del val_loader
            acc.append(top1.avg)
            self.logger.info(
                f"task: {task} || per task {task_id}, validate_with_fctm acc:{top1.avg}"
            )
        acc = np.array(acc)
        return acc
        pass

    def validate_with_fcn(self, old_model, fcn, task):
        acc = []
        old_model.eval()
        fcn.eval()
        active_classes_num = self.dataset_handler.classes_per_task * task
        for task_id in range(task):  # 这里的task 从0 开始
            val_loader = DataLoader(dataset=self.dataset_handler.test_datasets[task_id],
                                    batch_size=self.cfg.model.TRAIN.BATCH_SIZE,
                                    num_workers=self.cfg.model.TRAIN.NUM_WORKERS, shuffle=False, drop_last=False,
                                    persistent_workers=True)
            top1 = AverageMeter()
            correct = 0
            for inputs, labels in val_loader:
                correct_temp = 0
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                features = old_model(x=inputs, is_nograd=True, feature_flag=True)
                fctm_outs = fcn(features, is_nograd=True)
                _, balance_fc_y_hat = torch.max(fctm_outs[:, 0:active_classes_num], 1)
                correct_temp += balance_fc_y_hat.eq(labels.data).cpu().sum()
                correct += correct_temp
                top1.update((correct_temp / inputs.size(0)).item(), inputs.size(0))
            del val_loader
            acc.append(top1.avg)
            self.logger.info(
                f"task: {task} || per task {task_id}, validate_with_fctm acc:{top1.avg}"
            )
        acc = np.array(acc)
        return acc
        pass

    def validate_with_FC_taskIL(self, model=None, task=None, is_test=False):
        acc = []
        Model = model if model is not None else self.model
        mode = Model.training
        Model.eval()

        for task_id in range(task):  # 这里的task 从0 开始
            if self.dataset_handler.val_datasets and (not is_test):
                predict_result = self.validate_with_FC_per_task_taskIL(Model,
                                                                       self.dataset_handler.val_datasets[task_id],
                                                                       task_id, task=task)
            else:
                predict_result = self.validate_with_FC_per_task_taskIL(Model,
                                                                       self.dataset_handler.test_datasets[task_id],
                                                                       task_id, task=task)
            acc.append(predict_result)
            self.logger.info(
                f"task: {task} || per task {task_id}, validate_with_FC acc:{predict_result}"
            )
        acc = np.array(acc)
        Model.train(mode=mode)
        return acc
        pass

    def validate_with_FC_per_task_taskIL(self, Model, val_dataset, task_id, task=None):
        # todo
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.cfg.model.TRAIN.BATCH_SIZE,
                                num_workers=self.cfg.model.TRAIN.NUM_WORKERS, shuffle=False, drop_last=False,
                                persistent_workers=True)
        top1 = AverageMeter()
        correct = 0
        allowed_classes = [i for i in range(task_id * self.dataset_handler.classes_per_task,
                                            (task_id + 1) * self.dataset_handler.classes_per_task)]
        for inputs, labels in val_loader:
            correct_temp = 0
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            out = Model(x=inputs, is_nograd=True, get_classifier=True)
            _, balance_fc_y_hat = torch.max(out[:, allowed_classes], 1)
            balance_fc_y_hat += task_id * self.dataset_handler.classes_per_task
            correct_temp += balance_fc_y_hat.eq(labels.data).cpu().sum()
            correct += correct_temp
            top1.update((correct_temp / inputs.size(0)).item(), inputs.size(0))
        del val_loader
        return top1.avg
        pass
