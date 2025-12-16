import os
import random
import shutil

import numpy as np
import torch
from torchvision.transforms import transforms

from lib.continual import datafree
from lib.dataset import TransformedDataset, AVAILABLE_TRANSFORMS


class ZeroShotKD_handler:
    """Our approach DDC"""

    def __init__(self, dataset_handler, cfg, logger):
        self.dataset_handler = dataset_handler
        self.cfg = cfg
        self.logger = logger
        self.trainer_name = self.cfg.trainer_name
        self.trainer = None

        self.test_dataset = None

        self.handler_init()

    def handler_init(self):
        self.dataset_handler.get_dataset()
        self.trainer = datafree.__dict__[self.trainer_name](self.cfg, self.dataset_handler, self.logger)
        self.trainer.print_model()

        self.test_dataset = self.dataset_handler.test_datasets[-1]
        self.trainer.load_model(self.cfg.task1_MODEL)
        val_acc = self.trainer.validate_with_FC(task=1)  # task_id 从1开始
        pbar_str = f"verify self.model acc: {val_acc}, {val_acc.mean()}"
        self.logger.info(pbar_str)

    def deepInvert_KD_train_main(self):
        gpus = torch.cuda.device_count()
        self.logger.info(f"use {gpus} gpus")
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        torch.cuda.manual_seed(self.cfg.seed)
        # determinstic backend
        torch.backends.cudnn.deterministic = True

        '''mkdir direction for storing codes and checkpoint'''
        model_dir = os.path.join(self.cfg.OUTPUT_DIR, "models")
        code_dir = os.path.join(self.cfg.OUTPUT_DIR, "codes")

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        else:
            self.logger.info(
                "This directory has already existed, Please remember to modify your cfg.NAME"
            )
            print("os.path.exists(code_dir):", os.path.exists(code_dir))
            if os.path.exists(code_dir):
                shutil.rmtree(code_dir)
            assert not os.path.exists(code_dir)
        self.logger.info("=> output model will be saved in {}".format(model_dir))
        this_dir = os.path.dirname(__file__)
        ignore = shutil.ignore_patterns(
            "*.pyc", "*.so", "*.out", "*pycache*", "*.pth", "*build*", "*output*", "*datasets*"
        )
        shutil.copytree(os.path.join(this_dir, "../"), code_dir, ignore=ignore)

        '''Construct dataset for each task.'''
        if self.cfg.use_Contra_train_transform:
            train_dataset_transform = transforms.Compose([
                *AVAILABLE_TRANSFORMS[self.dataset_handler.dataset_name]['Contra_train_transform'],
            ])
        else:
            train_dataset_transform = transforms.Compose([
                *AVAILABLE_TRANSFORMS[self.dataset_handler.dataset_name]['train_transform'],
            ])
        active_classes_num = self.dataset_handler.all_classes
        train_dataset = TransformedDataset(self.dataset_handler.original_imgs_train_datasets[-1],
                                           transform=train_dataset_transform)
        self.trainer.train_main(train_dataset=train_dataset, active_classes_num=active_classes_num)
        self.trainer.evaluate_KL(datasets=self.test_dataset)
        self.trainer.features_extracted_store(dataset=self.test_dataset)

    def KL_metric_main(self):
        gpus = torch.cuda.device_count()
        self.logger.info(f"use {gpus} gpus")
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        torch.cuda.manual_seed(self.cfg.seed)
        # determinstic backend
        torch.backends.cudnn.deterministic = True
        '''mkdir direction for storing codes and checkpoint'''
        model_dir = os.path.join(self.cfg.OUTPUT_DIR, "models")
        code_dir = os.path.join(self.cfg.OUTPUT_DIR, "codes")

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        else:
            self.logger.info(
                "This directory has already existed, Please remember to modify your cfg.NAME"
            )
            print("os.path.exists(code_dir):", os.path.exists(code_dir))
            if os.path.exists(code_dir):
                shutil.rmtree(code_dir)
            assert not os.path.exists(code_dir)
        self.logger.info("=> output model will be saved in {}".format(model_dir))
        this_dir = os.path.dirname(__file__)
        ignore = shutil.ignore_patterns(
            "*.pyc", "*.so", "*.out", "*pycache*", "*.pth", "*build*", "*output*", "*datasets*"
        )
        shutil.copytree(os.path.join(this_dir, "../"), code_dir, ignore=ignore)
        self.trainer.load_student(self.cfg.student_MODEL)
        val_acc = self.trainer.validate_with_FC(task=1, test_model=self.trainer.student)  # task_id 从1开始
        pbar_str = f"verify self.student acc: {val_acc}, {val_acc.mean()}"
        self.logger.info(pbar_str)

        KL_Metric = self.trainer.evaluate_KL(datasets=self.test_dataset)
        self.logger.info(f'############# test_KL_metric between model and '
                         f'student model on the distribution of test_data: {KL_Metric.sum}, {KL_Metric.count}, '
                         f'{KL_Metric.avg}.##############')
        pass

    def cGAN_KD_train_main(self):
        gpus = torch.cuda.device_count()
        self.logger.info(f"use {gpus} gpus")
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        torch.cuda.manual_seed(self.cfg.seed)
        # determinstic backend
        torch.backends.cudnn.deterministic = True

        '''mkdir direction for storing codes and checkpoint'''
        model_dir = os.path.join(self.cfg.OUTPUT_DIR, "models")
        code_dir = os.path.join(self.cfg.OUTPUT_DIR, "codes")

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        else:
            self.logger.info(
                "This directory has already existed, Please remember to modify your cfg.NAME"
            )
            print("os.path.exists(code_dir):", os.path.exists(code_dir))
            if os.path.exists(code_dir):
                shutil.rmtree(code_dir)
            assert not os.path.exists(code_dir)
        self.logger.info("=> output model will be saved in {}".format(model_dir))
        this_dir = os.path.dirname(__file__)
        ignore = shutil.ignore_patterns(
            "*.pyc", "*.so", "*.out", "*pycache*", "*.pth", "*build*", "*output*", "*datasets*"
        )
        shutil.copytree(os.path.join(this_dir, "../"), code_dir, ignore=ignore)

        '''Construct dataset for each task.'''
        if self.cfg.use_Contra_train_transform:
            train_dataset_transform = transforms.Compose([
                *AVAILABLE_TRANSFORMS[self.dataset_handler.dataset_name]['Contra_train_transform'],
            ])
        else:
            train_dataset_transform = transforms.Compose([
                *AVAILABLE_TRANSFORMS[self.dataset_handler.dataset_name]['train_transform'],
            ])
        active_classes_num = self.dataset_handler.all_classes
        train_dataset = TransformedDataset(self.dataset_handler.original_imgs_train_datasets[-1],
                                           transform=train_dataset_transform)
        self.trainer.train_main(train_dataset=train_dataset, active_classes_num=active_classes_num)
        self.trainer.evaluate_KL(datasets=self.test_dataset)
        self.trainer.features_extracted_store(dataset=self.test_dataset)

    def GAN_KD_train_main(self):
        gpus = torch.cuda.device_count()
        self.logger.info(f"use {gpus} gpus")
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        torch.cuda.manual_seed(self.cfg.seed)
        # determinstic backend
        torch.backends.cudnn.deterministic = True

        '''mkdir direction for storing codes and checkpoint'''
        model_dir = os.path.join(self.cfg.OUTPUT_DIR, "models")
        code_dir = os.path.join(self.cfg.OUTPUT_DIR, "codes")

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        else:
            self.logger.info(
                "This directory has already existed, Please remember to modify your cfg.NAME"
            )
            print("os.path.exists(code_dir):", os.path.exists(code_dir))
            if os.path.exists(code_dir):
                shutil.rmtree(code_dir)
            assert not os.path.exists(code_dir)
        self.logger.info("=> output model will be saved in {}".format(model_dir))
        this_dir = os.path.dirname(__file__)
        ignore = shutil.ignore_patterns(
            "*.pyc", "*.so", "*.out", "*pycache*", "*.pth", "*build*", "*output*", "*datasets*"
        )
        shutil.copytree(os.path.join(this_dir, "../"), code_dir, ignore=ignore)

        '''Construct dataset for each task.'''
        if self.cfg.use_Contra_train_transform:
            train_dataset_transform = transforms.Compose([
                *AVAILABLE_TRANSFORMS[self.dataset_handler.dataset_name]['Contra_train_transform'],
            ])
        else:
            train_dataset_transform = transforms.Compose([
                *AVAILABLE_TRANSFORMS[self.dataset_handler.dataset_name]['train_transform'],
            ])
        active_classes_num = self.dataset_handler.all_classes
        train_dataset = TransformedDataset(self.dataset_handler.original_imgs_train_datasets[-1],
                                           transform=train_dataset_transform)
        self.trainer.train_main(train_dataset=train_dataset, active_classes_num=active_classes_num)
        # self.trainer.evaluate_KL(datasets=self.test_dataset)
        # self.trainer.features_extracted_store(dataset=self.test_dataset)

        # pseudo_dataset_root = os.path.join(self.cfg.OUTPUT_DIR, "pseudo_datasets")
        # self.trainer.construct_pseudo_datasets(pseudo_dataset_root=pseudo_dataset_root,
        #                                        active_classes_num=active_classes_num)
