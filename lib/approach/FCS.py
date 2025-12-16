import os
import random
import shutil

import numpy as np
import torch
from torchvision.transforms import transforms
from torch.backends import cudnn
from torch.utils.data import ConcatDataset
from lib.continual.FCS_method import *
from lib.data_transform.data_transform import CIFAR_10_normalize, CIFAR_100_normalize, tiny_imagenet_normalize, \
    imagenet_normalize
from lib.dataset import TransformedDataset, AVAILABLE_TRANSFORMS, contest_transforms, TransformedDataset_augment, \
    TransformedDataset_for_exemplars


class FCS_handler:
    """Our approach DDC"""

    def __init__(self, dataset_handler, cfg, logger):
        self.dataset_handler = dataset_handler
        self.cfg = cfg
        self.logger = logger
        self.trainer = None
        self.start_task_id = None
        self.handler_init()

    def handler_init(self):
        if self.cfg.RESUME.use_resume:
            self.logger.info(f"use_resume: {self.cfg.RESUME.resumed_model_path}")
            checkpoint = torch.load(self.cfg.RESUME.resumed_model_path)
            self.start_task_id = checkpoint['task_id']
            self.dataset_handler.update_split_selected_data(checkpoint["split_selected_data"])
            self.dataset_handler.get_dataset()
            self.trainer = FCS_trainer(self.cfg, self.dataset_handler, self.logger)
            self.trainer.resume(self.cfg.RESUME.resumed_model_path, checkpoint)
            self.trainer.print_model()
        else:
            self.dataset_handler.get_dataset()
            self.trainer = FCS_trainer(self.cfg, self.dataset_handler, self.logger)
            self.trainer.print_model()

    def fcs_train_main(self):
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
            # if os.path.exists(code_dir):
            #     shutil.rmtree(code_dir)
            assert not os.path.exists(code_dir)
        self.logger.info("=> output model will be saved in {}".format(model_dir))
        this_dir = os.path.dirname(__file__)
        ignore = shutil.ignore_patterns(
            "*.pyc", "*.so", "*.out", "*pycache*", "*.pth", "*build*", "*output*", "*datasets*"
        )
        # shutil.copytree(os.path.join(this_dir, "../"), code_dir, ignore=ignore)

        '''Construct dataset for each task.'''
        if self.cfg.use_Contra_train_transform:
            train_dataset_transform = transforms.Compose([
                *AVAILABLE_TRANSFORMS[self.dataset_handler.dataset_name]['Contra_train_transform'],
            ])
        else:
            train_dataset_transform = transforms.Compose([
                *AVAILABLE_TRANSFORMS[self.dataset_handler.dataset_name]['train_transform'],
            ])
        size = None
        normalize = None
        if "CIFAR10" == self.dataset_handler.dataset_name:
            size = 32
            normalize = CIFAR_10_normalize
        elif "CIFAR100" == self.dataset_handler.dataset_name:
            size = 32
            normalize = CIFAR_100_normalize
        elif "tiny" in self.dataset_handler.dataset_name:
            size = 64
            normalize = tiny_imagenet_normalize
        elif "imagenet" in self.dataset_handler.dataset_name:
            size = 224
            normalize = imagenet_normalize
        aug_trans = [
            transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
        # aug_trans = [
        #     transforms.RandomResizedCrop(size, scale=(0.1, 1.)),
        #     transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)],
        #                            p=0.8),
        #     transforms.RandomApply([transforms.GaussianBlur(kernel_size=[7, 7])], p=0.5),
        #     transforms.RandomGrayscale(p=0.2),
        #     transforms.RandomHorizontalFlip(p=0.5),  # 图像一半的概率翻转，一半的概率不翻转
        #     transforms.ToTensor(),
        #     # imagenet_normalize,  # use imagenet mean and std 归一化
        #     normalize,
        # ]
        augmentation_transform = transforms.Compose([*aug_trans])

        if not self.cfg.RESUME.use_resume:
            self.start_task_id = 1  # self.start_task_id 从 1 开始
        else:
            self.start_task_id += 1
        train_dataset = None
        train_dataset_for_EM = None
        for task, original_imgs_train_dataset in enumerate(self.dataset_handler.original_imgs_train_datasets,
                                                           1):
            self.logger.info(f'New task {task} begin.')

            if self.cfg.RESUME.use_resume and task < self.start_task_id:
                self.logger.info(f"Use resume. continue.")
                continue

            if self.cfg.use_base_half and task < int(self.dataset_handler.all_tasks / 2):
                train_dataset_temp = TransformedDataset_augment(original_imgs_train_dataset,
                                                                transform=train_dataset_transform,
                                                                augment_transform=augmentation_transform)

                if train_dataset is None:
                    train_dataset = train_dataset_temp
                else:
                    train_dataset = ConcatDataset([train_dataset, train_dataset_temp])

                if self.cfg.exemplar_manager.store_original_imgs:
                    train_dataset_for_EM_temp = TransformedDataset_for_exemplars(original_imgs_train_dataset,
                                                                                 transform=
                                                                                 self.dataset_handler.val_test_dataset_transform)
                else:
                    train_dataset_for_EM_temp = TransformedDataset_for_exemplars(original_imgs_train_dataset,
                                                                                 transform=train_dataset_transform)

                if train_dataset_for_EM is None:
                    train_dataset_for_EM = train_dataset_for_EM_temp
                else:
                    train_dataset_for_EM = ConcatDataset([train_dataset_for_EM, train_dataset_for_EM_temp])
                self.logger.info(f'task continue.')
                continue
            else:
                if self.cfg.use_base_half:
                    if task == int(self.dataset_handler.all_tasks / 2):
                        train_dataset_temp = TransformedDataset_augment(original_imgs_train_dataset,
                                                                        transform=train_dataset_transform,
                                                                        augment_transform=augmentation_transform)

                        train_dataset = ConcatDataset([train_dataset, train_dataset_temp])
                        self.logger.info(f'base_half dataset construct end.')
                        # self.batch_train_logger.info(f'base_half dataset construct end.')
                        self.logger.info(f'train_dataset length: {len(train_dataset)}.')
                    elif task > int(self.dataset_handler.all_tasks / 2):
                        train_dataset = TransformedDataset_augment(original_imgs_train_dataset,
                                                                   transform=train_dataset_transform,
                                                                   augment_transform=augmentation_transform)
                    else:
                        train_dataset = None
                else:
                    train_dataset = TransformedDataset_augment(original_imgs_train_dataset,
                                                               transform=train_dataset_transform,
                                                               augment_transform=augmentation_transform)

            '''Train models to learn tasks'''
            active_classes_num = self.dataset_handler.classes_per_task * task
            if self.cfg.exemplar_manager.store_original_imgs:
                train_dataset_for_EM_temp = TransformedDataset_for_exemplars(original_imgs_train_dataset,
                                                                             transform=
                                                                             self.dataset_handler.val_test_dataset_transform)
            else:
                train_dataset_for_EM_temp = TransformedDataset_for_exemplars(original_imgs_train_dataset,
                                                                             transform=train_dataset_transform)

            if self.cfg.use_base_half and task == int(self.dataset_handler.all_tasks / 2):
                train_dataset_for_EM = ConcatDataset([train_dataset_for_EM, train_dataset_for_EM_temp])
            else:
                train_dataset_for_EM = train_dataset_for_EM_temp

            if self.cfg.use_base_half and task == int(self.dataset_handler.all_tasks / 2) or \
                    (not self.cfg.use_base_half and task == 1):
                if self.cfg.train_first_task:
                    # self.trainer.first_task_train_main(train_dataset, active_classes_num, task)
                    self.trainer.first_task_train_main_cls(train_dataset, active_classes_num, task)
                else:
                    self.trainer.load_model(self.cfg.task1_MODEL)

                self.trainer._build_protos(train_dataset_for_EM)
                self.trainer.after_task()
            else:
                self.trainer.learn_new_task(train_dataset, active_classes_num, task)
                self.trainer._build_protos(train_dataset_for_EM)
                self.trainer.after_task()
            self.logger.info(f'#############MCFM train task {task} End.##############')
            self.logger.info(f'#############Example handler task {task} start.##############')

            '''Evaluation.'''
            val_acc = self.trainer.validate_with_FC(task=task)  # todo
            taskIL_FC_val_acc = self.trainer.validate_with_FC_taskIL(task)
            test_acc = None
            self.logger.info(f'#############task: {task:0>3d} is finished Test begin. ##############')
            if self.dataset_handler.val_datasets:
                test_acc = self.trainer.validate_with_FC(task=task, is_test=True)
                taskIL_FC_test_acc = self.trainer.validate_with_FC_taskIL(task, is_test=True)

                val_acc_FC_str = f'task: {task} classififer:{"FC"} val_acc: {val_acc}, avg: {val_acc.mean()} '
                test_acc_FC_str = f'task: {task} classififer:{"FC"} || test_acc: {test_acc}, avg: {test_acc.mean()} '
                self.logger.info(val_acc_FC_str)
                self.logger.info(test_acc_FC_str)
                self.logger.info(f"validate taskIL: val FC: {taskIL_FC_val_acc} || {taskIL_FC_val_acc.mean()}")
                self.logger.info(f"validate taskIL: test FC: {taskIL_FC_test_acc} || {taskIL_FC_test_acc.mean()}")
            else:
                test_acc_FC_str = f'task: {task} classififer:{"FC"} || test_acc: {val_acc}, avg: {val_acc.mean()} '
                self.logger.info(test_acc_FC_str)
                self.logger.info(f"validate taskIL: FC: {taskIL_FC_val_acc} || {taskIL_FC_val_acc.mean()}")

            val_nme_acc = self.trainer.validate_with_prototype(task=task)  # todo Done!
            test_acc_FC_str = f'task: {task} classififer:{"NME"} || test_acc: {val_nme_acc}, avg: {val_nme_acc.mean()} '
            self.logger.info(test_acc_FC_str)
            print(f"len(protos): {len(self.trainer._protos)}")

            if test_acc:
                if self.cfg.save_model:
                    self.trainer.save_best_latest_model_data(model_dir, task, test_acc.mean(),
                                                             self.cfg.model.TRAIN.MAX_EPOCH)
            else:
                if self.cfg.save_model:
                    self.trainer.save_best_latest_model_data(model_dir, task, val_acc.mean(),
                                                             self.cfg.model.TRAIN.MAX_EPOCH)



