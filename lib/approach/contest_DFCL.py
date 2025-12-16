import os
import random
import shutil

import numpy as np
import torch
from torch.backends import cudnn
from torch.utils.data import ConcatDataset
from lib.continual.contest_datafree import contest_DFCL
from lib.dataset import base


class contest_dataset_handler:
    """# Local_Datasets is dataset in our disc in the format of .jpeg, .png etc..
       # We give path to read the dataset and split it. The relative storage path is determined (refer to xxx)
       # The format of data_json_file is determined (refer to xx.json)
       # The format of split_selected_file is determined (refer to xx.json)"""

    def __init__(self, cfg):
        self.dataset_name = cfg.DATASET.dataset_name
        self.all_classes = cfg.DATASET.all_classes
        self.all_tasks = cfg.DATASET.all_tasks
        self.data_root = cfg.DATASET.data_root
        self.classes_per_task = cfg.DATASET.classes_per_task
        self.train_datasets = None
        self.val_datasets = None
        pass


class contest_handler:
    """Our approach DDC"""

    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.dataset_handler = None
        self.logger = logger
        self.trainer = None
        self.handler_init()

    def handler_init(self):
        self.construct_datasets()
        self.trainer = contest_DFCL(self.cfg, self.dataset_handler, self.logger)
        self.trainer.print_model()

    def construct_datasets(self):
        self.dataset_handler = contest_dataset_handler(self.cfg)
        self.dataset_handler.train_datasets = []
        self.dataset_handler.val_datasets = []
        for task in range(self.cfg.DATASET.all_tasks):
            train_dataset = base.__dict__[self.dataset_handler.dataset_name](self.dataset_handler.data_root, 'train',
                                                                             str(task))
            # val_dataset = base.__dict__[self.dataset_handler.dataset_name](self.dataset_handler.data_root, 'val',
            #                                                                str(task))
            self.dataset_handler.train_datasets.append(train_dataset)
            # self.dataset_handler.val_datasets.append(val_dataset)

    def contest_train_main(self):
        gpus = torch.cuda.device_count()
        self.logger.info(f"use {gpus} gpus")
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        torch.cuda.manual_seed(self.cfg.seed)
        # determinstic backend
        torch.backends.cudnn.deterministic = True

        for task, train_dataset in enumerate(self.dataset_handler.train_datasets, 1):
            self.logger.info(f'New task {task} begin.')

            '''Train models to learn tasks'''
            active_classes_num = self.dataset_handler.classes_per_task * task
            if task == 1:
                self.trainer.first_task_train_main(train_dataset, active_classes_num, task)
            else:
                self.trainer.learn_new_task(train_dataset, active_classes_num, task)
            self.logger.info(f'#############MCFM train task {task} End.##############')
            self.logger.info(f'#############Example handler task {task} start.##############')

            '''Evaluation.'''
            val_acc = self.trainer.validate_with_FC(task)
            taskIL_FC_val_acc = self.trainer.validate_with_FC_taskIL(task)
            self.logger.info(f'#############task: {task:0>3d} is finished Test begin. ##############')
            val_acc_FC_str = f'task: {task} classififer:{"FC"} || test_acc: {val_acc}, avg: {val_acc.mean()} '
            self.logger.info(val_acc_FC_str)
            self.logger.info(f"validate taskIL: FC: {taskIL_FC_val_acc} || {taskIL_FC_val_acc.mean()}")
