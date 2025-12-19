import os
import random
import shutil

import numpy as np
import torch
from torchvision.transforms import transforms
from torch.backends import cudnn
from torch.utils.data import ConcatDataset
from lib.continual import datafree
from lib.dataset import TransformedDataset, AVAILABLE_TRANSFORMS


class MI_DFCL_handler:
    """Our approach DDC"""

    def __init__(self, dataset_handler, cfg, logger):
        self.dataset_handler = dataset_handler  # Torchvision_Datasets_Split负责吧数据集划分成多个任务
        self.cfg = cfg
        self.logger = logger
        self.trainer_name = self.cfg.trainer_name  #这里会实例化你选择的“算法实现类”，比如 FARLSTM_Trainer 或 FARLSTM_EFCL_Trainer（具体类名看工程）。
        self.trainer = None

        self.start_task_id = None

        self.handler_init()

    def handler_init(self):
        """
        假设论文每一阶段都有一个旧模型f_t-1,和当前阶段数据D_t,
        RESUME分支：1、从checkpoint恢复之前的f_t-1、历史任务划分(split_selected_data)。2、从中间task继续Stage1+Stage2的训练
        :return:
        """
        if self.cfg.RESUME.use_resume:
            self.logger.info(f"use_resume: {self.cfg.RESUME.resumed_model_path}")
            checkpoint = torch.load(self.cfg.RESUME.resumed_model_path)
            self.start_task_id = checkpoint['task_id']
            self.dataset_handler.update_split_selected_data(checkpoint["split_selected_data"])
            self.dataset_handler.get_dataset()
            self.trainer = datafree.__dict__[self.trainer_name](self.cfg, self.dataset_handler, self.logger)
            self.trainer.resume(self.cfg.RESUME.resumed_model_path, checkpoint)
            self.trainer.print_model()
        else:
            #get_dataset()：1、划分数据集；2、初始化数据增强器；3、初始化数据集；4、初始化模型；5、初始化优化器；6、初始化学习率衰减器；7、初始化评价指标；8、初始化日志记录器；9、初始化训练器。
            self.dataset_handler.get_dataset()
            self.trainer = datafree.__dict__[self.trainer_name](self.cfg, self.dataset_handler, self.logger)
            self.trainer.print_model()

    def midfcl_train_main(self):
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

        '''Construct dataset for each task.
        数据增强：对应论文 IV.A 中的数据预处理，对应论文 Experiment Setup 里关于数据增强的描述（padding + random crop + horizontal flip 等
        '''
        if self.cfg.use_Contra_train_transform:
            train_dataset_transform = transforms.Compose([
                *AVAILABLE_TRANSFORMS[self.dataset_handler.dataset_name]['Contra_train_transform'],
            ])
        else:
            train_dataset_transform = transforms.Compose([
                *AVAILABLE_TRANSFORMS[self.dataset_handler.dataset_name]['train_transform'],
            ])
        """
        self.dataset_handler.dataset_name比如是 'CIFAR100' / 'mnist'，来自你前面的 Torchvision_Datasets_Split
            self.dataset_name = cfg.DATASET.dataset_name  # CIFAR100 / CIFAR10 / mnist…
            
        AVAILABLE_TRANSFORMS['CIFAR100'] = {
            "Contra_train_transform": [...],
            "train_transform": [...],
            "test_transform": [...],
        }

        
        """

        """
        任务循环：对应论文的「Base-0 / Base-Half 协议 + 每一阶段训练」
        确定从哪一个task开始训练，从哪一个task开始测试，以及每一阶段训练的task数量。
        """
        if not self.cfg.RESUME.use_resume:
            self.start_task_id = 1  # 如果没 resume，就从 task=1 开始（对应论文第一阶段t=1）
        else:
            self.start_task_id += 1 # 如果是 resume，则从 checkpoint 里的 task_id+1 开始继续训练

        #按阶段遍历数据集，每一阶段训练一个task
        train_dataset = None
        for task, original_imgs_train_dataset in enumerate(self.dataset_handler.original_imgs_train_datasets, #就是当前阶段的真实训练数据（还没加 transform 的）
                                                           1):
            self.logger.info(f'New task {task} begin.')

            if self.cfg.RESUME.use_resume and task < self.start_task_id:
                self.logger.info(f"Use resume. continue.")
                continue

                """
                Base-0 (cold-start)：所有类别平均切成 5/10/20 个阶段，第 1 阶段只看第一批类。

                Base-Half (warm-start)：前一半类别作为 base task，后面拆成 5 个阶段。
                """
            if self.cfg.use_base_half and task < int(self.dataset_handler.all_tasks / 2):
                train_dataset_temp = TransformedDataset(original_imgs_train_dataset, transform=train_dataset_transform)

                if train_dataset is None:
                    train_dataset = train_dataset_temp
                else:
                    train_dataset = ConcatDataset([train_dataset, train_dataset_temp])
                self.logger.info(f'task continue.')
                continue
            else:
                if self.cfg.use_base_half:
                    if task == int(self.dataset_handler.all_tasks / 2):
                        train_dataset_temp = TransformedDataset(original_imgs_train_dataset,
                                                                transform=train_dataset_transform)

                        train_dataset = ConcatDataset([train_dataset, train_dataset_temp])
                        self.logger.info(f'base_half dataset construct end.')
                        # self.batch_train_logger.info(f'base_half dataset construct end.')
                        self.logger.info(f'train_dataset length: {len(train_dataset)}.')
                    elif task > int(self.dataset_handler.all_tasks / 2):
                        train_dataset = TransformedDataset(original_imgs_train_dataset,
                                                           transform=train_dataset_transform)
                    else:
                        train_dataset = None
                else:
                    train_dataset = TransformedDataset(original_imgs_train_dataset, transform=train_dataset_transform)




            '''Train models to learn tasks
            真正训练的入口：和论文 Stage1 / Stage2 的连接点
            到了这里，我们已经拿到了当前阶段要用的 train_dataset，下一步是“让模型学这批任务”。
            对应论文的 Stage1 / Stage2 训练，我们会调用 trainer 里的 learn_new_task() 方法，具体实现看具体的 trainer 类。
            '''
            active_classes_num = self.dataset_handler.classes_per_task * task
            """active_classes_num= 每个任务的类别数 × 当前任务编号对应论文里的 “学到第 t 阶段时，模型要能分辨前 t 个阶段的所有类”。
            """
            if self.cfg.use_base_half and task == int(self.dataset_handler.all_tasks / 2) or \
                    (not self.cfg.use_base_half and task == 1):
                """
                Base-Half：task == all_tasks/2（即“base 任务”收尾阶段），或者Base-0：task == 1（第一阶段）， 则调用 first_task_train_main。
                """
                if self.cfg.train_first_task:
                    self.trainer.first_task_train_main(train_dataset, active_classes_num, task)#在第一阶段（或 base 任务）还不存在旧模型可以 distill，所以只需要传统的 CE 训练（没有 Stage1/Stage2 的复杂 KD）
                    #标准 supervised training，loss ≈ CE（可能带一点 regularization）。训练出初始的 𝑓_1，或base模型

                else:
                    self.trainer.load_model(self.cfg.task1_MODEL)
            else:
                self.trainer.learn_new_task(train_dataset, active_classes_num, task)
            if "DualConsistencyMI" == self.cfg.trainer_name:
                self.trainer.after_steps(train_dataset, task)
            self.logger.info(f'#############MCFM train task {task} End.##############')
            self.logger.info(f'#############Example handler task {task} start.##############')

            '''Evaluation.'''
            val_acc = self.trainer.validate_with_FC(task=task)
            if "DualConsistencyMI" == self.cfg.trainer_name:
                val_acc_Routing = self.trainer.validate_with_FC_Prototypical_Routing(task=task)
            taskIL_FC_val_acc = self.trainer.validate_with_FC_taskIL(task=task)
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
                if "DualConsistencyMI" == self.cfg.trainer_name:
                    test_acc_FC_str_Routing = f'task: {task} classififer:{"FC_Routing"} || test_acc: {val_acc_Routing}, ' \
                                      f'avg: {val_acc_Routing.mean()} '
                    self.logger.info(test_acc_FC_str_Routing)

                self.logger.info(f"validate taskIL: FC: {taskIL_FC_val_acc} || {taskIL_FC_val_acc.mean()}")

            if test_acc:
                if self.cfg.save_model:
                    self.trainer.save_best_latest_model_data(model_dir, task, test_acc.mean(),
                                                             self.cfg.model.TRAIN.MAX_EPOCH)
            else:
                if self.cfg.save_model:
                    self.trainer.save_best_latest_model_data(model_dir, task, val_acc.mean(),
                                                             self.cfg.model.TRAIN.MAX_EPOCH)


    def midfcl_train_main_for_local_dataset(self):
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

        if not self.cfg.RESUME.use_resume:
            self.start_task_id = 1  # self.start_task_id 从 1 开始
        else:
            self.start_task_id += 1
        train_dataset = None
        for task, original_imgs_train_dataset in enumerate(self.dataset_handler.original_imgs_train_datasets,
                                                           1):
            self.logger.info(f'New task {task} begin.')

            if self.cfg.RESUME.use_resume and task < self.start_task_id:
                self.logger.info(f"Use resume. continue.")
                continue

            if self.cfg.use_base_half and task < int(self.dataset_handler.all_tasks / 2):
                train_dataset_temp = TransformedDataset(original_imgs_train_dataset, transform=train_dataset_transform)

                if train_dataset is None:
                    train_dataset = train_dataset_temp
                else:
                    train_dataset = ConcatDataset([train_dataset, train_dataset_temp])
                self.logger.info(f'task continue.')
                continue
            else:
                if self.cfg.use_base_half:
                    if task == int(self.dataset_handler.all_tasks / 2):
                        train_dataset_temp = TransformedDataset(original_imgs_train_dataset,
                                                                transform=train_dataset_transform)

                        train_dataset = ConcatDataset([train_dataset, train_dataset_temp])
                        self.logger.info(f'base_half dataset construct end.')
                        # self.batch_train_logger.info(f'base_half dataset construct end.')
                        self.logger.info(f'train_dataset length: {len(train_dataset)}.')
                    elif task > int(self.dataset_handler.all_tasks / 2):
                        train_dataset = TransformedDataset(original_imgs_train_dataset,
                                                           transform=train_dataset_transform)
                    else:
                        train_dataset = None
                else:
                    train_dataset = TransformedDataset(original_imgs_train_dataset, transform=train_dataset_transform)

            '''Train models to learn tasks'''
            active_classes_num = self.dataset_handler.classes_per_task * task
            if self.cfg.use_base_half and task == int(self.dataset_handler.all_tasks / 2) or \
                    (not self.cfg.use_base_half and task == 1):
                if self.cfg.train_first_task:
                    self.trainer.first_task_train_main(train_dataset, active_classes_num, task)
                else:
                    self.trainer.load_model(self.cfg.task1_MODEL)
            else:
                self.trainer.learn_new_task(train_dataset, active_classes_num, task)
            self.logger.info(f'#############MCFM train task {task} End.##############')
            self.logger.info(f'#############Example handler task {task} start.##############')

            '''Evaluation.'''
            val_acc = self.trainer.validate_with_FC(task)
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

            if test_acc:
                if self.cfg.save_model:
                    self.trainer.save_best_latest_model_data(model_dir, task, test_acc.mean(),
                                                             self.cfg.model.TRAIN.MAX_EPOCH)
            else:
                if self.cfg.save_model:
                    self.trainer.save_best_latest_model_data(model_dir, task, val_acc.mean(),
                                                             self.cfg.model.TRAIN.MAX_EPOCH)
