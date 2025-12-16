import os
import random
import shutil

import numpy as np
import torch
from torchvision.transforms import transforms
from torch.backends import cudnn
from torch.utils.data import ConcatDataset, DataLoader
from lib.continual.ABD_for_CAM import ABD_for_CAM
from lib.dataset import TransformedDataset, AVAILABLE_TRANSFORMS, contest_transforms, copy
from lib.model import cam_resnet18, resnet_model, FCTM_model
from lib.utils import contruct_imgs_paths, AverageMeter
from lib.utils.util_class import GradCAM


class MI_DFCL_for_CAM_handler:
    """Our approach DDC"""

    def __init__(self, dataset_handler, cfg, logger):
        self.dataset_handler = dataset_handler
        self.cfg = cfg
        self.logger = logger
        self.trainer_name = self.cfg.trainer_name
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
            self.trainer = ABD_for_CAM(self.cfg, self.dataset_handler, self.logger)
            self.trainer.resume(self.cfg.RESUME.resumed_model_path, checkpoint)
            self.trainer.print_model()
        else:
            self.dataset_handler.get_dataset()
            self.trainer = ABD_for_CAM(self.cfg, self.dataset_handler, self.logger)
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

    def grad_cam_main(self, old_model_path, pre_model_path):
        self.dataset_handler.get_dataset()
        outputs = os.path.join(self.cfg.OUTPUT_DIR, "OldModel-cam-imgs")
        if not os.path.exists(outputs):
            os.makedirs(outputs)
        # old_model = cam_resnet18(class_num=self.dataset_handler.all_classes)
        pre_model = resnet_model(self.cfg)
        # old_model = FCTM_model(self.cfg)
        old_model = resnet_model(self.cfg)
        # if self.gpus > 1:
        #     if len(self.device_ids) > 1:
        #         self.model = torch.nn.DataParallel(self.model, device_ids=self.device_ids).cuda()
        #     else:
        #         self.model = torch.nn.DataParallel(self.model).cuda()
        # else:
        #     self.model = self.model.to("cuda")
        pre_model.load_model(pre_model_path)
        old_model.load_model(old_model_path)
        Model = copy.deepcopy(old_model)
        Pre_Model = copy.deepcopy(pre_model)
        val_acc = self.validate_with_FC_resnet18(model=Model, task=1)  # task_id 从1开始
        # val_acc = self.validate_with_FC_FCTM(model=Model, pre_model=Pre_Model, task=1)  # task_id 从1开始
        acc_avg = val_acc.mean()
        self.logger.info(
            "--------------current_Acc:{:>5.2f}%--------------".format(
                acc_avg * 100
            )
        )
        grad_cam = GradCAM(self.dataset_handler.dataset_name, old_model, 'stage5', pre_model, stored_path=outputs)
        # grad_cam = GradCAM(self.dataset_handler.dataset_name, old_model, target_layer="FCN", pre_model=pre_model,
        #                    stored_path=outputs)
        imgs_paths = contruct_imgs_paths(self.cfg.DATASET.data_json_file, self.cfg.DATASET.data_root)
        for class_name, paths in imgs_paths.items():
            for path in paths:
                grad_cam.forward(class_name, path, write=True)  # 直接修改图片路径就可以了
        pass

    def validate_with_FC_resnet18(self, model, task=None, is_test=False):
        acc = []
        Model = model.cuda()
        mode = Model.training
        Model.eval()
        for task_id in range(task):  # 这里的task 从0 开始
            if self.dataset_handler.val_datasets and (not is_test):
                predict_result = self.validate_with_FC_per_task_resent18(Model,
                                                                         self.dataset_handler.val_datasets[task_id],
                                                                         task)
            else:
                predict_result = self.validate_with_FC_per_task_resent18(Model,
                                                                         self.dataset_handler.test_datasets[task_id],
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

    def validate_with_FC_per_task_resent18(self, Model, val_dataset, task):
        # todo
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.cfg.model.TRAIN.BATCH_SIZE,
                                num_workers=self.cfg.model.TRAIN.NUM_WORKERS, shuffle=False, drop_last=False,
                                persistent_workers=True)
        top1 = AverageMeter()
        correct = 0
        active_classes_num = self.dataset_handler.classes_per_task * task
        with torch.no_grad():
            for inputs, labels in val_loader:
                correct_temp = 0
                inputs, labels = inputs.to("cuda"), labels.to("cuda")
                out, _ = Model(x=inputs)
                _, balance_fc_y_hat = torch.max(out[:, 0:active_classes_num], 1)
                correct_temp += balance_fc_y_hat.eq(labels.data).cpu().sum()
                correct += correct_temp
                top1.update((correct_temp / inputs.size(0)).item(), inputs.size(0))
        del val_loader
        return top1.avg
        pass

    def validate_with_FC_FCTM(self, model, pre_model, task=None, is_test=False):
        acc = []
        Model = model.cuda()
        pre_model = pre_model.cuda()
        mode = Model.training
        Model.eval()
        for task_id in range(task):  # 这里的task 从0 开始
            if self.dataset_handler.val_datasets and (not is_test):
                predict_result = self.validate_with_FC_per_task_FCTM(Model, pre_model,
                                                                         self.dataset_handler.val_datasets[task_id],
                                                                         task)
            else:
                predict_result = self.validate_with_FC_per_task_FCTM(Model, pre_model,
                                                                         self.dataset_handler.test_datasets[task_id],
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

    def validate_with_FC_per_task_FCTM(self, Model, pre_model, val_dataset, task):
        # todo
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.cfg.model.TRAIN.BATCH_SIZE,
                                num_workers=self.cfg.model.TRAIN.NUM_WORKERS, shuffle=False, drop_last=False,
                                persistent_workers=True)
        top1 = AverageMeter()
        correct = 0
        active_classes_num = self.dataset_handler.classes_per_task * task
        with torch.no_grad():
            for inputs, labels in val_loader:
                correct_temp = 0
                inputs, labels = inputs.to("cuda"), labels.to("cuda")
                pre_features = pre_model(inputs, is_nograd=True, feature_flag=True)
                outs = Model(x=inputs, pre_model_feature=pre_features)["all_logits"]
                _, balance_fc_y_hat = torch.max(outs[:, 0:active_classes_num], 1)
                correct_temp += balance_fc_y_hat.eq(labels.data).cpu().sum()
                correct += correct_temp
                top1.update((correct_temp / inputs.size(0)).item(), inputs.size(0))
        del val_loader
        return top1.avg
        pass
