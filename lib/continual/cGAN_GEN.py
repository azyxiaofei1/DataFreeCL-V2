import copy
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from lib import generator_models
from lib.continual.datafree_helper import cGAN_Teacher
from lib.model import CrossEntropy, resnet_model, compute_distill_loss
from lib.utils import AverageMeter, get_optimizer, get_scheduler

'''Code for <Always Be Dreaming: A New Approach for Data-Free Class-Incremental Learning>'''


class cGAN_gen:
    def __init__(self, cfg, dataset_handler, logger):
        self.cfg = cfg
        self.dataset_handler = dataset_handler
        self.logger = logger
        # gen parameters
        self.model = None
        self.pre_model = None
        self.cGAN_generator = None
        self.previous_generator = None
        # gen parameters
        self.cGAN_discriminator = None

        self.teacher = None  # a class object including generator and the model

        self.gpus = torch.cuda.device_count()
        self.device_ids = [] if cfg.availabel_cudas == "" else \
            [i for i in range(len(self.cfg.availabel_cudas.strip().split(',')))]
        self.device = torch.device("cuda:0")

        self.train_init()

    def construct_model(self):
        model = resnet_model(self.cfg)
        return model
        pass

    def build_optimize(self, model=None, base_lr=None, optimizer_type=None, momentum=None, weight_decay=None):
        # todo Done
        MODEL = model if model is not None else self.model
        optimizer = get_optimizer(MODEL, BASE_LR=base_lr, optimizer_type=optimizer_type, momentum=momentum,
                                  weight_decay=weight_decay)

        return optimizer

    def build_scheduler(self, optimizer, lr_type=None, lr_step=None, lr_factor=None, warmup_epochs=None, MAX_EPOCH=200):
        # todo optimizer, lr_type=None, lr_step=None, lr_factor=None, warmup_epochs=None
        scheduler = get_scheduler(optimizer=optimizer, lr_type=lr_type, lr_step=lr_step, lr_factor=lr_factor,
                                  warmup_epochs=warmup_epochs, MAX_EPOCH=MAX_EPOCH)
        return scheduler

    def train_init(self):
        self.model = self.construct_model()
        self.cGAN_generator = self.create_generator()
        self.cGAN_discriminator = self.create_discriminator()
        self.model = self.transfer_to_cuda(self.model)
        self.cGAN_generator = self.transfer_to_cuda(self.cGAN_generator)
        self.cGAN_discriminator = self.transfer_to_cuda(self.cGAN_discriminator)

    def transfer_to_cuda(self, model):
        if self.gpus > 1:
            if len(self.device_ids) > 1:
                model = torch.nn.DataParallel(model, device_ids=self.device_ids).to(self.device)
            else:
                model = torch.nn.DataParallel(model).to(self.device)
        else:
            model = model.to("cuda")
        return model

    def load_model(self, filename):
        assert filename is not None
        del self.model
        model = self.construct_model()
        model.load_model(filename)
        self.model = self.transfer_to_cuda(model)

    def create_generator(self):
        generator = generator_models.cGAN.__dict__[self.cfg.generator.gen_model_name](self.cfg.DATASET.all_classes,
                                                                                      self.cfg.generator.latent_size)
        return generator

    def create_discriminator(self):
        discriminator = generator_models.cGAN.__dict__[self.cfg.discriminator.dis_model_name](
            self.cfg.DATASET.all_classes)
        return discriminator

    def resume(self, resumed_model_path, checkpoint):
        acc_result = checkpoint['acc_result']
        self.start_task_id = checkpoint['task_id']
        self.load_model(resumed_model_path)
        self.is_resume_legal(acc_result)
        pass

    def is_resume_legal(self, acc_result):
        self.logger.info(f"Resumed acc_result: {acc_result}")
        FC_acc = self.validate_with_FC(task=self.start_task_id)
        self.logger.info(f"validate resumed model: {FC_acc} || {FC_acc.mean()}")

    def print_model(self):
        print(f"self.model: {self.model}")
        print(f"self.cGAN_generator: {self.cGAN_generator}")
        print(f"self.cGAN_discriminator: {self.cGAN_discriminator}")
        pass

    ##########################################
    #           MODEL TRAINING               #
    ##########################################

    def first_task_train_main(self, train_dataset, active_classes_num, task_id):
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
        criterion = CrossEntropy().to(self.device)
        best_acc = 0
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.cfg.model.TRAIN.BATCH_SIZE,
                                  num_workers=self.cfg.model.TRAIN.NUM_WORKERS, shuffle=True, drop_last=True,
                                  persistent_workers=True)
        iter_num = len(train_loader)
        self.logger.info(f"type(self.model): {type(self.model)}.")
        for epoch in range(1, self.cfg.model.TRAIN.MAX_EPOCH + 1):
            all_loss = AverageMeter()
            if float(torch.__version__[:3]) < 1.3:
                scheduler.step()
            iter_index = 0
            for x, y in train_loader:
                self.model.train()
                x, y = x.to(self.device), y.to(self.device)  # --> transfer them to correct device
                # ---> Train MAIN MODEL
                cnt = y.shape[0]
                output, _ = self.model(x)
                output = output[:, 0:active_classes_num]
                loss = criterion(output, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                all_loss.update(loss.data.item(), cnt)
                if iter_index % self.cfg.SHOW_STEP == 0:
                    pbar_str = "Epoch: {} || Batch:{:>3d}/{}|| lr: {} || Batch_cls_Loss:{:>5.3f}" \
                               "".format(epoch, iter_index, iter_num,
                                         optimizer.param_groups[0]['lr'],
                                         all_loss.val
                                         )
                    self.logger.info(pbar_str)
                iter_index += 1

            # if epoch % self.cfg.epoch_show_step == 0:
            if self.cfg.VALID_STEP != -1 and epoch % self.cfg.VALID_STEP == 0:
                pbar_str = "First task train, Validate Epoch: {} || lr: {} || epoch_Loss:{:>5.3f}".format(epoch,
                                                                                                          optimizer.param_groups[
                                                                                                              0]['lr'],
                                                                                                          all_loss.val)
                self.logger.info(pbar_str)

                val_acc = self.validate_with_FC(task=task_id)  # task_id 从1开始
                acc_avg = val_acc.mean()
                self.logger.info(
                    "--------------current_Epoch:{:>3d}    current_Acc:{:>5.2f}%--------------".format(
                        epoch, acc_avg * 100
                    )
                )
                if acc_avg > best_acc:
                    best_acc, best_epoch = acc_avg, epoch
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
        del train_loader
        self.after_steps(train_dataset, active_classes_num)

    def learn_new_task(self, train_dataset, active_classes_num, task):
        dp_classes_num = active_classes_num - self.dataset_handler.classes_per_task
        # trains
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
        criterion = CrossEntropy().to(self.device)
        best_acc = 0.
        for epoch in range(1, self.cfg.model.TRAIN.MAX_EPOCH + 1):
            all_loss = AverageMeter()
            if float(torch.__version__[:3]) < 1.3:
                scheduler.step()
            iter_index = 0
            for x, y in train_loader:
                self.model.train()
                x, y = x.to(self.device), y.to(self.device)  # --> transfer them to correct device
                # ---> Train MAIN MODEL
                # data replay
                x_replay, y_replay = self.sample(self.teacher, len(x))
                x_com = torch.cat([x, x_replay], dim=0)
                y_com = torch.cat([y, y_replay], dim=0)

                logits, _ = self.model(x_com)
                logits = logits[:, 0:active_classes_num]

                loss_class = criterion(logits, y_com)

                logits_for_distill = logits[:, 0:dp_classes_num]

                pre_logits, _ = self.pre_model(x_com, is_nograd=True)
                pre_logits_for_distill = pre_logits[:, 0:dp_classes_num]

                loss_kd = compute_distill_loss(logits_for_distill, pre_logits_for_distill,
                                               temp=self.cfg.model.T)
                total_loss = loss_class + loss_kd
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                # measure accuracy and record loss
                all_loss.update(total_loss.data.item(), y_com.shape[0])
                if iter_index % self.cfg.SHOW_STEP == 0:
                    pbar_str = "Approach: {}| | Epoch: {} || Batch:{:>3d}/{}|| lr: {} " \
                               "|| Batch_Loss:{:>5.3f}".format("LwF-UE", epoch, iter_index,
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
        self.after_steps(train_dataset, active_classes_num)

    def after_steps(self, train_dataset, active_classes_num):
        self.pre_model = copy.deepcopy(self.model).eval()
        if self.teacher is not None:
            self.previous_generator = copy.deepcopy(self.cGAN_generator).eval()

        if hasattr(self.cGAN_generator, "module"):
            self.cGAN_generator.module.update_class_num(active_classes_num)
            self.cGAN_discriminator.module.update_class_num(active_classes_num)
        else:
            self.cGAN_generator.update_class_num(active_classes_num)
            self.cGAN_discriminator.update_class_num(active_classes_num)

        self.teacher = cGAN_Teacher(cfg=self.cfg, generator=self.cGAN_generator,
                                    discriminator=self.cGAN_discriminator,
                                    previous_generator=self.previous_generator)
        if "conv" in self.cfg.generator.gen_model_name:
            self.teacher.train_cGAN(train_dataset)
        else:
            self.teacher.train_cGAN_linear(train_dataset, active_classes_num)
        pass

    ##########################################
    #             MODEL UTILS                #
    ##########################################

    def count_parameter_gen(self):
        return sum(p.numel() for p in self.cGAN_generator.parameters())

    def sample(self, teacher, batch_size, labels=None):
        return teacher.sample(batch_size, determined_label=labels)

    def validate_with_FC(self, task, test_model=None, is_test=False):
        acc = []
        model = test_model if test_model is not None else self.model
        mode = model.training
        model.eval()
        for task_id in range(task):  # 这里的task 从0 开始
            if self.dataset_handler.val_datasets and (not is_test):
                predict_result = self.validate_with_FC_per_task(model, self.dataset_handler.val_datasets[task_id], task)
            else:
                predict_result = self.validate_with_FC_per_task(model, self.dataset_handler.test_datasets[task_id],
                                                                task)
            acc.append(predict_result)
            self.logger.info(
                f"task: {task} || per task {task_id}, validate_with_FC acc:{predict_result}"
            )
        acc = np.array(acc)
        model.train(mode=mode)
        return acc
        pass

    def validate_with_FC_per_task(self, model, val_dataset, task):
        # todo
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.cfg.model.TRAIN.BATCH_SIZE,
                                num_workers=self.cfg.model.TRAIN.NUM_WORKERS, shuffle=False, drop_last=False)
        top1 = AverageMeter()
        correct = 0
        active_classes_num = self.dataset_handler.classes_per_task * task
        for inputs, labels in val_loader:
            correct_temp = 0
            inputs, labels = inputs.to("cuda"), labels.to("cuda")
            out = model(x=inputs, is_nograd=True, get_classifier=True)
            out = out[:, 0:active_classes_num]
            _, balance_fc_y_hat = torch.max(out, 1)
            correct_temp += balance_fc_y_hat.eq(labels.data).cpu().sum()
            correct += correct_temp
            top1.update((correct_temp / inputs.size(0)).item(), inputs.size(0))
        return top1.avg
        pass

    def validate_with_FC_taskIL(self, task, test_model=None, is_test=False):
        acc = []
        model = test_model if test_model is not None else self.model
        mode = model.training
        model.eval()
        for task_id in range(task):  # 这里的task 从0 开始
            if self.dataset_handler.val_datasets and (not is_test):
                predict_result = self.validate_with_FC_per_task_taskIL(self.dataset_handler.val_datasets[task_id],
                                                                       task_id, model)
            else:
                predict_result = self.validate_with_FC_per_task_taskIL(self.dataset_handler.test_datasets[task_id],
                                                                       task_id, model)
            acc.append(predict_result)
            self.logger.info(
                f"task: {task} || per task {task_id}, validate_with_FC acc:{predict_result}"
            )
        acc = np.array(acc)
        self.model.train(mode=mode)
        return acc
        pass

    def validate_with_FC_per_task_taskIL(self, val_dataset, task_id, model):
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
            out = model(x=inputs, is_nograd=True, get_classifier=True)
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

        pass
