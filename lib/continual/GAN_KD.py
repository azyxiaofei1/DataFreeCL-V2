import copy
import os

import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from .GAN_GEN import GAN_gen
from lib.data_transform.data_transform import AVAILABLE_TRANSFORMS, CIFAR_10_normalize
from lib.model import CrossEntropy, compute_distill_loss, compute_KL_distill_loss, compute_KL_distance
from lib.utils import AverageMeter, strore_features

'''Code for <Always Be Dreaming: A New Approach for Data-Free Class-Incremental Learning>'''


class GAN_KD(GAN_gen):

    def __init__(self, cfg, dataset_handler, logger):
        super(GAN_KD, self).__init__(cfg, dataset_handler, logger)
        self.criterion_fn = nn.CrossEntropyLoss(reduction="none")
        self.student = self.construct_model()
        self.student = self.transfer_to_cuda(self.student)

        self.best_student = None

        self.pseudo_datasets = None

        self.inv_normalize_transform = transforms.Compose([*AVAILABLE_TRANSFORMS[self.dataset_handler.dataset_name][
            'inv_normalize_transform']])

    def train_main(self, train_dataset, active_classes_num):
        self.after_steps(train_dataset)
        pseudo_dataset_root = os.path.join(self.cfg.OUTPUT_DIR, "pseudo_datasets")
        if not os.path.exists(pseudo_dataset_root):
            os.makedirs(pseudo_dataset_root)
        else:
            self.logger.info(
                "This directory has already existed, Please remember to modify your cfg.NAME"
            )
        self.save_pseudo_datasets(pseudo_dataset_root=pseudo_dataset_root,
                                  active_classes_num=active_classes_num)

        self.construct_pseudo_datasets(pseudo_dataset_root=pseudo_dataset_root,
                                       active_classes_num=active_classes_num)

        # self.gan_KD(train_dataset)

    def gan_KD(self, train_dataset):
        self.train_student(train_dataset)

        if self.cfg.save_model:
            model_dir = os.path.join(self.cfg.OUTPUT_DIR, "models")

            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            else:
                self.logger.info(
                    "This directory has already existed, Please remember to modify your cfg.NAME"
                )

            self.save_student(model_dir)

    def train_student(self, train_dataset):
        # trains
        optimizer = self.build_optimize(model=self.student,
                                        base_lr=self.cfg.student.TRAIN.OPTIMIZER.BASE_LR,
                                        optimizer_type=self.cfg.student.TRAIN.OPTIMIZER.TYPE,
                                        momentum=self.cfg.student.TRAIN.OPTIMIZER.MOMENTUM,
                                        weight_decay=self.cfg.student.TRAIN.OPTIMIZER.WEIGHT_DECAY)
        scheduler = self.build_scheduler(optimizer, lr_type=self.cfg.student.TRAIN.LR_SCHEDULER.TYPE,
                                         lr_step=self.cfg.student.TRAIN.LR_SCHEDULER.LR_STEP,
                                         lr_factor=self.cfg.student.TRAIN.LR_SCHEDULER.LR_FACTOR,
                                         warmup_epochs=self.cfg.student.TRAIN.LR_SCHEDULER.WARM_EPOCH,
                                         MAX_EPOCH=self.cfg.student.TRAIN.MAX_EPOCH)

        train_loader = DataLoader(dataset=train_dataset, batch_size=self.cfg.student.TRAIN.BATCH_SIZE,
                                  num_workers=self.cfg.student.TRAIN.NUM_WORKERS, shuffle=True, drop_last=True,
                                  persistent_workers=True)
        iter_num = len(train_loader)
        best_acc = 0.
        assert self.cfg.student.use_cls or self.cfg.student.use_kd
        test_transform = transforms.Compose([CIFAR_10_normalize])
        for epoch in range(1, self.cfg.student.TRAIN.MAX_EPOCH + 1):
            all_loss = AverageMeter()
            if float(torch.__version__[:3]) < 1.3:
                scheduler.step()
            iter_index = 0
            for x, y in train_loader:
                self.student.train()
                # ---> Train MAIN MODEL
                # data replay
                x_replay = self.sample(self.teacher, len(x))
                x_replay = self.transNorm(x_replay, transform=test_transform)

                logits, _ = self.student(x_replay)

                pre_logits_replay, _ = self.model(x_replay, is_nograd=True)

                loss_kd = compute_distill_loss(logits, pre_logits_replay, temp=self.cfg.student.temp)

                total_loss = loss_kd
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                # measure accuracy and record loss
                all_loss.update(total_loss.data.item(), y.shape[0])
                if iter_index % self.cfg.SHOW_STEP == 0:
                    pbar_str = "Approach: {}| | Epoch: {} || Batch:{:>3d}/{}|| lr: {} " \
                               "|| Batch_Loss:{:>5.3f}".format(self.cfg.trainer_name, epoch, iter_index,
                                                               iter_num,
                                                               optimizer.param_groups[
                                                                   0]['lr'],
                                                               all_loss.val
                                                               )
                    self.logger.info(pbar_str)
                iter_index += 1
            if self.cfg.VALID_STEP != -1 and epoch % self.cfg.VALID_STEP == 0:

                val_acc = self.validate_with_FC(task=1, test_model=self.student)  # task_id 从1开始

                if val_acc.mean() > best_acc:
                    best_acc, best_epoch = val_acc.mean(), epoch
                    self.best_student = copy.deepcopy(self.student)
                    self.best_epoch = best_epoch
                    self.best_acc = best_acc
                    self.logger.info(
                        "--------------Best_Epoch:{:>3d}    Best_Acc:{:>5.2f}%--------------".format(
                            best_epoch, best_acc * 100
                        )
                    )

            if float(torch.__version__[:3]) >= 1.3:
                scheduler.step()

    def construct_pseudo_datasets(self, pseudo_dataset_root, active_classes_num):
        min_class_sample = 0
        num_sample_per_class = np.array([0 for i in range(active_classes_num)])
        self.pseudo_datasets = [[] for i in range(active_classes_num)]
        while min_class_sample < 16:
            x_replay = self.sample(self.teacher, 128)
            y_replay = self.evaluate_x_replay(x_replay, active_classes_num)
            for index in range(len(y_replay)):
                label_item = y_replay[index].cpu().item()
                self.pseudo_datasets[label_item].append(x_replay[index].cpu())
                num_sample_per_class[label_item] += 1
                if len(self.pseudo_datasets[label_item]) > 32:
                    self.pseudo_datasets[label_item] = self.pseudo_datasets[label_item][0:32]
            min_class_sample = num_sample_per_class.min()
        for label in range(active_classes_num):
            class_dataset = self.pseudo_datasets[label]
            fp = os.path.join(pseudo_dataset_root, str(label))
            index = 0
            if not os.path.exists(fp):
                os.makedirs(fp)
            for img in class_dataset:
                torchvision.utils.save_image(img, fp=os.path.join(fp, str(index) + ".png"))
                index += 1

    def save_pseudo_datasets(self, pseudo_dataset_root, active_classes_num):
        if not os.path.exists(pseudo_dataset_root):
            os.makedirs(pseudo_dataset_root)
        for batch_item in range(5):
            x_replay = self.sample(self.teacher, 128)
            # fake = gnet(fixed_noises)
            grid = torchvision.utils.make_grid(x_replay, nrow=16)
            torchvision.utils.save_image(grid, fp=os.path.join(pseudo_dataset_root, str(batch_item) + ".png"))
            # save_image(fake, f'./GAN_saved02/images_epoch{epoch:02d}_batch{batch_idx:03d}.png')
            # y_replay = self.evaluate_x_replay(x_replay, active_classes_num)
            # for index in range(len(x_replay)):
            #     inv_transformed_img = self.inv_normalize_transform(x_replay[index].cpu())
            #     ndarr = inv_transformed_img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu",
            #                                                                                       torch.uint8).numpy()
            #     fp = os.path.join(pseudo_dataset_root, str(batch_item))
            #     if not os.path.exists(fp):
            #         os.makedirs(fp)
            #     im = Image.fromarray(ndarr)
            #     im.save(os.path.join(fp, str(y_replay[index]) + ".png"))

    def evaluate_KL(self, datasets):
        KL_metric = self.compute_KL_metric(dataset=datasets, pre_model=self.model, compared_model=self.student)
        self.logger.info(f'############# test_KL_metric between model and '
                         f'student model on the distribution of test_data: {KL_metric.sum}, {KL_metric.count}, '
                         f'{KL_metric.avg}.##############')
        return KL_metric
        pass

    def compute_KL_metric(self, dataset, pre_model, compared_model):
        val_loader = DataLoader(dataset=dataset, batch_size=self.cfg.student.TRAIN.BATCH_SIZE,
                                num_workers=self.cfg.student.TRAIN.NUM_WORKERS, shuffle=False, drop_last=False)
        pre_mode = pre_model.training
        compared_mode = compared_model.training
        pre_model.eval()
        compared_model.eval()
        KL_metric = AverageMeter()
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            cnt = labels.shape[0]
            pre_model_out = pre_model(x=inputs, is_nograd=True, get_classifier=True)
            pre_model_out_for_distill = pre_model_out
            compared_model_out = compared_model(x=inputs, is_nograd=True, get_classifier=True)
            compared_model_out_for_distill = compared_model_out
            KL_metric_temp = compute_KL_distance(compared_model_out_for_distill,
                                                 pre_model_out_for_distill,
                                                 temp=1.)
            KL_metric.update(KL_metric_temp.data.item(), cnt)
        pre_model.train(mode=pre_mode)
        compared_model.train(mode=compared_mode)
        return KL_metric
        pass

    def save_student(self, model_dir):
        torch.save({
            'state_dict': self.student.state_dict(),
        }, os.path.join(model_dir, "{}_latest_model.pth".format("student"))
        )

        torch.save({
            'state_dict': self.best_student.state_dict(),
        }, os.path.join(model_dir, "{}_best_model.pth".format("student"))
        )
        pass

    def features_extracted_store(self, dataset):
        # 初始化 Network
        file_dir = os.path.join(self.cfg.OUTPUT_DIR, "extracted_features")
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        strore_features(self.student, dataset, file_dir, features_file="student_features.npy",
                        label_file='student_labels.npy')
        strore_features(self.model, dataset, file_dir, features_file="model_features.npy",
                        label_file='model_labels.npy')
        pass

    def validate_with_FC_per_task(self, model, val_dataset, task):
        # todo
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.cfg.student.TRAIN.BATCH_SIZE,
                                num_workers=self.cfg.student.TRAIN.NUM_WORKERS, shuffle=False, drop_last=False)
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

    def load_student(self, filename):
        assert filename is not None
        model = self.construct_model()
        model.load_model(filename)
        self.student = self.transfer_to_cuda(model)

    def evaluate_x_replay(self, x_replay, active_classes_num):
        out = self.model(x=x_replay, is_nograd=True, get_classifier=True)
        out = out[:, 0:active_classes_num]
        _, y_hat = torch.max(out, 1)
        print(f"y_hat: {y_hat}")
        return y_hat

    def transNorm(self, recon, transform):
        print(type(recon))
        print(recon.shape)
        recon = recon.cpu()
        recon_tansNorm = []
        for img in recon:
            recon_tansNorm.append(transform(img))
        recon_tansNorm = torch.stack(recon_tansNorm)
        return recon_tansNorm.to(self.device)
