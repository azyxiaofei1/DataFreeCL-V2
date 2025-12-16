import torch
from torch import nn, optim
import torch.nn.functional as F
import math
import numpy as np
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import trange

from lib.deep_inversion.feature_hook import DeepInversionFeatureHook
from lib.model import discriminator_loss, generator_loss
from lib.utils import AverageMeter
from lib.utils.freeze_util import unfreeze, model_freeze

"""
Some content adapted from the following:
@article{fang2019datafree,
    title={Data-Free Adversarial Distillation},	
    author={Gongfan Fang and Jie Song and Chengchao Shen and Xinchao Wang and Da Chen and Mingli Song},	  
    journal={arXiv preprint arXiv:1912.11006},	
    year={2019}
}
@inproceedings{yin2020dreaming,
	title = {Dreaming to Distill: Data-free Knowledge Transfer via DeepInversion},
	author = {Yin, Hongxu and Molchanov, Pavlo and Alvarez, Jose M. and Li, Zhizhong and Mallya, Arun and Hoiem, Derek and Jha, Niraj K and Kautz, Jan},
	booktitle = {The IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)},
	month = June,
	year = {2020}
}
"""


class Teacher:
    def __init__(self, solver, generator, generator_iters, class_idx, deep_inv_params):

        super().__init__()
        self.solver = solver
        self.generator = generator
        self.max_iters = generator_iters
        self.device = torch.device("cuda:0")
        # hyperparameters
        self.generator_lr = deep_inv_params[0]
        self.r_feature_weight = deep_inv_params[1]
        self.pr_scale = deep_inv_params[2]
        self.content_temp = deep_inv_params[3]
        print(
            f"{type(deep_inv_params)}: {deep_inv_params}, self.generator_lr: {type(self.generator_lr)}, {self.generator_lr}")

        # get class keys
        self.class_idx = list(class_idx)
        self.dp_classes_num = len(self.class_idx)

        # set up criteria for optimization
        self.criterion = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss(reduction="none").to(self.device)
        self.smoothing = Gaussiansmoothing(3, 5, 1)
        self.criterion_ce = nn.CrossEntropyLoss().to(self.device)

        # Create hooks for feature statistics catching
        self.feature_hooks = []

    @torch.no_grad()
    def sample(self, batch_size: int = None, dp_classes_num: int = None, return_scores: bool = False):
        solver_mode = self.solver.training
        gen_mode = self.generator.training
        self.solver.eval()
        self.generator.eval()
        if hasattr(self.generator, "module"):
            # print(f"self.generator has module")
            input = self.generator.module.sample(batch_size)
        else:
            # print(f"self.generator has not module")
            input = self.generator.sample(batch_size)
        # input = self.generator.sample(batch_size)
        logits, _ = self.solver(input)
        logits = logits[:, :dp_classes_num]
        target = logits.argmax(dim=1)
        self.solver.train(mode=solver_mode)
        self.generator.train(mode=gen_mode)
        return (input, target, logits) if return_scores else (input, target)

    def generate_score_by_features(self, features, active_classes_num=None, is_nograd=False):
        self.solver.eval()
        self.solver.zero_grad()
        if is_nograd:
            with torch.no_grad():
                y_hat = self.solver(features, is_nograd=True, get_out_use_features=True)
        else:
            y_hat = self.solver(features, train_cls_use_features=True)
        return y_hat[:, 0:active_classes_num]
        pass

    def generate_features_scores(self, x):
        self.solver.eval()
        with torch.no_grad():
            scores, features = self.solver(x, is_nograd=True)
        return scores, features
        pass

    def generate_features(self, x):
        self.solver.eval()
        with torch.no_grad():
            features = self.solver(x, is_nograd=True, feature_flag=True)
        return features
        pass

    def generate_scores(self, x, active_classes_num=None, return_label=False):

        # make sure solver is eval mode
        self.solver.eval()

        # get predicted logit-scores
        with torch.no_grad():
            y_hat, _ = self.solver(x)
        y_hat = y_hat[:, :active_classes_num]

        # get predicted class-labels (indexed according to each class' position in [allowed_predictions]!)
        _, y = torch.max(y_hat, dim=1)

        return (y, y_hat) if return_label else y_hat

    def generate_scores_pen(self, x):

        # make sure solver is eval mode
        self.solver.eval()

        # get predicted logit-scores
        with torch.no_grad():
            y_hat = self.solver.forward(x=x, pen=True)

        return y_hat

    def criterion_pr(self, inputs):
        input_pad = F.pad(inputs, (2, 2, 2, 2), mode="reflect")
        input_smooth = self.smoothing(input_pad)
        return F.mse_loss(inputs, input_smooth)

    def criterion_rf(self):
        #  return sum([hook.r_feature for hook in self.feature_hooks])
        return torch.stack([h.r_feature.to(self.device) for h in self.feature_hooks]).mean()

    def criterion_cb(self, output: torch.Tensor):
        logit_mu = output.softmax(dim=1).mean(dim=0)
        # ignore sign
        entropy = (logit_mu * logit_mu.log() / math.log(self.dp_classes_num)).sum()
        return 1 + entropy

    def train_step(self, batch_size):
        if hasattr(self.generator, "module"):
            # print(f"self.generator has module")
            input = self.generator.module.sample(batch_size)
        else:
            # print(f"self.generator has not module")
            input = self.generator.sample(batch_size)
        # input = self.generator.sample(batch_size)
        output, _ = self.solver(input)
        output = output[:, :self.dp_classes_num]
        target = output.data.argmax(dim=1)

        # print("input device:", input.device)
        # print("output device:", output.device)
        # print("target device:", target.device)
        # content loss
        loss_ce = self.criterion_ce(output / self.content_temp, target)

        # print("loss_ce  device:", loss_ce.device)

        # label diversity loss
        loss_cb = self.criterion_cb(output)

        # print("loss_cb  device:", loss_cb.device)

        # locally smooth prior
        loss_pr = self.pr_scale * self.criterion_pr(input)

        # print("loss_pr  device:", loss_pr.device)

        # feature statistics regularization
        loss_rf = self.r_feature_weight * self.criterion_rf()

        # print("loss_rf  device:", loss_rf.device)

        loss = loss_ce + loss_cb + loss_pr + loss_rf.to(loss_pr.device)

        loss_dict = {
            "ce": loss_ce,
            "cb": loss_cb,
            "pr": loss_pr,
            "rf": loss_rf,
            "total": loss,
        }

        return loss, loss_dict

    def register_feature_hooks(self):
        # Remove old before register
        for hook in self.feature_hooks:
            hook.remove()

        ## Create hooks for feature statistics catching
        for module in self.solver.modules():
            if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
                self.feature_hooks.append(DeepInversionFeatureHook(module, 0, self.r_feature_weight))

    def train_generator(self, batch_size):
        torch.cuda.empty_cache()
        self.register_feature_hooks()
        unfreeze(self.generator)
        gen_opt = Adam(params=self.generator.parameters(), lr=self.generator_lr)
        miniters = max(self.max_iters // 100, 1)
        pbar = trange(self.max_iters, miniters=miniters, desc="Inversion")
        print("self.solver device:", next(self.solver.parameters()).device)
        for current_iter in pbar:
            gen_opt.zero_grad()
            self.solver.zero_grad()
            self.generator.train()
            self.solver.eval()
            loss, loss_dict = self.train_step(batch_size)
            loss.backward()
            gen_opt.step()
            if (current_iter + 1) % miniters == 0:
                pbar.set_postfix({k: f"{v:.4f}" for k, v in loss_dict.items()})
        torch.cuda.empty_cache()
        self.generator.eval()


class Gaussiansmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(Gaussiansmoothing, self).__init__()
        kernel_size = [kernel_size] * dim
        sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1)).to(torch.device("cuda:0"))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)


class FCTM_Teacher:
    def __init__(self, model_solver, FCTM, generator, generator_iters, class_idx, old_classes_num, deep_inv_params):

        super().__init__()
        self.model_solver = model_solver
        self.FCTM = FCTM
        self.generator = generator
        self.max_iters = generator_iters

        # hyperparameters
        self.generator_lr = deep_inv_params[0]
        self.r_feature_weight = deep_inv_params[1]
        self.pr_scale = deep_inv_params[2]
        self.content_temp = deep_inv_params[3]
        print(
            f"{type(deep_inv_params)}: {deep_inv_params}, self.generator_lr: {type(self.generator_lr)}, {self.generator_lr}")

        # get class keys
        self.class_idx = list(class_idx)
        self.classes_num = len(self.class_idx)
        self.old_classes_num = old_classes_num

        # set up criteria for optimization
        self.criterion = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss(reduction="none").cuda()
        self.smoothing = Gaussiansmoothing(3, 5, 1)
        self.criterion_ce = nn.CrossEntropyLoss().cuda()

        # Create hooks for feature statistics catching
        self.model_feature_hooks = []
        self.FCTM_feature_hooks = []

    @torch.no_grad()
    def sample(self, batch_size: int = None, classes_num: int = None, return_scores: bool = False):
        solver_mode = self.model_solver.training
        FCTM_mode = self.FCTM.training
        gen_mode = self.generator.training
        self.model_solver.eval()
        self.FCTM.eval()
        self.generator.eval()
        inputs = self.generator.sample(batch_size)
        _, pre_model_features = self.model_solver(inputs)
        logits = self.FCTM(x=inputs, pre_model_feature=pre_model_features, is_nograd=True)
        logits = logits[:, :classes_num]
        target = logits.argmax(dim=1)
        self.model_solver.train(mode=solver_mode)
        self.FCTM.train(mode=FCTM_mode)
        self.generator.train(mode=gen_mode)
        return (inputs, target, logits) if return_scores else (inputs, target)

    @torch.no_grad()
    def condition_sample(self, batch_size: int = None, classes_condition: torch.tensor = None):
        solver_mode = self.model_solver.training
        FCTM_mode = self.FCTM.training
        gen_mode = self.generator.training
        self.model_solver.eval()
        self.FCTM.eval()
        self.generator.eval()
        inputs = self.generator.condition_sample(batch_size, classes_condition)
        _, pre_model_features = self.model_solver(inputs)
        logits = self.FCTM(x=inputs, pre_model_feature=pre_model_features, is_nograd=True)
        logits = logits[:, :self.classes_num]
        target = classes_condition.squeeze(dim=1)
        self.model_solver.train(mode=solver_mode)
        self.FCTM.train(mode=FCTM_mode)
        self.generator.train(mode=gen_mode)
        return inputs, target, logits

    def generate_score_by_features(self, features, active_classes_num=None, is_nograd=False):
        self.FCTM.eval()
        self.FCTM.zero_grad()
        if is_nograd:
            with torch.no_grad():
                y_hat = self.FCTM(features, pre_model_feature=None, train_cls_use_features=True)["all_logits"]
        else:
            y_hat = self.FCTM(features, pre_model_feature=None, train_cls_use_features=True)["all_logits"]
        return y_hat[:, 0:active_classes_num]
        pass

    '''outputs: logits & calibrated features'''

    def generate_features_scores(self, x):
        self.model_solver.eval()
        self.FCTM.eval()
        with torch.no_grad():
            _, features = self.model_solver(x, is_nograd=True)
            outputs = self.FCTM(x=x, pre_model_feature=features)
        return outputs["all_logits"], outputs["features"]
        pass

    '''outputs: calibrated features'''

    def generate_features(self, x):
        self.model_solver.eval()
        self.FCTM.eval()
        with torch.no_grad():
            _, features = self.model_solver(x, is_nograd=True)
            outputs = self.FCTM(x=x, pre_model_feature=features)
        return outputs["features"]
        pass

    def generate_scores(self, x, active_classes_num=None, return_label=False):

        # make sure solver is eval mode
        self.model_solver.eval()
        self.FCTM.eval()

        # get predicted logit-scores
        with torch.no_grad():
            _, features = self.model_solver(x, is_nograd=True)
            outputs = self.FCTM(x=x, pre_model_feature=features)
        y_hat = outputs["all_logits"][:, :active_classes_num]

        # get predicted class-labels (indexed according to each class' position in [allowed_predictions]!)
        _, y = torch.max(y_hat, dim=1)

        return (y, y_hat) if return_label else y_hat

    # def generate_scores_pen(self, x):
    #
    #     # make sure solver is eval mode
    #     self.model_solver.eval()
    #     self.FCTM.eval()
    #
    #     # get predicted logit-scores
    #     with torch.no_grad():
    #         y_hat = self.solver.forward(x=x, pen=True)
    #
    #     return y_hat

    def criterion_pr(self, inputs):
        input_pad = F.pad(inputs, (2, 2, 2, 2), mode="reflect")
        input_smooth = self.smoothing(input_pad)
        return F.mse_loss(inputs, input_smooth)

    # todo
    def criterion_rf(self, hooks):
        #  return sum([hook.r_feature for hook in self.feature_hooks])
        return torch.stack([h.r_feature for h in hooks]).mean()

    def criterion_cb(self, output: torch.Tensor):
        logit_mu = output.softmax(dim=1).mean(dim=0)
        # ignore sign
        entropy = (logit_mu * logit_mu.log() / math.log(self.classes_num)).sum()
        return 1 + entropy

    def train_step(self, batch_size):
        if hasattr(self.generator, "module"):
            input = self.generator.module.sample(batch_size)
        else:
            input = self.generator.sample(batch_size)
        _, pre_model_features = self.model_solver(input)
        outputs = self.FCTM(x=input, pre_model_feature=pre_model_features)
        outputs = outputs["all_logits"][:, :self.classes_num]
        target = outputs.data.argmax(dim=1)

        # content loss
        loss_ce = self.criterion_ce(outputs / self.content_temp, target)

        # label diversity loss
        loss_cb = self.criterion_cb(outputs)

        # locally smooth prior
        loss_pr = self.pr_scale * self.criterion_pr(input)

        # feature statistics regularization
        loss_rf = self.r_feature_weight * (self.criterion_rf(self.model_feature_hooks) +
                                           self.criterion_rf(self.FCTM_feature_hooks))

        loss = loss_ce + loss_cb + loss_pr + loss_rf

        loss_dict = {
            "ce": loss_ce,
            "cb": loss_cb,
            "pr": loss_pr,
            "rf": loss_rf,
            "total": loss,
        }

        return loss, loss_dict

    def train_step_for_old_classes(self, batch_size, old_classes_condition):
        if hasattr(self.generator, "module"):
            input = self.generator.module.condition_sample(batch_size, old_classes_condition)
        else:
            input = self.generator.condition_sample(batch_size, old_classes_condition)
        _, pre_model_features = self.model_solver(input)
        outputs = self.FCTM(x=input, pre_model_feature=pre_model_features)
        outputs = outputs["all_logits"][:, :self.classes_num]
        labels = old_classes_condition.squeeze(dim=1)

        # content loss
        loss_ce = self.criterion_ce(outputs / self.content_temp, labels)

        # label diversity loss
        # loss_cb = self.criterion_cb(outputs)

        # locally smooth prior
        loss_pr = self.pr_scale * self.criterion_pr(input)

        # feature statistics regularization
        loss_rf = self.r_feature_weight * self.criterion_rf(self.model_feature_hooks)

        loss = loss_ce + loss_pr + loss_rf

        loss_dict = {
            "ce": loss_ce,
            "pr": loss_pr,
            "rf": loss_rf,
            "total": loss,
        }

        return loss, loss_dict

    def register_feature_hooks(self):
        # Remove old before register
        for hook in self.model_feature_hooks:
            hook.remove()

        for hook in self.FCTM_feature_hooks:
            hook.remove()

        ## Create hooks for feature statistics catching
        for module in self.model_solver.modules():
            if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
                self.model_feature_hooks.append(DeepInversionFeatureHook(module, 0, self.r_feature_weight))

        for module in self.FCTM.fctm.modules():
            if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
                self.FCTM_feature_hooks.append(DeepInversionFeatureHook(module, 0, self.r_feature_weight))

    def train_generator(self, batch_size):
        torch.cuda.empty_cache()
        self.register_feature_hooks()
        unfreeze(self.generator)
        gen_opt = Adam(params=self.generator.parameters(), lr=self.generator_lr)
        miniters = max(self.max_iters // 100, 1)
        pbar = trange(self.max_iters, miniters=miniters, desc="Inversion")
        for current_iter in pbar:
            gen_opt.zero_grad()
            self.model_solver.zero_grad()
            self.FCTM.zero_grad()
            self.generator.train()
            self.model_solver.eval()
            self.FCTM.eval()
            loss, loss_dict = self.train_step(batch_size)
            loss.backward()
            gen_opt.step()
            if (current_iter + 1) % miniters == 0:
                pbar.set_postfix({k: f"{v:.4f}" for k, v in loss_dict.items()})
        torch.cuda.empty_cache()
        self.generator.eval()

    def train_generator_for_old_classes(self, batch_size, old_classes_condition=None):
        torch.cuda.empty_cache()
        self.register_feature_hooks()
        print(f"self.FCTM_feature_hooks: {self.FCTM_feature_hooks}, {len(self.FCTM_feature_hooks)}")
        unfreeze(self.generator)
        gen_opt = Adam(params=self.generator.parameters(), lr=self.generator_lr)
        miniters = max(self.max_iters // 100, 1)
        pbar = trange(self.max_iters, miniters=miniters, desc="Inversion")
        for current_iter in pbar:
            gen_opt.zero_grad()
            self.model_solver.zero_grad()
            self.FCTM.zero_grad()
            self.generator.train()
            self.model_solver.eval()
            self.FCTM.eval()
            loss, loss_dict = self.train_step_for_old_classes(batch_size, old_classes_condition)
            loss.backward()
            gen_opt.step()
            if (current_iter + 1) % miniters == 0:
                pbar.set_postfix({k: f"{v:.4f}" for k, v in loss_dict.items()})
        torch.cuda.empty_cache()
        self.generator.eval()


class cGAN_Teacher:
    def __init__(self, cfg, generator, discriminator, previous_generator):

        super().__init__()
        self.cfg = cfg
        self.cGAN_generator = generator
        self.cGAN_discriminator = discriminator
        self.pre_generator = previous_generator
        self.max_epochs = cfg.generator.generator_epochs
        self.device = torch.device("cuda:0")
        # hyperparameters
        self.generator_lr = cfg.generator.generator_lr
        self.discriminator_lr = cfg.discriminator.discriminator_lr

    @torch.no_grad()
    def sample(self, batch_size: int = None, determined_label=None):
        gen_mode = self.cGAN_generator.training
        self.cGAN_generator.eval()
        if hasattr(self.cGAN_generator, "module"):
            imgs, labels = self.cGAN_generator.module.sample(batch_size, is_nograd=True,
                                                             determined_labels=determined_label)
        else:
            imgs, labels = self.cGAN_generator.sample(batch_size, is_nograd=True,
                                                      determined_labels=determined_label)

        self.cGAN_generator.train(mode=gen_mode)
        return imgs, labels

    def train_cGAN(self, train_dataset):
        torch.cuda.empty_cache()
        unfreeze(self.cGAN_generator)
        unfreeze(self.cGAN_discriminator)
        if self.pre_generator is not None:
            model_freeze(self.pre_generator)
        generator_optimizer = optim.Adam(self.cGAN_generator.parameters(), lr=self.generator_lr, betas=(0.5, 0.999))
        discriminator_optimizer = optim.Adam(self.cGAN_discriminator.parameters(), lr=self.discriminator_lr,
                                             betas=(0.5, 0.999))

        train_loader = DataLoader(dataset=train_dataset, batch_size=self.cfg.generator.TRAIN.BATCH_SIZE,
                                  num_workers=self.cfg.generator.TRAIN.NUM_WORKERS, shuffle=True, drop_last=True,
                                  persistent_workers=True)
        iter_num = len(train_loader)
        for epoch in range(1, self.max_epochs + 1):
            D_loss_ave = AverageMeter()
            G_loss_ave = AverageMeter()
            for index, (real_images, real_labels) in enumerate(train_loader):
                self.cGAN_generator.train()
                self.cGAN_discriminator.train()
                discriminator_optimizer.zero_grad()
                real_images = real_images.to(self.device)
                real_labels = real_labels.to(self.device)

                if self.pre_generator is not None:
                    if hasattr(self.pre_generator, "module"):
                        pre_gen_imgs, pre_gen_labels = self.pre_generator.module.sample(
                            self.cfg.generator.TRAIN.BATCH_SIZE,
                            is_nograd=True)
                    else:
                        pre_gen_imgs, pre_gen_labels = self.pre_generator.sample(self.cfg.generator.TRAIN.BATCH_SIZE,
                                                                                 is_nograd=True)
                    real_images = torch.cat([pre_gen_imgs, real_images], dim=0)
                    real_labels = torch.cat([pre_gen_labels, real_labels], dim=0)
                real_labels = real_labels.unsqueeze(1).long()

                real_target = Variable(torch.ones(real_images.size(0), 1).to(self.device))
                fake_target = Variable(torch.zeros(real_images.size(0), 1).to(self.device))

                D_real_loss = discriminator_loss(self.cGAN_discriminator((real_images, real_labels)), real_target)

                # GENERATOR

                # noise_vector = torch.randn(real_images.size(0), self.cfg.generator.latent_size, device=self.device)
                # noise_vector = noise_vector.to(self.device)

                noise_vectors = torch.FloatTensor(
                    np.random.normal(0, 1, (real_images.size(0), self.cfg.generator.latent_size)))
                noise_vectors = noise_vectors.to(device=self.device)

                generated_image = self.cGAN_generator((noise_vectors, real_labels))
                # print(generated_image.size())

                output = self.cGAN_discriminator((generated_image.detach(), real_labels))
                D_fake_loss = discriminator_loss(output, fake_target)

                # TRAIN DISCRIMINATOR
                D_total_loss = D_real_loss + D_fake_loss

                D_total_loss.backward()
                discriminator_optimizer.step()
                D_loss_ave.update(D_total_loss.data.item(), real_labels.shape[0])

                # TRAIN GENERATOR
                generator_optimizer.zero_grad()
                G_loss = generator_loss(self.cGAN_discriminator((generated_image, real_labels)), real_target)

                G_loss.backward()
                generator_optimizer.step()
                G_loss_ave.update(G_loss.data.item(), real_labels.shape[0])
                if index % 30 == 0:
                    print('Epoch %d: [%d/%d]: D_loss: %.3f, G_loss: %.3f' % (epoch, index, iter_num, D_loss_ave.val,
                                                                             G_loss_ave.val))
            if self.cfg.VALID_STEP != -1 and epoch % self.cfg.VALID_STEP == 0:
                print('Epoch: [%d/%d]: D_loss: %.3f, G_loss: %.3f' % ((epoch), self.max_epochs, D_loss_ave.val,
                                                                      G_loss_ave.val))
        torch.cuda.empty_cache()

    def train_cGAN_linear(self, train_dataset, active_classes_num):
        torch.cuda.empty_cache()
        unfreeze(self.cGAN_generator)
        unfreeze(self.cGAN_discriminator)
        if self.pre_generator is not None:
            model_freeze(self.pre_generator)
        generator_optimizer = optim.Adam(self.cGAN_generator.parameters(), lr=self.generator_lr, betas=(0.5, 0.999))
        discriminator_optimizer = optim.Adam(self.cGAN_discriminator.parameters(), lr=self.discriminator_lr,
                                             betas=(0.5, 0.999))

        train_loader = DataLoader(dataset=train_dataset, batch_size=self.cfg.generator.TRAIN.BATCH_SIZE,
                                  num_workers=self.cfg.generator.TRAIN.NUM_WORKERS, shuffle=True, drop_last=True,
                                  persistent_workers=True)
        iter_num = len(train_loader)
        for epoch in range(1, self.max_epochs + 1):
            D_loss_ave = AverageMeter()
            G_loss_ave = AverageMeter()
            for index, (real_images, real_labels) in enumerate(train_loader):
                # convert img, labels into proper form
                real_images = real_images.to(self.device)
                real_labels = real_labels.to(self.device)

                if self.pre_generator is not None:
                    if hasattr(self.pre_generator, "module"):
                        pre_gen_imgs, pre_gen_labels = self.pre_generator.module.sample(
                            self.cfg.generator.TRAIN.BATCH_SIZE,
                            is_nograd=True)
                    else:
                        pre_gen_imgs, pre_gen_labels = self.pre_generator.sample(self.cfg.generator.TRAIN.BATCH_SIZE,
                                                                                 is_nograd=True)
                    real_images = torch.cat([pre_gen_imgs, real_images], dim=0)
                    real_labels = torch.cat([pre_gen_labels, real_labels], dim=0)

                # creating real and fake tensors of labels
                reall = Variable(torch.cuda.FloatTensor(real_images.size(0), 1).fill_(0.9))
                f_label = Variable(torch.cuda.FloatTensor(real_images.size(0), 1).fill_(0.))

                # initializing gradient
                generator_optimizer.zero_grad()
                discriminator_optimizer.zero_grad()

                #### TRAINING GENERATOR ####
                # Feeding generator noise and labels
                noise = Variable(torch.cuda.FloatTensor(
                    np.random.normal(0, 1, (real_images.size(0), self.cfg.generator.latent_size))))
                gen_labels = Variable(
                    torch.cuda.LongTensor(np.random.randint(0, active_classes_num, real_images.size(0))))

                gen_imgs = self.cGAN_generator(noise, gen_labels)

                # Ability for discriminator to discern the real v generated images
                validity = self.cGAN_discriminator(gen_imgs, gen_labels)

                # Generative loss function
                g_loss = generator_loss(validity, reall)

                # Gradients
                g_loss.backward()
                generator_optimizer.step()

                #### TRAINING DISCRIMINTOR ####

                discriminator_optimizer.zero_grad()

                # Loss for real images and labels
                validity_real = self.cGAN_discriminator(real_images, real_labels)
                d_real_loss = discriminator_loss(validity_real, reall)

                # Loss for fake images and labels
                validity_fake = self.cGAN_discriminator(gen_imgs.detach(), gen_labels)
                d_fake_loss = discriminator_loss(validity_fake, f_label)

                # Total discriminator loss
                d_loss = 0.5 * (d_fake_loss + d_real_loss)

                # calculates discriminator gradients
                d_loss.backward()
                discriminator_optimizer.step()
                D_loss_ave.update(d_loss.data.item(), real_labels.shape[0])
                G_loss_ave.update(g_loss.data.item(), real_labels.shape[0])
                if index % 30 == 0:
                    print('Epoch %d: [%d/%d]: D_loss: %.3f, G_loss: %.3f' % (epoch, index, iter_num, D_loss_ave.val,
                                                                             G_loss_ave.val))
            if self.cfg.VALID_STEP != -1 and epoch % self.cfg.VALID_STEP == 0:
                print('Epoch: [%d/%d]: D_loss: %.3f, G_loss: %.3f' % ((epoch), self.max_epochs, D_loss_ave.val,
                                                                      G_loss_ave.val))
        torch.cuda.empty_cache()


class GAN_Teacher:
    def __init__(self, cfg, generator, discriminator, previous_generator):

        super().__init__()
        self.cfg = cfg
        self.GAN_generator = generator
        self.GAN_discriminator = discriminator
        self.pre_generator = previous_generator
        self.max_epochs = cfg.generator.generator_epochs
        self.device = torch.device("cuda:0")
        self.generator_lr = cfg.generator.generator_lr
        self.discriminator_lr = cfg.discriminator.discriminator_lr

        self.criterion = nn.BCEWithLogitsLoss().cuda()

    @torch.no_grad()
    def sample(self, batch_size: int = None):
        gen_mode = self.GAN_generator.training
        self.GAN_generator.eval()
        if hasattr(self.GAN_generator, "module"):
            imgs = self.GAN_generator.module.sample(batch_size, is_nograd=True)
        else:
            imgs = self.GAN_generator.sample(batch_size, is_nograd=True)

        self.GAN_generator.train(mode=gen_mode)
        return imgs

    def train_GAN(self, train_dataset_for_gan):
        torch.cuda.empty_cache()
        unfreeze(self.GAN_generator)
        unfreeze(self.GAN_discriminator)
        if self.pre_generator is not None:
            model_freeze(self.pre_generator)
        generator_optimizer = optim.Adam(self.GAN_generator.parameters(), lr=self.generator_lr, betas=(0.5, 0.999))
        discriminator_optimizer = optim.Adam(self.GAN_discriminator.parameters(), lr=self.discriminator_lr,
                                             betas=(0.5, 0.999))

        train_loader = DataLoader(dataset=train_dataset_for_gan, batch_size=self.cfg.generator.TRAIN.BATCH_SIZE,
                                  num_workers=self.cfg.generator.TRAIN.NUM_WORKERS, shuffle=True, drop_last=True,
                                  persistent_workers=True)
        iter_num = len(train_loader)
        for epoch in range(1, self.max_epochs + 1):
            D_loss_ave = AverageMeter()
            G_loss_ave = AverageMeter()
            for batch_idx, (real_images, _) in enumerate(train_loader):
                real_images = real_images.to(self.device)
                if self.pre_generator is not None:
                    if hasattr(self.pre_generator, "module"):
                        pre_gen_imgs = self.pre_generator.module.sample(
                            self.cfg.generator.TRAIN.BATCH_SIZE,
                            is_nograd=True)
                    else:
                        pre_gen_imgs = self.pre_generator.sample(self.cfg.generator.TRAIN.BATCH_SIZE,
                                                                 is_nograd=True)
                    real_images = torch.cat([pre_gen_imgs, real_images], dim=0)

                batch_size = real_images.size(0)

                labels = torch.ones(batch_size).to(self.device)
                preds = self.GAN_discriminator(real_images)
                outputs = preds.reshape(-1)
                dloss_real = self.criterion(outputs, labels)
                dmean_real = outputs.sigmoid().mean()

                noises = torch.randn(batch_size, self.cfg.generator.latent_size, 1, 1).to(self.device)
                fake_images = self.GAN_generator(noises)
                labels = torch.zeros(batch_size).to(self.device)
                fake = fake_images.detach()

                preds = self.GAN_discriminator(fake)
                outputs = preds.view(-1)
                dloss_fake = self.criterion(outputs, labels)
                dmean_fake = outputs.sigmoid().mean()

                dloss = (dloss_real + dloss_fake) / 2
                self.GAN_discriminator.zero_grad()
                # discriminator_optimizer.zero_grad()
                dloss.backward()
                discriminator_optimizer.step()

                labels = torch.ones(batch_size).to(self.device)
                preds = self.GAN_discriminator(fake_images)
                outputs = preds.view(-1)
                gloss = self.criterion(outputs, labels)
                gmean_fake = outputs.sigmoid().mean()
                self.GAN_generator.zero_grad()
                # generator_optimizer.zero_grad()
                gloss.backward()
                generator_optimizer.step()

                D_loss_ave.update(dloss.data.item(), batch_size)
                G_loss_ave.update(gloss.data.item(), batch_size)

                if batch_idx % 100 == 0:
                    print('Epoch %d: [%d/%d]: D_loss: %.3f, G_loss: %.3f' % (epoch, batch_idx, iter_num, D_loss_ave.val,
                                                                             G_loss_ave.val))
                    print(f'Discriminator tells real images real ability: {dmean_real}', '\n',
                          f'Discriminator tells fake images real ability: {dmean_fake:g}/{gmean_fake:g}')

            if self.cfg.VALID_STEP != -1 and epoch % self.cfg.VALID_STEP == 0:
                print('Epoch: [%d/%d]: D_loss: %.3f, G_loss: %.3f' % ((epoch), self.max_epochs, D_loss_ave.val,
                                                                      G_loss_ave.val))
        torch.cuda.empty_cache()
