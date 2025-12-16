import torch
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchmetrics.functional import pairwise_cosine_similarity
from tqdm import trange

from lib.deep_inversion import DeepInversionFeatureHook
from lib.model import SoftTarget_CrossEntropy
from lib.utils import AverageMeter
from lib.utils.freeze_util import model_freeze, unfreeze


class dual_consist_Teacher:
    def __init__(self, cfg, solver, generator, discriminator, prototypes, class_idx):

        super().__init__()
        self.cfg = cfg
        self.GAN_generator = generator
        self.GAN_discriminator = discriminator
        self.max_iters = cfg.generator.generator_iters
        self.critic_iters = cfg.generator.critic_iters
        self.device = torch.device("cuda:0")
        self.solver = solver
        self.prototypes = prototypes
        prototypes_smi_matrix = pairwise_cosine_similarity(self.prototypes, self.prototypes)
        self.alpha_matrix_softmax = F.softmax(prototypes_smi_matrix, dim=0)
        # hyperparameters

        # get class keys
        self.class_idx = list(class_idx)
        self.dp_classes_num = len(self.class_idx)

        self.criterion_bce = nn.BCEWithLogitsLoss().cuda()
        self.mse_loss = nn.MSELoss(reduction="none").to(self.device)
        self.criterion_cls = nn.CrossEntropyLoss().to(self.device)
        self.criterion_SoftTarget_CE = SoftTarget_CrossEntropy().cuda()

    @torch.no_grad()
    def get_pre_outputs(self, imgs):
        logits, features = self.solver(imgs, is_nograd=True)
        return logits, features
        pass

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


    @torch.no_grad()
    def sample(self, batch_size: int = None, dp_classes_num: int = None, return_scores: bool = False):
        solver_mode = self.solver.training
        gen_mode = self.GAN_generator.training
        self.solver.eval()
        self.GAN_generator.eval()
        if hasattr(self.GAN_generator, "module"):
            # print(f"self.generator has module")
            # GAN_generator.sample: alpha_matrix_softmax, batch_size
            input, _ = self.GAN_generator.module.sample(alpha_matrix_softmax=self.alpha_matrix_softmax,
                                                     batch_size=batch_size)
        else:
            # print(f"self.generator has no module")
            input, _ = self.GAN_generator.sample(alpha_matrix_softmax=self.alpha_matrix_softmax,
                                              batch_size=batch_size)
        self.GAN_generator.train(mode=gen_mode)
        # input = self.generator.sample(batch_size)
        logits, _ = self.solver(input)
        logits = logits[:, :dp_classes_num]
        target = logits.argmax(dim=1)
        self.solver.train(mode=solver_mode)
        self.GAN_generator.train(mode=gen_mode)
        return (input, target, logits) if return_scores else (input, target)

    def train_generator(self, train_dataset_for_gan, batch_size):
        torch.cuda.empty_cache()
        model_freeze(self.solver)
        # generator_optimizer = optim.Adam(self.GAN_generator.parameters(), lr=self.cfg.generator.generator_lr,
        #                                  betas=(0.5, 0.999))
        # discriminator_optimizer = optim.Adam(self.GAN_discriminator.parameters(),
        #                                      lr=self.cfg.generator.discriminator_lr,
        #                                      betas=(0.5, 0.999))
        generator_optimizer = torch.optim.RMSprop(self.GAN_generator.parameters(), lr=self.cfg.generator.generator_lr)
        discriminator_optimizer = torch.optim.RMSprop(self.GAN_discriminator.parameters(),
                                                      lr=self.cfg.generator.discriminator_lr)
        miniters = max(self.max_iters // 100, 1)
        pbar = trange(self.max_iters, miniters=miniters, desc="Inversion")
        print("self.solver device:", next(self.solver.parameters()).device)
        dataset_loader = DataLoader(dataset=train_dataset_for_gan, batch_size=batch_size,
                                    num_workers=self.cfg.model.TRAIN.NUM_WORKERS, shuffle=True,
                                    drop_last=True, persistent_workers=True)
        dataset_iter_loader = iter(dataset_loader)
        dataset_iter_num = len(dataset_iter_loader)
        datasets_iter_index = 0

        for current_iter in pbar:
            self.solver.zero_grad()
            self.solver.eval()
            model_freeze(self.GAN_generator)
            unfreeze(self.GAN_discriminator)
            for critic_index in range(self.critic_iters):
                discriminator_optimizer.zero_grad()
                self.GAN_discriminator.zero_grad()
                self.GAN_discriminator.train()
                if datasets_iter_index == dataset_iter_num:
                    dataset_iter_loader = iter(dataset_loader)
                    datasets_iter_index = 0
                real_imgs, _ = next(dataset_iter_loader)
                datasets_iter_index += 1
                real_imgs = real_imgs.to(self.device)
                batch_size = real_imgs.size(0)
                for p in self.GAN_discriminator.parameters():
                    p.data.clamp_(-self.cfg.generator.weight_cliping_limit, self.cfg.generator.weight_cliping_limit)

                dis_real_preds = self.GAN_discriminator(real_imgs)
                dis_real_loss = dis_real_preds.mean().view(1)

                if hasattr(self.GAN_generator, "module"):
                    fake_imgs_for_old_classes, _ = \
                        self.GAN_generator.module.sample(alpha_matrix_softmax=self.alpha_matrix_softmax,
                                                         batch_size=batch_size)
                else:
                    fake_imgs_for_old_classes, _ = \
                        self.GAN_generator.sample(alpha_matrix_softmax=self.alpha_matrix_softmax, batch_size=batch_size)
                dis_fake_preds = self.GAN_discriminator(fake_imgs_for_old_classes)
                dis_fake_loss = dis_fake_preds.mean().view(1)
                d_loss = (dis_real_loss - dis_fake_loss) * self.cfg.generator.ParaLambda
                d_loss.backward()
                discriminator_optimizer.step()
            model_freeze(self.GAN_discriminator)
            unfreeze(self.GAN_generator)
            generator_optimizer.zero_grad()
            self.GAN_generator.zero_grad()
            self.GAN_generator.train()
            if hasattr(self.GAN_generator, "module"):
                gen_imgs_for_old_classes, seletected_alpha_softmax = \
                    self.GAN_generator.module.sample(alpha_matrix_softmax=self.alpha_matrix_softmax, batch_size=batch_size)
            else:
                gen_imgs_for_old_classes, seletected_alpha_softmax = \
                    self.GAN_generator.sample(alpha_matrix_softmax=self.alpha_matrix_softmax, batch_size=batch_size)
            gloss_for_wgan = self.GAN_discriminator(gen_imgs_for_old_classes)
            gloss_for_wgan = gloss_for_wgan.mean().view(1)
            gloss_for_wgan = gloss_for_wgan * self.cfg.generator.ParaLambda
            solver_outputs, solver_features = self.solver(gen_imgs_for_old_classes)
            # target = solver_outputs.data.argmax(dim=1)
            logits_gen_imgs_for_old_classes = pairwise_cosine_similarity(solver_features, self.prototypes)
            softmaxs_gen_imgs_for_old_classes = F.softmax(logits_gen_imgs_for_old_classes, dim=0)
            gloss_for_SC_old = self.criterion_SoftTarget_CE(softmaxs_gen_imgs_for_old_classes, seletected_alpha_softmax)

            _, solver_features_for_new_classes = self.solver(real_imgs, is_nograd=True)
            logits_for_real_imgs = pairwise_cosine_similarity(solver_features_for_new_classes, self.prototypes)
            softmaxs_for_real_imgs = F.softmax(logits_for_real_imgs, dim=0)
            # sample: alpha_matrix_softmax, batch_size
            if hasattr(self.GAN_generator, "module"):
                gen_imgs_for_new_classes, beta_softmax_for_new_classes = \
                    self.GAN_generator.module.sample(alpha_matrix_softmax=softmaxs_for_real_imgs, batch_size=batch_size)
            else:
                gen_imgs_for_new_classes, beta_softmax_for_new_classes = \
                    self.GAN_generator.sample(alpha_matrix_softmax=softmaxs_for_real_imgs, batch_size=batch_size)
            _, solver_features_for_gen_imgs_new_classes = self.solver(gen_imgs_for_new_classes)
            logits_gen_imgs_for_new_classes = pairwise_cosine_similarity(solver_features_for_gen_imgs_new_classes,
                                                                         self.prototypes)

            softmaxs_gen_imgs_for_new_classes = F.softmax(logits_gen_imgs_for_new_classes, dim=0)
            gloss_for_SC_new = self.criterion_SoftTarget_CE(softmaxs_gen_imgs_for_new_classes,
                                                                  beta_softmax_for_new_classes)
            gloss = gloss_for_wgan + gloss_for_SC_old + gloss_for_SC_new
            gloss.backward()
            generator_optimizer.step()
            loss_dict = {
                    "dis_loss": d_loss.data.item(),
                    "dis_real_loss": dis_real_loss.data.item(),
                    "dis_fake_loss": dis_fake_loss.data.item(),
                    "gloss_for_wgan": gloss_for_wgan.data.item(),
                    "gloss_for_SC_old": gloss_for_SC_old.data.item(),
                    "gloss_for_SC_new": gloss_for_SC_new.data.item(),
                    "gloss": gloss.data.item(),
            }
            if (current_iter + 1) % miniters == 0:
                # print(f"solver_features_for_new_classes: {solver_features_for_new_classes}")
                # print(f"loss_dict: {loss_dict}")
                pbar.set_postfix({k: f"{v:.4f}" for k, v in loss_dict.items()})
        torch.cuda.empty_cache()

    # def train_step(self, real_images, generator_optimizer, discriminator_optimizer):
    #     # for critic_index in range(self.critic_iters):
    #     batch_size = real_images.size(0)
    #     real_labels = torch.ones(batch_size).to(self.device)
    #     dis_real_preds = self.GAN_discriminator(real_images)
    #     dis_real_outputs = dis_real_preds.reshape(-1)
    #     dloss_real = self.criterion_bce(dis_real_outputs, real_labels)
    #     # dmean_real = dis_real_outputs.sigmoid().mean()
    #     # sample: alpha_matrix_softmax, batch_size
    #     if hasattr(self.GAN_generator, "module"):
    #         gen_imgs_for_old_classes, seletected_alpha_softmax = \
    #             self.GAN_generator.module.sample(alpha_matrix_softmax=self.alpha_matrix_softmax, batch_size=batch_size)
    #     else:
    #         gen_imgs_for_old_classes, seletected_alpha_softmax = \
    #             self.GAN_generator.sample(alpha_matrix_softmax=self.alpha_matrix_softmax, batch_size=batch_size)
    #
    #     gen_labels = torch.zeros(batch_size).to(self.device)
    #     fake_imgs_for_old_classes = gen_imgs_for_old_classes.detach()
    #     dis_fake_preds = self.GAN_discriminator(fake_imgs_for_old_classes)
    #     dis_fake_outputs = dis_fake_preds.view(-1)
    #     dloss_fake = self.criterion_bce(dis_fake_outputs, gen_labels)
    #     # dmean_fake = dis_fake_outputs.sigmoid().mean()
    #
    #     dloss = (dloss_real + dloss_fake) / 2
    #
    #     self.GAN_discriminator.zero_grad()
    #     discriminator_optimizer.zero_grad()
    #     dloss.backward()
    #     discriminator_optimizer.step()
    #
    #     gen_real_labels = torch.ones(batch_size).to(self.device)
    #     dis_gen_real_preds = self.GAN_discriminator(gen_imgs_for_old_classes)
    #     dis_gen_real_outputs = dis_gen_real_preds.view(-1)
    #     gloss_for_gan = self.criterion_bce(dis_gen_real_outputs, gen_real_labels)
    #     # gmean_fake = dis_gen_real_outputs.sigmoid().mean()
    #
    #     solver_outputs, solver_features = self.solver(gen_imgs_for_old_classes)
    #     # target = solver_outputs.data.argmax(dim=1)
    #     logits_gen_imgs_for_old_classes = pairwise_cosine_similarity(solver_features, self.prototypes)
    #     softmaxs_gen_imgs_for_old_classes = F.softmax(logits_gen_imgs_for_old_classes, dim=0)
    #     gloss_for_SC_old = self.criterion_SoftTarget_CE(softmaxs_gen_imgs_for_old_classes, seletected_alpha_softmax)
    #
    #     _, solver_features_for_new_classes = self.solver(real_images, is_nograd=True)
    #     logits_for_real_imgs = pairwise_cosine_similarity(solver_features_for_new_classes, self.prototypes)
    #     softmaxs_for_real_imgs = F.softmax(logits_for_real_imgs, dim=0)
    #     # sample: alpha_matrix_softmax, batch_size
    #
    #     if hasattr(self.GAN_generator, "module"):
    #         gen_imgs_for_new_classes, beta_softmax_for_new_classes = \
    #             self.GAN_generator.module.sample(alpha_matrix_softmax=softmaxs_for_real_imgs, batch_size=batch_size)
    #     else:
    #         gen_imgs_for_new_classes, beta_softmax_for_new_classes = \
    #             self.GAN_generator.sample(alpha_matrix_softmax=softmaxs_for_real_imgs, batch_size=batch_size)
    #     _, solver_features_for_gen_imgs_new_classes = self.solver(gen_imgs_for_new_classes)
    #     logits_gen_imgs_for_new_classes = pairwise_cosine_similarity(solver_features_for_gen_imgs_new_classes,
    #                                                                  self.prototypes)
    #
    #     softmaxs_gen_imgs_for_new_classes = F.softmax(logits_gen_imgs_for_new_classes, dim=0)
    #     gloss_for_SC_new = self.criterion_SoftTarget_CE(softmaxs_gen_imgs_for_new_classes, beta_softmax_for_new_classes)
    #     gloss = gloss_for_gan + gloss_for_SC_old + gloss_for_SC_new
    #     self.GAN_generator.zero_grad()
    #     generator_optimizer.zero_grad()
    #     gloss.backward()
    #     generator_optimizer.step()
    #     loss_dict = {
    #         "dis_loss": dloss.data.item(),
    #         "gloss_for_gan": gloss_for_gan.data.item(),
    #         "gloss_for_SC_old": gloss_for_SC_old.data.item(),
    #         "gloss_for_SC_new": gloss_for_SC_new.data.item(),
    #         "gloss": gloss.data.item(),
    #     }
    #     return loss_dict
    #     # if batch_idx % 100 == 0:
    #     #     print('Epoch %d: [%d/%d]: D_loss: %.3f, G_loss: %.3f' % (epoch, batch_idx, iter_num, D_loss_ave.val,
    #     #                                                              G_loss_ave.val))
    #     #     print(f'Discriminator tells real images real ability: {dmean_real}', '\n',
    #     #           f'Discriminator tells fake images real ability: {dmean_fake:g}/{gmean_fake:g}')
    #
    # # if self.cfg.VALID_STEP != -1 and epoch % self.cfg.VALID_STEP == 0:
    # #     print('Epoch: [%d/%d]: D_loss: %.3f, G_loss: %.3f' % ((epoch), self.max_epochs, D_loss_ave.val,
    # #                                                           G_loss_ave.val))
    #     pass
