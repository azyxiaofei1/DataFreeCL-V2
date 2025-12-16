# building generator
import random

import torch
from torch import nn
import numpy as np

#
# image_size = 32
# batch_size = 100
# latent_size = 100
# class_num = 10

__all__ = [
    "CIFAR_GAN",
    "CIFAR_DIS",
]

from torch.nn import init


class Generator(nn.Module):
    def __init__(self, class_num, latent_size, n_channel=3, n_g_feature=64):
        super(Generator, self).__init__()
        self._latent_size = latent_size
        self._class_num = class_num
        self.n_g_feature = n_g_feature
        self.n_channel = n_channel
        self.gnet = nn.Sequential(
            nn.ConvTranspose2d(latent_size, 4 * self.n_g_feature, kernel_size=4, bias=False),
            nn.BatchNorm2d(4 * self.n_g_feature),
            nn.ReLU(),

            nn.ConvTranspose2d(4 * self.n_g_feature, 2 * self.n_g_feature, kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(2 * self.n_g_feature),
            nn.ReLU(),

            nn.ConvTranspose2d(2 * self.n_g_feature, self.n_g_feature, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.n_g_feature),
            nn.ReLU(),

            nn.ConvTranspose2d(self.n_g_feature, n_channel, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, noise):
        image = self.gnet(noise)
        return image

    def sample(self, batch_size, is_nograd=True):
        device = torch.device("cuda:0")
        # noise_vectors = torch.randn(batch_size, self._latent_size, device=device)
        # noise_vectors = noise_vectors.to(device=device)

        noise_vectors = torch.randn(batch_size, self._latent_size, 1, 1)
        noise_vectors = noise_vectors.to(device=device)

        if is_nograd:
            with torch.no_grad():
                images = self.forward(noise_vectors)
        else:
            images = self.forward(noise_vectors)
        return images


class Discriminator(nn.Module):
    def __init__(self, n_channel=3, n_d_feature=64):
        super(Discriminator, self).__init__()
        self.dnet = nn.Sequential(
            nn.Conv2d(n_channel, n_d_feature, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(n_d_feature, 2 * n_d_feature, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2 * n_d_feature),
            nn.LeakyReLU(0.2),

            nn.Conv2d(2 * n_d_feature, 4 * n_d_feature, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(4 * n_d_feature),
            nn.LeakyReLU(0.2),

            nn.Conv2d(4 * n_d_feature, 1, kernel_size=4)
        )

    def forward(self, imgs):
        output = self.dnet(imgs)
        return output


def weights_init(m):
    if type(m) in [nn.ConvTranspose2d, nn.Conv2d]:
        init.xavier_normal_(m.weight)
    elif type(m) == nn.BatchNorm2d:
        init.normal_(m.weight, 1.0, 0.02)
        init.constant_(m.bias, 0)


def CIFAR_GAN(class_num, latent_size):
    generator = Generator(class_num, latent_size)
    generator.apply(weights_init)
    return generator


def CIFAR_DIS():
    discriminator = Discriminator()
    discriminator.apply(weights_init)
    return discriminator
