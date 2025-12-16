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
    "CIFAR_cGAN",
    "CIFAR_cDIS",
    "CIFAR_cGAN_conv",
    "CIFAR_cDIS_conv"
]


class Generator_conv(nn.Module):
    def __init__(self, class_num, latent_size):
        super(Generator_conv, self).__init__()
        self._latent_size = latent_size
        self._class_num = class_num

        # LABEL
        # Embedding & turn to 1x64
        self.label_embedding = nn.Sequential(
            nn.Embedding(class_num, latent_size),
            nn.Linear(latent_size, 64)
        )

        # LATENT
        # 1x100 to 8x8x128
        self.latent = nn.Sequential(
            nn.Linear(latent_size, 8 * 8 * 128),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # MODEL
        self.model = nn.Sequential(
            # 8x8x129 to 16x16x64
            nn.ConvTranspose2d(129, 64, kernel_size=(4, 4), stride=(2, 2), padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=0.1, eps=0.8),
            nn.ReLU(True),

            # 16x16x64 to 32x32x32
            nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=1, bias=False),
            nn.BatchNorm2d(32, momentum=0.1, eps=0.8),
            nn.ReLU(True),

            # 32x32x32 to 32x32x3
            nn.ConvTranspose2d(32, 3, kernel_size=(5, 5), stride=(1, 1), padding=2, bias=False),
            nn.Tanh()
        )

    def forward(self, inputs):
        latent_vector, label = inputs
        # LABEL
        label_output = self.label_embedding(label)  # 1x(8x8)
        label_output = label_output.view(-1, 1, 8, 8)

        # LATENT
        latent_output = self.latent(latent_vector)
        latent_output = latent_output.view(-1, 128, 8, 8)

        concat = torch.cat((latent_output, label_output), dim=1)

        image = self.model(concat)
        return image

    def update_class_num(self, class_num):
        self._class_num = class_num
        device = torch.device("cuda:0")
        label_embedding = nn.Sequential(
            nn.Embedding(class_num, self._latent_size),
            nn.Linear(self._latent_size, 64)
        )
        self.label_embedding = label_embedding.to(device)

    def sample(self, batch_size, determined_labels=None, is_nograd=True):
        device = torch.device("cuda:0")
        # noise_vectors = torch.randn(batch_size, self._latent_size, device=device)
        # noise_vectors = noise_vectors.to(device=device)

        noise_vectors = torch.FloatTensor(np.random.normal(0, 1, (batch_size, self._latent_size)))
        noise_vectors = noise_vectors.to(device=device)

        if determined_labels is not None:
            y = determined_labels
        else:
            y = torch.tensor([random.randint(0, self._class_num - 1) for i in range(batch_size)]).to(device)
        labels = y.unsqueeze(1).long()
        inputs = (noise_vectors, labels)
        if is_nograd:
            with torch.no_grad():
                images = self.forward(inputs)
        else:
            images = self.forward(inputs)
        return images, y


class Discriminator_conv(nn.Module):
    def __init__(self, class_num):
        super(Discriminator_conv, self).__init__()

        self.label_embedding = nn.Sequential(
            nn.Embedding(class_num, 50),
            nn.Linear(50, 3 * 32 * 32)
        )

        self.model = nn.Sequential(
            # 32x32x6 to 16x16x64
            nn.Conv2d(6, 64, kernel_size=(4, 4), stride=(2, 2), padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # 16x16x64 to 8x8x128
            nn.Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=1, bias=False),
            nn.BatchNorm2d(128, momentum=0.1, eps=0.8),
            nn.LeakyReLU(0.2, inplace=True),

            # Output
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(8192, 1),
            nn.Sigmoid()
        )

    def update_class_num(self, class_num):
        device = torch.device("cuda:0")
        label_embedding = nn.Sequential(
            nn.Embedding(class_num, 50),
            nn.Linear(50, 3 * 32 * 32)
        )
        self.label_embedding = label_embedding.to(device)

    def forward(self, inputs):
        image, label = inputs
        # LABEL
        label_output = self.label_embedding(label)
        label_output = label_output.view(-1, 3, 32, 32)

        concat = torch.cat((image, label_output), dim=1)

        output = self.model(concat)
        return output


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


class Generator(nn.Module):
    def __init__(self, class_num, latent_size, img_shape=(3, 32, 32)):
        super(Generator, self).__init__()
        self._latent_size = latent_size
        self._class_num = class_num
        self.img_shape = img_shape
        self.label_embed = nn.Embedding(class_num, class_num)
        self.depth = 128

        self.generator = nn.Sequential(
            *self.init(self._latent_size + self._class_num, self.depth),
            *self.init(self.depth, self.depth * 2),
            *self.init(self.depth * 2, self.depth * 4),
            *self.init(self.depth * 4, self.depth * 8),
            nn.Linear(self.depth * 8, int(np.prod(img_shape))),
            nn.Tanh()

        )

    def init(self, input, output, normalize=True):
        layers = [nn.Linear(input, output)]
        if normalize:
            layers.append(nn.BatchNorm1d(output, 0.8))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return layers

    # torchcat needs to combine tensors
    def forward(self, noise, labels):
        gen_input = torch.cat((self.label_embed(labels), noise), -1)
        img = self.generator(gen_input)
        img = img.view(img.size(0), *self.img_shape)
        return img

    def update_class_num(self, class_num):
        self._class_num = class_num
        device = torch.device("cuda:0")
        label_embed = nn.Embedding(class_num, class_num)
        self.label_embed = label_embed.to(device)
        generator = nn.Sequential(

            *self.init(self._latent_size + self._class_num, self.depth),
            *self.init(self.depth, self.depth * 2),
            *self.init(self.depth * 2, self.depth * 4),
            *self.init(self.depth * 4, self.depth * 8),
            nn.Linear(self.depth * 8, int(np.prod(self.img_shape))),
            nn.Tanh()

        )
        self.generator = generator.to(device)

    def sample(self, batch_size, determined_labels=None, is_nograd=True):
        device = torch.device("cuda:0")
        noise_vectors = torch.FloatTensor(np.random.normal(0, 1, (batch_size, self._latent_size)))
        noise_vectors = noise_vectors.to(device=device)
        if determined_labels is not None:
            y = determined_labels
        else:
            y = torch.tensor([random.randint(0, self._class_num - 1) for i in range(batch_size)]).to(device)
        labels = y.long()
        if is_nograd:
            with torch.no_grad():
                images = self.forward(noise_vectors, labels)
        else:
            images = self.forward(noise_vectors, labels)
        return images, y


class Discriminator(nn.Module):
    def __init__(self, class_num, img_shape=(3, 32, 32)):
        super(Discriminator, self).__init__()
        self.label_embed1 = nn.Embedding(class_num, class_num)
        self.dropout = 0.4
        self.depth = 512
        self.img_shape = img_shape

        self.discriminator = nn.Sequential(
            *self.init(class_num + int(np.prod(self.img_shape)), self.depth, normalize=False),
            *self.init(self.depth, self.depth),
            *self.init(self.depth, self.depth),
            nn.Linear(self.depth, 1),
            nn.Sigmoid()
        )

    def init(self, input, output, normalize=True):
        layers = [nn.Linear(input, output)]
        if normalize:
            layers.append(nn.Dropout(self.dropout))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return layers

    def forward(self, img, labels):
        imgs = img.view(img.size(0), -1)
        inpu = torch.cat((imgs, self.label_embed1(labels)), -1)
        validity = self.discriminator(inpu)
        return validity

    def update_class_num(self, class_num):
        device = torch.device("cuda:0")
        label_embed1 = nn.Embedding(class_num, class_num)
        self.label_embed1 = label_embed1.to(device)
        discriminator = nn.Sequential(
            *self.init(class_num + int(np.prod(self.img_shape)), self.depth, normalize=False),
            *self.init(self.depth, self.depth),
            *self.init(self.depth, self.depth),
            nn.Linear(self.depth, 1),
            nn.Sigmoid()
        )
        self.discriminator = discriminator.to(device)


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


def CIFAR_cGAN(class_num, latent_size):
    generator = Generator(class_num, latent_size)
    # generator.apply(weights_init)
    return generator


def CIFAR_cDIS(class_num):
    discriminator = Discriminator(class_num)
    # discriminator.apply(weights_init)
    return discriminator


def CIFAR_cGAN_conv(class_num, latent_size):
    generator = Generator_conv(class_num, latent_size)
    generator.apply(weights_init)
    return generator


def CIFAR_cDIS_conv(class_num):
    discriminator = Discriminator_conv(class_num)
    discriminator.apply(weights_init)
    return discriminator
