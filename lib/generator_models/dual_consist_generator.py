import torch
import torch.nn as nn
import numpy as np

"""
@article{fang2019datafree,
    title={Data-Free Adversarial Distillation},	
    author={Gongfan Fang and Jie Song and Chengchao Shen and Xinchao Wang and Da Chen and Mingli Song},	  
    journal={arXiv preprint arXiv:1912.11006},	
    year={2019}
}
"""

__all__ = [
    "dual_consist_CIFAR_GEN",
    "dual_consist_TINYIMNET_GEN",
    "dual_consist_IMNET_GEN",
    "dual_consist_CIFAR_DIS",
]


class dual_consist_GeneratorTiny(nn.Module):
    # zdim: length of Noise, in_channel: channel of image, img_size: width == height
    def __init__(self, prototypes, zdim, in_channel, img_sz):
        super(dual_consist_GeneratorTiny, self).__init__()
        self.noize_z_dim = zdim - prototypes.size(1)
        self.embeddings_for_old_classes = nn.Parameter(prototypes)

        self.init_size = img_sz // 4
        self.l1 = nn.Sequential(nn.Linear(zdim, 128 * self.init_size ** 2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(128),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, in_channel, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(in_channel, affine=False)
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks2(img)
        return img

    def update_embeddings(self, prototypes):
        self.embeddings_for_old_classes = None
        self.embeddings_for_old_classes = nn.Parameter(prototypes)
        pass

    def sample(self, alpha_matrix_softmax, batch_size):
        # sample z
        assert alpha_matrix_softmax.size(1) == self.embeddings_for_old_classes.size(0)
        if alpha_matrix_softmax.size(0) != batch_size:
            seletected_embeddings_index = np.random.choice(alpha_matrix_softmax.size(0), size=batch_size)
            seletected_alpha_softmax = alpha_matrix_softmax[seletected_embeddings_index]
        else:
            # seletected_embeddings_index = np.arange(batch_size)
            seletected_alpha_softmax = alpha_matrix_softmax
        constructed_embeddings = self.embeddings_for_old_classes * seletected_alpha_softmax.unsqueeze(dim=-1)
        constructed_embeddings = torch.sum(constructed_embeddings, dim=1)
        z = torch.randn(batch_size, self.noize_z_dim)
        z = z.cuda()
        inputs = torch.cat([constructed_embeddings, z], dim=1)
        X = self.forward(inputs)
        # Y = torch.from_numpy(seletected_embeddings_index).cuda()
        return X, seletected_alpha_softmax


class dual_consist_GeneratorMed(nn.Module):
    def __init__(self, prototypes, zdim, in_channel, img_sz):
        super(dual_consist_GeneratorMed, self).__init__()
        self.noize_z_dim = zdim - prototypes.size(1)
        self.embeddings_for_old_classes = nn.Parameter(prototypes)

        self.init_size = img_sz // 8
        self.l1 = nn.Sequential(nn.Linear(zdim, 128 * self.init_size ** 2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(128),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks3 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, in_channel, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(in_channel, affine=False)
        )

    def update_embeddings(self, prototypes):
        self.embeddings_for_old_classes = None
        self.embeddings_for_old_classes = nn.Parameter(prototypes)
        pass

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks2(img)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks3(img)
        return img

    def sample(self, alpha_matrix_softmax, batch_size):
        assert alpha_matrix_softmax.size(1) == self.embeddings_for_old_classes.size(0)
        if alpha_matrix_softmax.size(0) != batch_size:
            seletected_embeddings_index = np.random.choice(alpha_matrix_softmax.size(0), size=batch_size)
            seletected_alpha_softmax = alpha_matrix_softmax[seletected_embeddings_index]
        else:
            # seletected_embeddings_index = np.arange(batch_size)
            seletected_alpha_softmax = alpha_matrix_softmax
        constructed_embeddings = self.embeddings_for_old_classes * seletected_alpha_softmax.unsqueeze(dim=-1)
        constructed_embeddings = torch.sum(constructed_embeddings, dim=1)
        z = torch.randn(batch_size, self.noize_z_dim)
        z = z.cuda()
        inputs = torch.cat([constructed_embeddings, z], dim=1)
        X = self.forward(inputs)
        # Y = torch.from_numpy(seletected_embeddings_index).cuda()
        return X, seletected_alpha_softmax


class dual_consist_GeneratorBig(nn.Module):
    def __init__(self, prototypes, zdim, in_channel, img_sz):
        super(dual_consist_GeneratorBig, self).__init__()
        self.noize_z_dim = zdim - prototypes.size(1)
        self.embeddings_for_old_classes = nn.Parameter(prototypes)

        self.init_size = img_sz // (2 ** 5)
        self.l1 = nn.Sequential(nn.Linear(zdim, 64 * self.init_size ** 2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(64),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks5 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks6 = nn.Sequential(
            nn.Conv2d(64, in_channel, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(in_channel, affine=False)
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 64, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks2(img)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks3(img)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks4(img)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks5(img)
        img = self.conv_blocks6(img)
        return img

    def update_embeddings(self, prototypes):
        self.embeddings_for_old_classes = None
        self.embeddings_for_old_classes = nn.Parameter(prototypes)
        pass

    def sample(self, alpha_matrix_softmax, batch_size):
        # sample z
        assert alpha_matrix_softmax.size(1) == self.embeddings_for_old_classes.size(0)
        if alpha_matrix_softmax.size(0) != batch_size:
            seletected_embeddings_index = np.random.choice(alpha_matrix_softmax.size(0), size=batch_size)
            seletected_alpha_softmax = alpha_matrix_softmax[seletected_embeddings_index]
        else:
            # seletected_embeddings_index = np.arange(batch_size)
            seletected_alpha_softmax = alpha_matrix_softmax
        constructed_embeddings = self.embeddings_for_old_classes * seletected_alpha_softmax.unsqueeze(dim=-1)
        constructed_embeddings = torch.sum(constructed_embeddings, dim=1)
        z = torch.randn(batch_size, self.noize_z_dim)
        z = z.cuda()
        inputs = torch.cat([constructed_embeddings, z], dim=1)
        X = self.forward(inputs)
        # Y = torch.from_numpy(seletected_embeddings_index).cuda()
        return X, seletected_alpha_softmax


class dual_consist_DiscriminatorTiny(nn.Module):
    def __init__(self, n_channel=3, n_d_feature=64):
        super(dual_consist_DiscriminatorTiny, self).__init__()
        self.dnet = nn.Sequential(
            nn.Conv2d(n_channel, n_d_feature, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(n_d_feature),
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


class dual_consist_DiscriminatorMed(nn.Module):
    def __init__(self, n_channel=3, n_d_feature=64):
        super(dual_consist_DiscriminatorMed, self).__init__()
        self.dnet = nn.Sequential(
            nn.Conv2d(n_channel, n_d_feature, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(n_d_feature),
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


class dual_consist_DiscriminatorBig(nn.Module):
    def __init__(self, n_channel=3, n_d_feature=64):
        super(dual_consist_DiscriminatorBig, self).__init__()
        self.dnet = nn.Sequential(
            nn.Conv2d(n_channel, n_d_feature, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(n_d_feature),
            nn.LeakyReLU(0.2),

            nn.Conv2d(n_d_feature, 2 * n_d_feature, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2 * n_d_feature),
            nn.LeakyReLU(0.2),

            nn.Conv2d(2 * n_d_feature, 4 * n_d_feature, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(4 * n_d_feature),
            nn.LeakyReLU(0.2),

            nn.Conv2d(4 * n_d_feature, 8 * n_d_feature, kernel_size=4),
            nn.BatchNorm2d(8 * n_d_feature),
            nn.LeakyReLU(0.2),

            nn.Conv2d(8 * n_d_feature, 1, kernel_size=4)
        )

    def forward(self, imgs):
        output = self.dnet(imgs)
        return output


def dual_consist_CIFAR_GEN(prototypes=None, bn=False):
    return dual_consist_GeneratorTiny(prototypes=prototypes, zdim=1000, in_channel=3, img_sz=32)


def dual_consist_CIFAR_DIS(bn=False):
    return dual_consist_DiscriminatorTiny()


def dual_consist_TINYIMNET_GEN(prototypes=None, bn=False):
    return dual_consist_GeneratorMed(prototypes=prototypes, zdim=1000, in_channel=3, img_sz=64)


def dual_consist_TINYIMNET_DIS(bn=False):
    return dual_consist_DiscriminatorMed()


def dual_consist_IMNET_GEN(prototypes=None, bn=False):
    return dual_consist_GeneratorBig(prototypes=prototypes, zdim=1000, in_channel=3, img_sz=224)


def dual_consist_IMNET_DIS(bn=False):
    return dual_consist_DiscriminatorBig()
