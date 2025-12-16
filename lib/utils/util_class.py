import os.path

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision

from lib.data_transform.data_transform import imagenet_normalize, AVAILABLE_TRANSFORMS


class FCTM_GradCAM:
    def __init__(self, dataset_name, old_model, FCTM, target_layer="FCN", stored_path="./", size=(224, 224)):
        self.dataset_name = dataset_name
        self.old_model = old_model
        self.old_model.eval()
        self.FCTM = FCTM
        self.FCTM.eval()
        getattr(self.FCTM, target_layer).register_forward_hook(self.__forward_hook)
        getattr(self.FCTM, target_layer).register_full_backward_hook(self.__backward_hook)
        self.size = size
        self.stored_path = stored_path
        # self.ann = {}
        # with open(r'D:\jupyter\imagenet_ann.txt', 'r') as f:
        #     file = f.readlines()
        #     for line in file:
        #         key, _, value = line.split(' ', 2)
        #         self.ann[key] = value

    def forward(self, class_name, path, write=True):
        # 读取图片
        print("img_path:", path)
        origin_img = cv2.imread(path)
        origin_size = (origin_img.shape[1], origin_img.shape[0])  # [H, W, C]
        transform = transforms.Compose([
            transforms.ToPILImage(),
            *AVAILABLE_TRANSFORMS[self.dataset_name]
            ['test_transform']])
        img = transform(origin_img[:, :, ::-1]).unsqueeze(0)

        # 输入模型以获取特征图和梯度
        # output = self.model(img)
        _, features = self.old_model(img)
        output = self.FCTM(img, features)
        self.old_model.zero_grad()
        self.FCTM.zero_grad()
        loss, index = torch.max(output["all_logits"], dim=1)  # 这个output的下标就是模型预测的label
        print('预测label为：', index.item())
        # print('对应类别为：', self.ann[str(index.item())])
        loss.backward()
        # 计算cam图片
        cam = np.zeros(self.fmaps.shape[1:], dtype=np.float32)
        alpha = np.mean(self.grads, axis=(1, 2))
        for k, ak in enumerate(alpha):
            cam += ak * self.fmaps[k]  # linear combination
        cam[cam < 0] = 0
        cam = cv2.resize(np.array(cam), origin_size)
        cam /= np.max(cam)

        # 把cam图变成热力图，再与原图相加
        cam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        cam = np.float32(cam) + np.float32(origin_img)
        # 两幅图的色彩相加后范围要回归到（0，1）之间,再乘以255
        cam = np.uint8(255 * cam / np.max(cam))
        if write:
            img_stored_path = os.path.join(self.stored_path, class_name)
            if not os.path.exists(img_stored_path):
                os.makedirs(img_stored_path)
            img_name = path.split("/")[-1]
            img_stored_img_file = os.path.join(img_stored_path, img_name)
            cv2.imwrite(img_stored_img_file, cam)
        # if show:
        #     # 要显示RGB的图片，如果是BGR的 热力图是反过来的
        #     plt.imshow(cam[:, :, ::-1])
        #     plt.show()

    def __backward_hook(self, module, output, grad_out):
        self.grads = np.array(grad_out[0].detach().squeeze())

    def __forward_hook(self, module, input, output):
        self.fmaps = np.array(output.detach().squeeze())


class GradCAM:
    def __init__(self, dataset_name, old_model, target_layer="FCN", pre_model=None, stored_path="./", size=(224, 224)):
        self.dataset_name = dataset_name
        self.old_model = old_model
        self.pre_model = pre_model
        self.old_model.eval()
        getattr(self.old_model.extractor[0], target_layer).register_forward_hook(self.__forward_hook)
        getattr(self.old_model.extractor[0], target_layer).register_full_backward_hook(self.__backward_hook)
        # getattr(self.old_model.fctm.extractor[0], target_layer).register_forward_hook(self.__forward_hook)
        # getattr(self.old_model.fctm.extractor[0], target_layer).register_full_backward_hook(self.__backward_hook)
        # getattr(self.old_model.FCN.hidden_fc_layers, target_layer).register_forward_hook(self.__forward_hook)
        # getattr(self.old_model.FCN.hidden_fc_layers, target_layer).register_full_backward_hook(self.__backward_hook)
        # getattr(self.old_model, target_layer).register_forward_hook(self.__forward_hook)
        # getattr(self.old_model, target_layer).register_full_backward_hook(self.__backward_hook)
        self.size = size
        self.stored_path = stored_path
        # self.ann = {}
        # with open(r'D:\jupyter\imagenet_ann.txt', 'r') as f:
        #     file = f.readlines()
        #     for line in file:
        #         key, _, value = line.split(' ', 2)
        #         self.ann[key] = value

    def forward(self, class_name, path, write=True):
        # 读取图片
        print("img_path:", path)
        origin_img = cv2.imread(path)
        origin_size = (origin_img.shape[1], origin_img.shape[0])  # [H, W, C]
        transform = transforms.Compose([
            transforms.ToPILImage(),
            *AVAILABLE_TRANSFORMS[self.dataset_name]
            ['test_transform']])
        img = transform(origin_img[:, :, ::-1]).unsqueeze(0)
        # pre_model_feature = self.pre_model(img, is_nograd=True, feature_flag=True)
        # 输入模型以获取特征图和梯度
        # output = self.model(img)
        output, features = self.old_model(img)
        # outputs = self.old_model(img, pre_model_feature)
        self.old_model.zero_grad()
        # loss, index = torch.max(outputs["all_logits"], dim=1)  # 这个output的下标就是模型预测的label
        loss, index = torch.max(output, dim=1)  # 这个output的下标就是模型预测的label
        print('预测label为：', index.item())
        # print('对应类别为：', self.ann[str(index.item())])
        loss.backward()
        # 计算cam图片
        cam = np.zeros(self.fmaps.shape[1:], dtype=np.float32)
        alpha = np.mean(self.grads, axis=(1, 2))
        for k, ak in enumerate(alpha):
            cam += ak * self.fmaps[k]  # linear combination
        cam[cam < 0] = 0
        cam = cv2.resize(np.array(cam), origin_size)
        cam /= np.max(cam)

        # 把cam图变成热力图，再与原图相加
        cam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        cam = np.float32(cam) + np.float32(origin_img)
        # 两幅图的色彩相加后范围要回归到（0，1）之间,再乘以255
        cam = np.uint8(255 * cam / np.max(cam))
        if write:
            img_stored_path = os.path.join(self.stored_path, class_name)
            if not os.path.exists(img_stored_path):
                os.makedirs(img_stored_path)
            img_name = path.split("/")[-1]
            img_stored_img_file = os.path.join(img_stored_path, img_name)
            cv2.imwrite(img_stored_img_file, cam)
        # if show:
        #     # 要显示RGB的图片，如果是BGR的 热力图是反过来的
        #     plt.imshow(cam[:, :, ::-1])
        #     plt.show()

    def __backward_hook(self, module, grad_in, grad_out):
        self.grads = np.array(grad_out[0].detach().squeeze())
        print("shape:", self.grads.shape)

    def __forward_hook(self, module, input, output):
        self.fmaps = np.array(output.detach().squeeze())
