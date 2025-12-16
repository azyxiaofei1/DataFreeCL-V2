import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

__all__ = ['ResNet', 'resnet18_cbam', 'resnet34_cbam', 'resnet50_cbam', 'resnet101_cbam',
           'resnet152_cbam']

from lib.backbone.resnet_cifar import _weights_init

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

# class BasicBlock(nn.Module):
#     """Basic Block for resnet 18 and resnet 34
#     """
#     expansion = 1
#
#     def __init__(self, in_channels, out_channels, stride=1, last_relu=True):
#         super(BasicBlock, self).__init__()
#         self.last_relu = last_relu
#         self.residual_branch = nn.Sequential(
#             nn.Conv2d(in_channels,
#                       out_channels,
#                       kernel_size=3,
#                       stride=stride,
#                       padding=1,
#                       bias=False), nn.BatchNorm2d(out_channels),
#             nn.ReLU(),
#             nn.Conv2d(out_channels,
#                       out_channels * BasicBlock.expansion,
#                       kernel_size=3,
#                       padding=1,
#                       bias=False),
#             nn.BatchNorm2d(out_channels * BasicBlock.expansion))
#
#         self.shortcut = nn.Sequential()
#
#         if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels,
#                           out_channels * BasicBlock.expansion,
#                           kernel_size=1,
#                           stride=stride,
#                           bias=False),
#                 nn.BatchNorm2d(out_channels * BasicBlock.expansion))
#
#     def forward(self, x):
#         if self.last_relu:
#             return F.relu(self.residual_branch(x) + self.shortcut(x))
#         else:
#             return self.residual_branch(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.ca = ChannelAttention(planes * 4)
        self.sa = SpatialAttention()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


# class ResNet(nn.Module):
#     def __init__(self, block, layers, rate=1, inter_layer=False, **kwars):
#         super(ResNet, self).__init__()
#         self.inter_layer = inter_layer
#         self.in_channels = int(64 * rate)
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(3, int(64 * rate), kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(int(64 * rate)), nn.ReLU(inplace=False))
#
#         self.layer1 = self._make_layer(block, int(64 * rate), layers[0], 1)
#         self.layer2 = self._make_layer(block, int(128 * rate), layers[1], 2)
#         self.layer3 = self._make_layer(block, int(256 * rate), layers[2], 2)
#         self.layer4 = self._make_layer(block, int(512 * rate), layers[3], 2)
#         self.apply(_weights_init)
#         self.out_dim = 512 * block.expansion
#
#     def _make_layer(self, block, out_channels, num_blocks, stride):
#         """make resnet layers(by layer i didnt mean this 'layer' was the
#         same as a neuron netowork layer, ex. conv layer), one layer may
#         contain more than one residual block
#         Args:
#             block: block type, basic block or bottle neck block
#             out_channels: output depth channel number of this layer
#             num_blocks: how many blocks per layer
#             stride: the stride of the first block of this layer
#
#         Return:
#             return a resnet layer
#         """
#
#         strides = [stride] + [1] * (num_blocks - 1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_channels, out_channels, stride))
#             self.in_channels = out_channels * block.expansion
#
#         return nn.Sequential(*layers)
#
#     def load_model(self, pretrain):
#         print("Loading Backbone pretrain model from {}......".format(pretrain))
#         model_dict = self.state_dict()
#         pretrain_dict = torch.load(pretrain)
#         pretrain_dict = pretrain_dict["state_dict"] if "state_dict" in pretrain_dict else pretrain_dict
#         from collections import OrderedDict
#
#         new_dict = OrderedDict()
#         for k, v in pretrain_dict.items():
#             if k.startswith("module"):
#                 k = k[7:]
#             if "last_linear" not in k and "classifier" not in k and "linear" not in k and "fd" not in k:
#                 k = k.replace("backbone.", "")
#                 k = k.replace("fr", "layer3.4")
#                 new_dict[k] = v
#         model_dict.update(new_dict)
#         self.load_state_dict(model_dict)
#         print("Backbone model has been loaded......")
#
#     # def forward(self, x, **kwargs):
#     #     x = self.conv1(x)
#     #     x = self.stage2(x)
#     #     x = self.stage3(x)
#     #     x = self.stage4(x)
#     #     x = self.stage5(x)
#     #     return x
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         dim = x.size()[-1]
#         pool = nn.AvgPool2d(dim, stride=1)
#         x = pool(x)
#
#         x = x.view(x.size(0), -1)
#         return {"features": x}
#
#     def freeze(self):
#         for name, param in self.named_parameters():
#             # print(name, "->", param.requires_grad)
#             param.requires_grad = False
#             # print(" || requires_grad = False", name, "->", param.requires_grad)
#         pass
#
#     def unfreeze_module(self, module_name=None):
#         if module_name is None:
#             module_name = ["layer4", "stage5", "fc"]
#         # if hasattr(self.extractor, "module"):
#         #     self.extractor.module.unfreeze(module_name)
#         # else:
#         #     self.extractor.unfreeze(module_name)
#         for name, param in self.named_parameters():
#             for module_item in module_name:
#                 if module_item in name:
#                     param.requires_grad = True
#             # print(name, "->", param.requires_grad)
#         pass

class ResNet(nn.Module):

    def __init__(self, block, layers, dataset_name=None):
        self.inplanes = 64
        super(ResNet, self).__init__()

        assert dataset_name is not None, "you should pass args to resnet"
        if 'cifar' in dataset_name.lower():
            self.conv1 = nn.Sequential(nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(self.inplanes), nn.ReLU(inplace=True))
            print("Success")

        elif 'imagenet' in dataset_name.lower():

            self.conv1 = nn.Sequential(
                nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(self.inplanes),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.feature = nn.AvgPool2d(4, stride=1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.out_dim = 512 * block.expansion

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        dim = x.size()[-1]
        pool = nn.AvgPool2d(dim, stride=1)
        x = pool(x)

        x = x.view(x.size(0), -1)
        return {"features": x}

class ResNet_imagenet(nn.Module):

    def __init__(self, block, layers, num_classes=100, args=None):
        self.inplanes = 64
        super(ResNet, self).__init__()
        assert args is not None, "you should pass args to resnet"
        if 'cifar' in args["dataset"]:
            self.conv1 = nn.Sequential(nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(self.inplanes), nn.ReLU(inplace=True))
        elif 'imagenet' in args["dataset"]:
            if args["init_cls"] == args["increment"]:
                self.conv1 = nn.Sequential(
                    nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
                    nn.BatchNorm2d(self.inplanes),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                )
            else:
                self.conv1 = nn.Sequential(
                    nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(self.inplanes),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                )
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.feature = nn.AvgPool2d(4, stride=1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.out_dim = 512 * block.expansion

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        dim = x.size()[-1]
        pool = nn.AvgPool2d(dim, stride=1)
        x = pool(x)
        x = x.view(x.size(0), -1)
        return {"features": x}


def resnet18_cbam(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet18'])
        now_state_dict        = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model


def resnet34_cbam(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet34'])
        now_state_dict        = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model


def resnet50_cbam(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet50'])
        now_state_dict        = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model


def resnet101_cbam(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet101'])
        now_state_dict        = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model


def resnet152_cbam(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet152'])
        now_state_dict        = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model