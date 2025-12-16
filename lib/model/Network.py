import torch

import torch.nn as nn
import torch.nn.functional as F

from lib.backbone import resnet, resnet_cifar
from lib.backbone.resnet_cifar import _weights_init
from lib.modules import GAP


def cosine_similarity(a, b):
    return torch.mm(F.normalize(a, p=2, dim=-1), F.normalize(b, p=2, dim=-1).T)


def stable_cosine_distance(a, b, squared=True):
    """Computes the pairwise distance matrix with numerical stability."""
    mat = torch.cat([a, b])

    pairwise_distances_squared = torch.add(
        mat.pow(2).sum(dim=1, keepdim=True).expand(mat.size(0), -1),
        torch.t(mat).pow(2).sum(dim=0, keepdim=True).expand(mat.size(0), -1)
    ) - 2 * (torch.mm(mat, torch.t(mat)))

    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances_squared = torch.clamp(pairwise_distances_squared, min=0.0)

    # Get the mask where the zero distances are at.
    error_mask = torch.le(pairwise_distances_squared, 0.0)

    # Optionally take the sqrt.
    if squared:
        pairwise_distances = pairwise_distances_squared
    else:
        pairwise_distances = torch.sqrt(pairwise_distances_squared + error_mask.float() * 1e-16)

    # Undo conditionally adding 1e-16.
    pairwise_distances = torch.mul(pairwise_distances, (error_mask == False).float())

    # Explicitly set diagonals to zero.
    mask_offdiagonals = 1 - torch.eye(*pairwise_distances.size(), device=pairwise_distances.device)
    pairwise_distances = torch.mul(pairwise_distances, mask_offdiagonals)

    return pairwise_distances[:a.shape[0], a.shape[0]:]


class BiasLayer(nn.Module):
    def __init__(self, current_classes_num, active_classes_num):
        super(BiasLayer, self).__init__()
        self.params = nn.Parameter(torch.Tensor([1, 0]))
        self.current_classes_num = current_classes_num
        self.active_classes_num = active_classes_num

    def forward(self, x):
        x = x[:, 0: self.active_classes_num]
        x[:, -self.current_classes_num:] *= self.params[0]
        x[:, -self.current_classes_num:] += self.params[1]
        return x


class fc_relu(nn.Module):
    def __init__(self, input_dim, out_dim, bias=True):
        super(fc_relu, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.fc = torch.nn.Linear(self.input_dim, self.out_dim, bias=bias)
        self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, input):
        assert input.size(1) == self.input_dim
        features = self.fc(input)
        nor_features = self.bn(features)
        output = self.relu(nor_features)
        return output

    def load_model(self, model_path, logger=None):
        pretrain_dict = torch.load(
            model_path, map_location="cpu" if self.cfg.CPU_MODE else "cuda"
        )
        pretrain_dict = pretrain_dict['state_dict'] if 'state_dict' in pretrain_dict else pretrain_dict
        model_dict = self.state_dict()
        from collections import OrderedDict
        new_dict = OrderedDict()
        for k, v in pretrain_dict.items():
            if k.startswith("module"):
                new_dict[k[7:]] = v
            else:
                new_dict[k] = v

        model_dict.update(new_dict)
        self.load_state_dict(model_dict)
        if logger:
            logger.info("BarlowTwins Model has been loaded...")


class MLP_classifier(nn.Module):
    def __init__(self, input_feature_dim, layer_nums, output_feature_dim, hidden_layer_rate=1,
                 last_hidden_layer_use_relu=False, bias=True, all_classes=100):
        super(MLP_classifier, self).__init__()
        self.layer_nums = layer_nums
        self.input_feature_dim = input_feature_dim
        self.output_feature_dim = output_feature_dim
        self.last_hidden_layer_use_relu = last_hidden_layer_use_relu
        self.hidden_layer_rate = hidden_layer_rate
        self.hidden_fc_layers = self._make_fc(bias)
        self.cls_layer = torch.nn.Linear(self.output_feature_dim, all_classes, bias=True)

    def _make_fc(self, bias=True):
        hidden_fc_layers = []
        input_dim = self.input_feature_dim
        if self.layer_nums == 1:
            if self.last_hidden_layer_use_relu:
                hidden_fc_layers.append(
                    fc_relu(self.input_feature_dim, self.output_feature_dim, bias=bias)
                )
            else:
                hidden_fc_layers = torch.nn.Linear(self.input_feature_dim, self.output_feature_dim, bias=bias)

        else:
            for layer in range(self.layer_nums):
                if layer < self.layer_nums - 1:
                    hidden_fc_layers.append(
                        fc_relu(input_dim, int(self.hidden_layer_rate * self.input_feature_dim), bias=bias)
                    )
                    input_dim = int(self.hidden_layer_rate * self.input_feature_dim)
                else:
                    if self.last_hidden_layer_use_relu:
                        hidden_fc_layers.append(
                            fc_relu(input_dim, self.output_feature_dim, bias=bias)
                        )
                    else:
                        hidden_fc_layers.append(
                            torch.nn.Linear(input_dim, self.output_feature_dim, bias=bias)
                        )

                    input_dim = self.input_feature_dim
        return nn.Sequential(*hidden_fc_layers)

    def forward(self, din, **kwcfg):
        assert din.size(1) == self.input_feature_dim
        calibrated_features = self.hidden_fc_layers(din)
        if "feature_flag" in kwcfg:
            return calibrated_features
        else:
            return self.cls_layer(calibrated_features)


class Projector_head(nn.Module):
    def __init__(self, input_feature_dim, layer_nums, output_feature_dim,
                 hidden_layer_rate=1, last_hidden_layer_use_relu=False, bias=True):
        super(Projector_head, self).__init__()
        self.layer_nums = layer_nums
        self.input_feature_dim = input_feature_dim
        self.output_feature_dim = output_feature_dim
        self.last_hidden_layer_use_relu = last_hidden_layer_use_relu
        self.hidden_layer_rate = hidden_layer_rate
        self.hidden_fc_layers = self._make_fc(bias)

    def _make_fc(self, bias=True):
        hidden_fc_layers = []
        input_dim = self.input_feature_dim
        if self.layer_nums == 1:
            if self.last_hidden_layer_use_relu:
                hidden_fc_layers.append(
                    fc_relu(self.input_feature_dim, self.output_feature_dim, bias=bias)
                )
            else:
                hidden_fc_layers = torch.nn.Linear(self.input_feature_dim, self.output_feature_dim, bias=bias)

        else:
            for layer in range(self.layer_nums):
                if layer < self.layer_nums - 1:
                    hidden_fc_layers.append(
                        fc_relu(input_dim, int(self.hidden_layer_rate * self.input_feature_dim), bias=bias)
                    )
                    input_dim = int(self.hidden_layer_rate * self.input_feature_dim)
                else:
                    if self.last_hidden_layer_use_relu:
                        hidden_fc_layers.append(
                            fc_relu(input_dim, self.output_feature_dim, bias=bias)
                        )
                    else:
                        hidden_fc_layers.append(
                            torch.nn.Linear(input_dim, self.output_feature_dim, bias=bias)
                        )

                    input_dim = self.input_feature_dim
        return nn.Sequential(*hidden_fc_layers)

    def forward(self, din):
        assert din.size(1) == self.input_feature_dim
        calibrated_features = self.hidden_fc_layers(din)
        return calibrated_features


class BarlowTwins(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # self.barlowtwins_extractor = nn.Sequential(*[resnet.__dict__[self.cfg.barlowtwins.BACKBONE.TYPE](
        # rate=self.cfg.barlowtwins.BACKBONE.rate), GAP()])
        if "18" in self.cfg.barlowtwins.BACKBONE.TYPE or "34" in self.cfg.barlowtwins.BACKBONE.TYPE:
            self.barlowtwins_extractor = nn.Sequential(*[resnet.__dict__[self.cfg.barlowtwins.BACKBONE.TYPE](
                rate=self.cfg.barlowtwins.BACKBONE.rate), GAP()])
        else:
            self.barlowtwins_extractor = nn.Sequential(*[resnet_cifar.__dict__[self.cfg.barlowtwins.BACKBONE.TYPE](
                rate=self.cfg.barlowtwins.BACKBONE.rate), GAP()])
        # projector
        self.projector = Projector_head(input_feature_dim=self.cfg.barlowtwins.PH.input_feature_dim,
                                        output_feature_dim=self.cfg.barlowtwins.PH.output_feature_dim,
                                        layer_nums=self.cfg.barlowtwins.PH.layer_nums,
                                        hidden_layer_rate=self.cfg.barlowtwins.PH.hidden_layer_rate,
                                        last_hidden_layer_use_relu=self.cfg.barlowtwins.PH.last_hidden_layer_use_relu,
                                        bias=self.cfg.barlowtwins.PH.bias)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(self.cfg.barlowtwins.PH.output_feature_dim, affine=False)
        if "linear" == self.cfg.barlowtwins.classifier.TYPE:
            self.classifier = torch.nn.Linear(self.cfg.barlowtwins.PH.output_feature_dim,
                                              self.cfg.DATASET.all_classes, bias=True)
        elif "mlp" == self.cfg.barlowtwins.classifier.TYPE:
            self.classifier = MLP_classifier(input_feature_dim=self.cfg.barlowtwins.PH.output_feature_dim,
                                             output_feature_dim=self.cfg.barlowtwins.PH.output_feature_dim,
                                             layer_nums=self.cfg.barlowtwins.classifier.layer_nums,
                                             hidden_layer_rate=self.cfg.barlowtwins.classifier.hidden_layer_rate,
                                             last_hidden_layer_use_relu=self.cfg.barlowtwins.classifier.last_hidden_layer_use_relu,
                                             bias=self.cfg.barlowtwins.classifier.bias,
                                             all_classes=self.cfg.DATASET.all_classes)

    def forward(self, x1, x2=None, finetune=False, **kwcfg):
        if "no_grad" in kwcfg or "is_nograd" in kwcfg:
            if "train_classifier" in kwcfg:
                features = self.forward_func(x1, finetune=finetune, feature_flag=True)
                return self.classifier(features)
            elif "get_classifier" in kwcfg:
                with torch.no_grad():
                    features = self.forward_func(x1, feature_flag=True)
                    return self.classifier(features)
            elif "feature_flag" in kwcfg or "use_projected_feature" in kwcfg:
                return self.forward_func(x1, **kwcfg)
            elif "MLP_feature" in kwcfg:
                with torch.no_grad():
                    features = self.forward_func(x1, feature_flag=True)
                    MLP_features = self.classifier(features, feature_flag=True)
                    return MLP_features
            else:
                with torch.no_grad():
                    f1 = self.barlowtwins_extractor(x1)
                    f2 = self.barlowtwins_extractor(x2)
                    f1 = f1.view(f1.shape[0], -1)
                    f2 = f2.view(f2.shape[0], -1)
                    z1 = self.projector(f1)
                    z2 = self.projector(f2)
                    return z1, z2
                    # return self.bn(z1), self.bn(z2)
        else:
            if "train_classifier" in kwcfg:
                features = self.forward_func(x1, finetune=finetune, feature_flag=True)
                return self.classifier(features)
            else:
                f1 = self.barlowtwins_extractor(x1)
                f2 = self.barlowtwins_extractor(x2)
                f1 = f1.view(f1.shape[0], -1)
                f2 = f2.view(f2.shape[0], -1)
                z1 = self.projector(f1)
                z2 = self.projector(f2)
                # return self.bn(z1), self.bn(z2)
                return z1, z2

    def forward_func(self, x, finetune=False, **kwcfg):
        features = None
        if finetune:
            if "feature_flag" in kwcfg:
                features = self.barlowtwins_extractor(x)
                features = features.view(features.shape[0], -1)
            elif "use_projected_feature" in kwcfg:
                features = self.barlowtwins_extractor(x)
                features = features.view(features.shape[0], -1)
                features = self.projector(features)
        else:
            mode = self.barlowtwins_extractor.training
            self.barlowtwins_extractor.eval()
            with torch.no_grad():
                if "feature_flag" in kwcfg:
                    features = self.barlowtwins_extractor(x)
                    features = features.view(features.shape[0], -1)
                elif "use_projected_feature" in kwcfg:
                    features = self.barlowtwins_extractor(x)
                    features = features.view(features.shape[0], -1)
                    features = self.projector(features)
            self.barlowtwins_extractor.train(mode)
        return features

    def load_model(self, model_path, logger=None):
        pretrain_dict = torch.load(
            model_path, map_location="cpu" if self.cfg.CPU_MODE else "cuda"
        )
        pretrain_dict = pretrain_dict['state_dict'] if 'state_dict' in pretrain_dict else pretrain_dict
        model_dict = self.state_dict()
        from collections import OrderedDict
        new_dict = OrderedDict()
        for k, v in pretrain_dict.items():
            if k.startswith("module"):
                new_dict[k[7:]] = v
            else:
                new_dict[k] = v

        model_dict.update(new_dict)
        self.load_state_dict(model_dict)
        if logger:
            logger.info("BarlowTwins Model has been loaded...")


class resnet_model(nn.Module):
    def __init__(self, cfg, cnn_type=None, rate=1., output_feature_dim=None):
        super().__init__()
        self.cfg = cfg
        if cnn_type is not None:
            assert output_feature_dim is not None
            if "18" in cnn_type or "34" in cnn_type or "50" in self.cfg.extractor.TYPE:
                self.extractor = nn.Sequential(*[resnet.__dict__[cnn_type](
                    rate=rate), GAP()])
            else:
                self.extractor = nn.Sequential(*[resnet_cifar.__dict__[cnn_type](
                    rate=rate), GAP()])

            self.linear_classifier = torch.nn.Linear(output_feature_dim,
                                                     self.cfg.DATASET.all_classes,
                                                     bias=True)
        else:
            if "18" in self.cfg.extractor.TYPE or "34" in self.cfg.extractor.TYPE or "50" in self.cfg.extractor.TYPE:
                self.extractor = nn.Sequential(*[resnet.__dict__[self.cfg.extractor.TYPE](
                    rate=self.cfg.extractor.rate), GAP()])
            else:
                self.extractor = nn.Sequential(*[resnet_cifar.__dict__[self.cfg.extractor.TYPE](
                    rate=self.cfg.extractor.rate), GAP()])
            self.linear_classifier = torch.nn.Linear(self.cfg.extractor.output_feature_dim,
                                                     self.cfg.DATASET.all_classes,
                                                     bias=True)

    def forward(self, x, **kwcfg):
        if "no_grad" in kwcfg or "is_nograd" in kwcfg:
            if "train_classifier" in kwcfg:
                features = self.forward_func(x)
                return self.linear_classifier(features), features
            elif "get_classifier" in kwcfg:
                with torch.no_grad():
                    features = self.forward_func(x)
                    return self.linear_classifier(features)
            elif "feature_flag" in kwcfg:
                return self.forward_func(x)
            elif "herding_feature" in kwcfg:
                return self.forward_func(x)
            elif "get_out_use_features" in kwcfg:
                with torch.no_grad():
                    return self.linear_classifier(x)
            else:
                with torch.no_grad():
                    features = self.forward_func(x)
                    return self.linear_classifier(features), features
        else:
            if "train_classifier" in kwcfg:
                features = self.forward_func(x)
                return self.linear_classifier(features)
            elif "train_extractor" in kwcfg:
                features = self.extractor(x)
                features = features.view(features.shape[0], -1)
                return features
            elif "train_cls_use_features" in kwcfg:
                return self.linear_classifier(x)
            else:
                features = self.extractor(x)
                features = features.view(features.shape[0], -1)
                return self.linear_classifier(features), features

    def forward_func(self, x):
        mode = self.extractor.training
        self.extractor.eval()
        with torch.no_grad():
            features = self.extractor(x)
        self.extractor.train(mode)
        features = features.view(features.shape[0], -1)
        return features

    def load_model(self, model_path):
        pretrain_dict = torch.load(
            model_path, map_location="cpu" if self.cfg.CPU_MODE else "cuda"
        )
        pretrain_dict = pretrain_dict['state_dict'] if 'state_dict' in pretrain_dict else pretrain_dict
        model_dict = self.state_dict()
        from collections import OrderedDict
        new_dict = OrderedDict()
        for k, v in pretrain_dict.items():
            if k.startswith("module"):
                new_dict[k[7:]] = v
            else:
                new_dict[k] = v

        model_dict.update(new_dict)
        self.load_state_dict(model_dict)

    def freeze(self):
        for name, param in self.named_parameters():
            param.requires_grad = False
        pass

    def unfreeze_module(self, module_name=None):
        if module_name is None:
            module_name = ["layer3", "stage5", "linear_classifier"]
        for name, param in self.named_parameters():
            for module_item in module_name:
                if module_item in name:
                    param.requires_grad = True
        pass

    def unfreeze_all(self):
        for name, param in self.named_parameters():
            param.requires_grad = True
        pass


class warm_start_resnet_model(nn.Module):
    def __init__(self, cfg, cnn_type=None, rate=1., output_feature_dim=None):
        super().__init__()
        self.cfg = cfg
        if cnn_type is not None:
            assert output_feature_dim is not None
            if "18" in cnn_type or "34" in cnn_type:
                self.extractor = nn.Sequential(*[resnet.__dict__[cnn_type](
                    rate=rate), GAP()])
            else:
                self.extractor = nn.Sequential(*[resnet_cifar.__dict__[cnn_type](
                    rate=rate), GAP()])

            self.linear_classifier = torch.nn.Linear(self.cfg.extractor.output_feature_dim,
                                                     cfg.DATASET.warm_start_all_classes,
                                                     bias=True)
        else:
            if "18" in self.cfg.extractor.TYPE or "34" in self.cfg.extractor.TYPE:
                self.extractor = nn.Sequential(*[resnet.__dict__[self.cfg.extractor.TYPE](
                    rate=self.cfg.extractor.rate), GAP()])
            else:
                self.extractor = nn.Sequential(*[resnet_cifar.__dict__[self.cfg.extractor.TYPE](
                    rate=self.cfg.extractor.rate), GAP()])
            self.linear_classifier = torch.nn.Linear(self.cfg.extractor.output_feature_dim,
                                                     cfg.DATASET.warm_start_all_classes,
                                                     bias=True)
        self.linear_classifier.apply(_weights_init)

    def forward(self, x, **kwcfg):
        if "no_grad" in kwcfg or "is_nograd" in kwcfg:
            if "train_classifier" in kwcfg:
                features = self.forward_func(x)
                return self.linear_classifier(features), features
            elif "get_classifier" in kwcfg:
                with torch.no_grad():
                    features = self.forward_func(x)
                    return self.linear_classifier(features)
            elif "feature_flag" in kwcfg:
                return self.forward_func(x)
            elif "herding_feature" in kwcfg:
                return self.forward_func(x)
            elif "get_out_use_features" in kwcfg:
                with torch.no_grad():
                    return self.linear_classifier(x)
            else:
                with torch.no_grad():
                    features = self.forward_func(x)
                    return self.linear_classifier(features), features
        else:
            if "train_classifier" in kwcfg:
                features = self.forward_func(x)
                return self.linear_classifier(features)
            elif "train_extractor" in kwcfg:
                features = self.extractor(x)
                features = features.view(features.shape[0], -1)
                return features
            elif "train_cls_use_features" in kwcfg:
                return self.linear_classifier(x)
            else:
                features = self.extractor(x)
                features = features.view(features.shape[0], -1)
                return self.linear_classifier(features), features

    def forward_func(self, x):
        mode = self.extractor.training
        self.extractor.eval()
        with torch.no_grad():
            features = self.extractor(x)
        self.extractor.train(mode)
        features = features.view(features.shape[0], -1)
        return features

    def load_model(self, model_path):
        pretrain_dict = torch.load(
            model_path, map_location="cpu" if self.cfg.CPU_MODE else "cuda"
        )
        pretrain_dict = pretrain_dict['state_dict'] if 'state_dict' in pretrain_dict else pretrain_dict
        model_dict = self.state_dict()
        from collections import OrderedDict
        new_dict = OrderedDict()
        for k, v in pretrain_dict.items():
            if k.startswith("module"):
                new_dict[k[7:]] = v
            else:
                new_dict[k] = v

        model_dict.update(new_dict)
        self.load_state_dict(model_dict)


class resnet_model_for_podnet(nn.Module):
    def __init__(self, cfg, cnn_type=None, rate=1., output_feature_dim=None, last_relu=True):
        super().__init__()
        self.cfg = cfg
        self.relu = nn.ReLU(inplace=False)
        self.avgpool = GAP()
        if cnn_type is not None:
            assert output_feature_dim is not None
            if "18" in cnn_type or "34" in cnn_type:
                self.extractor = resnet.__dict__[cnn_type](rate=rate, last_relu=last_relu)
            else:
                self.extractor = resnet_cifar.__dict__[cnn_type](rate=rate, last_relu=last_relu)

            self.linear_classifier = torch.nn.Linear(output_feature_dim,
                                                     self.cfg.DATASET.all_classes, bias=True)
        else:
            if "18" in self.cfg.extractor.TYPE or "34" in self.cfg.extractor.TYPE:
                self.extractor = resnet.__dict__[self.cfg.extractor.TYPE](rate=self.cfg.extractor.rate,
                                                                          last_relu=self.cfg.extractor.last_relu)
            else:
                self.extractor = resnet_cifar.__dict__[self.cfg.extractor.TYPE](rate=self.cfg.extractor.rate,
                                                                                last_relu=self.cfg.extractor.last_relu)

            if self.cfg.classifier.classifier_type == "cosine":
                self.linear_classifier = CosineClassifier(all_classes=self.cfg.DATASET.all_classes,
                                                          all_tasks=self.cfg.DATASET.all_tasks,
                                                          features_dim=self.cfg.extractor.output_feature_dim,
                                                          proxy_per_class=self.cfg.classifier.proxy_per_class,
                                                          distance=self.cfg.classifier.distance,
                                                          merging=self.cfg.classifier.merging,
                                                          scaling=None,
                                                          gamma=1.)
            else:
                self.linear_classifier = torch.nn.Linear(self.cfg.extractor.output_feature_dim,
                                                         self.cfg.DATASET.all_classes, bias=True)

    def forward(self, x, **kwcfg):
        if "no_grad" in kwcfg or "is_nograd" in kwcfg:
            if "train_classifier" in kwcfg:
                outputs = self.forward_func(x)
                if self.cfg.classifier.classifier_type == "cosine":
                    return self.linear_classifier(outputs["raw_features"]), outputs["raw_features"]
                else:
                    return self.linear_classifier(outputs["features"]), outputs["features"]
            elif "get_classifier" in kwcfg:
                with torch.no_grad():
                    outputs = self.forward_func(x)
                    if self.cfg.classifier.classifier_type == "cosine":
                        return self.linear_classifier(outputs["raw_features"])
                    else:
                        return self.linear_classifier(outputs["features"])
            elif "feature_flag" in kwcfg:
                return self.forward_func(x)
            elif "herding_feature" in kwcfg:
                outputs = self.forward_func(x)
                if self.cfg.classifier.classifier_type == "cosine":
                    return outputs["raw_features"]
                else:
                    return outputs["features"]
        else:
            if "train_classifier" in kwcfg:
                outputs = self.forward_func(x)
                if self.cfg.classifier.classifier_type == "cosine":
                    return self.linear_classifier(outputs["raw_features"])
                else:
                    return self.linear_classifier(outputs["features"])
            elif "feature_flag" in kwcfg:
                atts = self.extractor(x)
                raw_features = self.end_features(atts[-1])
                features = self.end_features(self.relu(atts[-1]))
                return {
                    "raw_features": raw_features,
                    "features": features,
                    "attention": atts
                }
            else:
                atts = self.extractor(x)
                raw_features = self.end_features(atts[-1])
                features = self.end_features(self.relu(atts[-1]))
                if self.cfg.classifier.classifier_type == "cosine":
                    clf_outputs = self.linear_classifier(raw_features)
                else:
                    clf_outputs = self.linear_classifier(features)
                return {
                    "raw_features": raw_features,
                    "features": features,
                    "attention": atts,
                    "logits": clf_outputs
                }

    def forward_func(self, x):
        mode = self.extractor.training
        self.extractor.eval()
        with torch.no_grad():
            atts = self.extractor(x)
        raw_features = self.end_features(atts[-1])
        features = self.end_features(self.relu(atts[-1]))
        self.extractor.train(mode)
        return {
            "raw_features": raw_features,
            "features": features,
            "attention": atts
        }

    def end_features(self, x):
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        return x

    def load_model(self, model_path, logger=None):
        pretrain_dict = torch.load(
            model_path, map_location="cpu" if self.cfg.CPU_MODE else "cuda"
        )
        pretrain_dict = pretrain_dict['state_dict'] if 'state_dict' in pretrain_dict else pretrain_dict
        model_dict = self.state_dict()
        from collections import OrderedDict
        new_dict = OrderedDict()
        for k, v in pretrain_dict.items():
            if k.startswith("module"):
                new_dict[k[7:]] = v
            else:
                new_dict[k] = v

        model_dict.update(new_dict)
        self.load_state_dict(model_dict)
        if logger:
            logger.info("BarlowTwins Model has been loaded...")


class domain_model(nn.Module):
    # cfg, cnn_type = None, rate = 1., output_feature_dim = None
    def __init__(self, cfg):
        super(domain_model, self).__init__()
        self.cfg = cfg
        self.extractor = eval(self.cfg.extractor.TYPE)(rate=self.cfg.extractor.rate)
        if self.cfg.model.use_dif_domain:
            self.linear_classifier = nn.Linear(self.cfg.extractor.output_feature_dim,
                                               self.cfg.DATASET.all_classes, bias=True)
        else:
            self.linear_classifier = nn.Linear(self.cfg.extractor.output_feature_dim,
                                               int(self.cfg.DATASET.all_classes / self.cfg.DATASET.all_tasks),
                                               bias=True)

    def forward(self, x, **kwcfg):
        if "no_grad" in kwcfg or "is_nograd" in kwcfg:
            if "train_classifier" in kwcfg:
                features = self.forward_func(x)
                return self.linear_classifier(features), features
            elif "get_classifier" in kwcfg:
                with torch.no_grad():
                    features = self.forward_func(x)
                    return self.linear_classifier(features)
            elif "feature_flag" in kwcfg:
                return self.forward_func(x)
        else:
            if "train_classifier" in kwcfg:
                features = self.forward_func(x)
                return self.linear_classifier(features)
            else:
                features = self.extractor(x)
                return self.linear_classifier(features), features

    def forward_func(self, x):
        mode = self.extractor.training
        self.extractor.eval()
        with torch.no_grad():
            features = self.extractor(x)
        self.extractor.train(mode)
        return features

    def load_model(self, model_path, logger=None):
        pretrain_dict = torch.load(
            model_path, map_location="cuda"
        )
        pretrain_dict = pretrain_dict['state_dict'] if 'state_dict' in pretrain_dict else pretrain_dict
        model_dict = self.state_dict()
        from collections import OrderedDict
        new_dict = OrderedDict()
        for k, v in pretrain_dict.items():
            if k.startswith("module"):
                new_dict[k[7:]] = v
            else:
                new_dict[k] = v

        model_dict.update(new_dict)
        self.load_state_dict(model_dict)
        if logger:
            logger.info("CNN Model has been loaded...")

    def freeze_backbone(self, logger=None):
        if logger:
            logger.info("Freezing backbone .......")
        for p in self.extractor.parameters():
            p.requires_grad = False


class FactorScalar(nn.Module):

    def __init__(self, initial_value=1., **kwcfg):
        super().__init__()

        self.factor = nn.Parameter(torch.tensor(initial_value))

    def on_task_end(self):
        pass

    def on_epoch_end(self):
        pass

    def forward(self, inputs):
        return self.factor * inputs

    def __mul__(self, other):
        return self.forward(other)

    def __rmul__(self, other):
        return self.forward(other)


class CosineClassifier(nn.Module):
    classifier_type = "cosine"

    def __init__(
            self,
            all_classes,
            all_tasks,
            features_dim,
            proxy_per_class=1,
            distance="cosine",
            merging="softmax",
            scaling=None,
            gamma=1.,
    ):
        super().__init__()
        self.all_classes = all_classes
        self.all_tasks = all_tasks
        self.classes_per_task = int(all_classes / all_tasks)
        self.bias = None
        self.features_dim = features_dim
        self.proxy_per_class = proxy_per_class
        self.distance = distance
        self.merging = merging
        self.gamma = gamma
        self._weights = nn.Parameter(torch.zeros(self.all_tasks * self.proxy_per_class * self.classes_per_task,
                                                 self.features_dim))
        if isinstance(scaling, int) or isinstance(scaling, float):
            self.scaling = scaling
        else:
            self.scaling = FactorScalar(1.)

    def forward(self, features):
        if self.distance == "cosine":
            raw_similarities = cosine_similarity(features, self._weights)
        elif self.distance == "neg_stable_cosine_distance":
            features = self.scaling * F.normalize(features, p=2, dim=-1)
            weights = self.scaling * F.normalize(self._weights, p=2, dim=-1)
            raw_similarities = -stable_cosine_distance(features, weights)
        else:
            raise NotImplementedError("Unknown distance function {}.".format(self.distance))

        if self.proxy_per_class > 1:
            similarities = self._reduce_proxies(raw_similarities)
        else:
            similarities = raw_similarities

        return similarities

    def _reduce_proxies(self, similarities):
        # shape (batch_size, n_classes * proxy_per_class)
        n_classes = similarities.shape[1] / self.proxy_per_class
        assert n_classes.is_integer(), (similarities.shape[1], self.proxy_per_class)
        n_classes = int(n_classes)
        assert n_classes == self.all_classes, (n_classes, self.all_classes)
        bs = similarities.shape[0]

        if self.merging == "mean":
            return similarities.view(bs, n_classes, self.proxy_per_class).mean(-1)
        elif self.merging == "softmax":
            simi_per_class = similarities.view(bs, n_classes, self.proxy_per_class)
            attentions = F.softmax(self.gamma * simi_per_class, dim=-1)  # shouldn't be -gamma?
            return (simi_per_class * attentions).sum(-1)
        elif self.merging == "max":
            return similarities.view(bs, n_classes, self.proxy_per_class).max(-1)[0]
        elif self.merging == "min":
            return similarities.view(bs, n_classes, self.proxy_per_class).min(-1)[0]
        else:
            raise ValueError("Unknown merging for multiple centers: {}.".format(self.merging))


class FCTM_model(nn.Module):
    def __init__(self, cfg, backbone_model=None):
        super(FCTM_model, self).__init__()
        self.fctm = backbone_model if backbone_model is not None else resnet_model(cfg)
        self.FCN = None
        if cfg.FCTM.use_FCN:
            self.FCN = Projector_head(input_feature_dim=cfg.FCTM.FCN.in_feature_dim, layer_nums=cfg.FCTM.FCN.layer_nums,
                                      output_feature_dim=cfg.FCTM.FCN.out_feature_dim,
                                      hidden_layer_rate=cfg.FCTM.FCN.hidden_layer_rate,
                                      last_hidden_layer_use_relu=cfg.FCTM.FCN.last_hidden_layer_use_relu)
            self.global_fc = nn.Linear(cfg.FCTM.FCN.out_feature_dim,
                                       cfg.DATASET.all_classes, bias=True)
        else:
            self.global_fc = nn.Linear(cfg.FCTM.FCN.in_feature_dim,
                                       cfg.DATASET.all_classes, bias=True)

    def forward(self, x, pre_model_feature, **kwcfg):
        if "no_grad" in kwcfg or "is_nograd" in kwcfg:
            fctm_features = self.fctm(x, is_nograd=True, feature_flag=True)
            features = torch.cat([pre_model_feature, fctm_features], dim=1)
            if "feature_flag" in kwcfg:
                return fctm_features
            elif "calibrated_features_flag" in kwcfg:
                with torch.no_grad():
                    if self.FCN is not None:
                        calibrated_features = self.FCN(features)
                        return calibrated_features
                    else:
                        return features
            else:
                with torch.no_grad():
                    if self.FCN is not None:
                        calibrated_features = self.FCN(features)
                        outputs = self.global_fc(calibrated_features)
                    else:
                        outputs = self.global_fc(features)
                return outputs
        else:
            if "train_cls_use_features" in kwcfg:
                features = x
                outputs = self.global_fc(features)
                return {
                    "features": features,
                    "all_logits": outputs,
                    "fctm_logits": None
                }
            else:
                fctm_outputs, fctm_features = self.fctm(x)
                features = torch.cat([pre_model_feature, fctm_features], dim=1)
                if self.FCN is not None:
                    calibrated_features = self.FCN(features)
                    outputs = self.global_fc(calibrated_features)
                else:
                    calibrated_features = features
                    outputs = self.global_fc(features)
                return {
                    "features": calibrated_features,
                    "all_logits": outputs,
                    "fctm_logits": fctm_outputs
                }

    def load_model(self, model_path, logger=None):
        pretrain_dict = torch.load(
            model_path, map_location="cuda"
        )
        pretrain_dict = pretrain_dict['state_dict'] if 'state_dict' in pretrain_dict else pretrain_dict
        model_dict = self.state_dict()
        from collections import OrderedDict
        new_dict = OrderedDict()
        for k, v in pretrain_dict.items():
            if k.startswith("module"):
                new_dict[k[7:]] = v
            else:
                new_dict[k] = v

        model_dict.update(new_dict)
        self.load_state_dict(model_dict)
        if logger:
            logger.info("CNN Model has been loaded...")


class warm_start_FCTM_model(nn.Module):
    def __init__(self, cfg):
        super(warm_start_FCTM_model, self).__init__()
        self.fctm = resnet_model(cfg)
        self.FCN = None
        if cfg.FCTM.use_FCN:
            self.FCN = Projector_head(input_feature_dim=cfg.FCTM.FCN.in_feature_dim, layer_nums=cfg.FCTM.FCN.layer_nums,
                                      output_feature_dim=cfg.FCTM.FCN.out_feature_dim,
                                      hidden_layer_rate=cfg.FCTM.FCN.hidden_layer_rate,
                                      last_hidden_layer_use_relu=cfg.FCTM.FCN.last_hidden_layer_use_relu)
            self.global_fc = nn.Linear(cfg.FCTM.FCN.out_feature_dim,
                                       cfg.DATASET.warm_start_all_classes,
                                       bias=True)
        else:
            self.global_fc = nn.Linear(cfg.FCTM.FCN.in_feature_dim,
                                       cfg.DATASET.warm_start_all_classes,
                                       bias=True)

    def forward(self, x, pre_model_feature, **kwcfg):
        if "no_grad" in kwcfg or "is_nograd" in kwcfg:
            fctm_features = self.fctm(x, is_nograd=True, feature_flag=True)
            features = torch.cat([pre_model_feature, fctm_features], dim=1)
            if "feature_flag" in kwcfg:
                return fctm_features
            elif "calibrated_features_flag" in kwcfg:
                with torch.no_grad():
                    calibrated_features = self.FCN(features)
                return calibrated_features
            else:
                with torch.no_grad():
                    if self.FCN is not None:
                        calibrated_features = self.FCN(features)
                        outputs = self.global_fc(calibrated_features)
                    else:
                        outputs = self.global_fc(features)
                return outputs
        else:
            fctm_outputs, fctm_features = self.fctm(x)
            features = torch.cat([pre_model_feature, fctm_features], dim=1)
            if self.FCN is not None:
                calibrated_features = self.FCN(features)
                outputs = self.global_fc(calibrated_features)
            else:
                outputs = self.global_fc(features)
            return {
                "all_logits": outputs,
                "fctm_logits": fctm_outputs
            }

    def load_model(self, model_path, logger=None):
        pretrain_dict = torch.load(
            model_path, map_location="cuda"
        )
        pretrain_dict = pretrain_dict['state_dict'] if 'state_dict' in pretrain_dict else pretrain_dict
        model_dict = self.state_dict()
        from collections import OrderedDict
        new_dict = OrderedDict()
        for k, v in pretrain_dict.items():
            if k.startswith("module"):
                new_dict[k[7:]] = v
            else:
                new_dict[k] = v

        model_dict.update(new_dict)
        self.load_state_dict(model_dict)
        if logger:
            logger.info("CNN Model has been loaded...")


class FCN_model(nn.Module):
    def __init__(self, cfg):
        super(FCN_model, self).__init__()
        self.FCN = None
        self.FCN = Projector_head(input_feature_dim=cfg.FCTM.FCN.in_feature_dim, layer_nums=cfg.FCTM.FCN.layer_nums,
                                  output_feature_dim=cfg.FCTM.FCN.out_feature_dim,
                                  hidden_layer_rate=cfg.FCTM.FCN.hidden_layer_rate,
                                  last_hidden_layer_use_relu=cfg.FCTM.FCN.last_hidden_layer_use_relu)
        self.global_fc = nn.Linear(cfg.FCTM.FCN.in_feature_dim,
                                   cfg.DATASET.all_classes, bias=True)

    def forward(self, pre_model_feature, **kwcfg):
        if "no_grad" in kwcfg or "is_nograd" in kwcfg:
            features = pre_model_feature
            with torch.no_grad():
                calibrated_features = self.FCN(features)
            if "feature_flag" in kwcfg:
                return calibrated_features
            else:
                with torch.no_grad():
                    outputs = self.global_fc(calibrated_features)
                return outputs
        else:
            if "calibrate_features_2_logits" in kwcfg:
                calibrated_features = pre_model_feature
                outputs = self.global_fc(calibrated_features)
            else:
                features = pre_model_feature
                calibrated_features = self.FCN(features)
                outputs = self.global_fc(calibrated_features)
            return {
                "all_logits": outputs,
                "calibrated_features": calibrated_features
            }

    def load_model(self, model_path, logger=None):
        pretrain_dict = torch.load(
            model_path, map_location="cuda"
        )
        pretrain_dict = pretrain_dict['state_dict'] if 'state_dict' in pretrain_dict else pretrain_dict
        model_dict = self.state_dict()
        from collections import OrderedDict
        new_dict = OrderedDict()
        for k, v in pretrain_dict.items():
            if k.startswith("module"):
                new_dict[k[7:]] = v
            else:
                new_dict[k] = v

        model_dict.update(new_dict)
        self.load_state_dict(model_dict)
        if logger:
            logger.info("FCN Model has been loaded...")


class resnet_model_with_adjusted_layer(nn.Module):
    def __init__(self, cfg, cnn_type=None, rate=1., output_feature_dim=None):
        super(resnet_model_with_adjusted_layer, self).__init__()
        self.cfg = cfg
        self.class_per_task = int(self.cfg.DATASET.all_classes / self.cfg.DATASET.all_tasks)
        self.adjusted_layer = None
        if "linear" == self.cfg.model.adjusted_layer_type:
            self.adjusted_layer = nn.Parameter(torch.Tensor([1. for p in range(self.cfg.DATASET.all_classes -
                                                                               self.class_per_task)]))
        elif "para-1" == self.cfg.model.adjusted_layer_type:
            self.adjusted_layer = nn.Parameter(torch.Tensor([1.]))
        elif "para-2" == self.cfg.model.adjusted_layer_type:
            self.adjusted_layer = nn.Parameter(torch.Tensor([1., 0]))
        if cnn_type is None:
            self.resnet_model = resnet_model(self.cfg)
        else:
            self.resnet_model = resnet_model(self.cfg, cnn_type=cnn_type, rate=rate,
                                             output_feature_dim=output_feature_dim)

    def forward(self, x, img_index=None, **kwcfg):
        outs = self.resnet_model(x, **kwcfg)
        if "no_grad" in kwcfg or "is_nograd" in kwcfg:
            return outs
        else:
            if "train_classifier" in kwcfg:
                return {"logits": outs, "adjusted_logits": self.forward_adjusted_layer(outs, img_index),
                        "features": None}
            elif "train_extractor" in kwcfg:
                return {"logits": None, "adjusted_logits": None, "features": outs}
            elif "train_cls_use_features" in kwcfg:
                return {"logits": outs, "adjusted_logits": self.forward_adjusted_layer(outs, img_index),
                        "features": None}
            else:
                return {"logits": outs[0], "adjusted_logits": self.forward_adjusted_layer(outs[0], img_index),
                        "features": outs[1]}

    def forward_adjusted_layer(self, inputs, img_index=None):
        # print("inputs:", inputs.shape)
        # print("adjusted_layer:", self.adjusted_layer)
        if "linear" == self.cfg.model.adjusted_layer_type:
            temps = inputs * self.adjusted_layer
            temps[:, -self.class_per_task:] = inputs[:, -self.class_per_task:]
        elif "para-1" == self.cfg.model.adjusted_layer_type:
            temps = inputs * self.adjusted_layer[0]
            temps[:, -self.class_per_task:] = inputs[:, -self.class_per_task:]
        else:
            temps = inputs * self.adjusted_layer[0]
            temps += self.adjusted_layer[1]
            temps[:, -self.class_per_task:] = inputs[:, -self.class_per_task:]
        if img_index is not None:
            to_be_replaced = torch.where(img_index > -0.5)[0]
            original_imgs_index = torch.cuda.LongTensor(img_index[to_be_replaced])
            mask = torch.zeros_like(inputs).bool()
            mask = mask.index_fill_(0, to_be_replaced, 1)
            outs = torch.masked_scatter(inputs, mask, temps[original_imgs_index])
            return outs
        else:
            return temps

    def load_model(self, model_path, logger=None):
        pretrain_dict = torch.load(
            model_path, map_location="cpu" if self.cfg.CPU_MODE else "cuda"
        )
        pretrain_dict = pretrain_dict['state_dict'] if 'state_dict' in pretrain_dict else pretrain_dict
        model_dict = self.state_dict()
        from collections import OrderedDict
        new_dict = OrderedDict()
        for k, v in pretrain_dict.items():
            if k.startswith("module"):
                new_dict[k[7:]] = v
            else:
                new_dict[k] = v

        model_dict.update(new_dict)
        self.load_state_dict(model_dict)
        if logger:
            logger.info("BarlowTwins Model has been loaded...")


class LeNet5(nn.Module):
    """
    略微修改过的 LeNet5 模型
    Attributes:
        need_dropout (bool): 是否需要增加随机失活层
        conv1 (nn.Conv2d): 卷积核1，默认维度 (6, 5, 5)
        pool1 (nn.MaxPool2d): 下采样函数1，维度 (2, 2)
        conv2 (nn.Conv2d): 卷积核2，默认维度 (16, 5, 5)
        pool2 (nn.MaxPool2d): 下采样函数2，维度 (2, 2)
        conv3 (nn.Conv2d): 卷积核3，默认维度 (120, 5, 5)
        fc1 (nn.Linear): 全连接函数1，维度 (120, 84)
        fc2 (nn.Linear): 全连接函数2，维度 (84, 10)
        dropout (nn.Dropout): 随机失活函数
    """

    def __init__(self, dropout_prob=0., halve_conv_kernels=False):
        """
        初始化模型各层函数
        :param dropout_prob: 随机失活参数
        :param halve_conv_kernels: 是否将卷积核数量减半
        """
        super(LeNet5, self).__init__()
        kernel_nums = [6, 16]
        if halve_conv_kernels:
            kernel_nums = [num // 2 for num in kernel_nums]
        self.need_dropout = dropout_prob > 0

        # 卷积层 1，6个 5*5 的卷积核
        # 由于输入图像是 28*28，所以增加 padding=2，扩充到 32*32
        self.conv1 = nn.Conv2d(1, kernel_nums[0], (5, 5), padding=2)
        # 下采样层 1，采样区为 2*2
        self.pool1 = nn.MaxPool2d((2, 2))
        # 卷积层 2，16个 5*5 的卷积核
        self.conv2 = nn.Conv2d(kernel_nums[0], kernel_nums[1], (5, 5))
        # 下采样层 2，采样区为 2*2
        self.pool2 = nn.MaxPool2d((2, 2))
        # 卷积层 3，120个 5*5 的卷积核
        self.conv3 = nn.Conv2d(kernel_nums[1], 120, (5, 5))
        self.extractor = nn.Sequential(*[self.conv1, self.pool1, self.conv2, self.pool2, self.conv3])
        # 全连接层 1，120*84 的全连接矩阵
        self.fc1 = nn.Linear(120, 84)
        # 全连接层 2，84*10 的全连接矩阵
        self.fc2 = nn.Linear(84, 10)
        self.classifier = nn.Sequential(*[self.fc1, self.fc2])

    # def forward(self, x):
    #     """
    #     前向传播函数，返回给定输入数据的预测标签数组
    #     :param x: 维度为 (batch_size, 28, 28) 的图像数据
    #     :return: 维度为 (batch_size, 10) 的预测标签
    #     """
    #     x = x.unsqueeze(1)                      # (batch_size, 1, 28, 28)
    #     feature_map = self.conv1(x)             # (batch_size, 6, 28, 28)
    #     feature_map = self.pool1(feature_map)   # (batch_size, 6, 14, 14)
    #     feature_map = self.conv2(feature_map)   # (batch_size, 16, 10, 10)
    #     feature_map = self.pool2(feature_map)   # (batch_size, 16, 5, 5)
    #     feature_map = self.conv3(feature_map).squeeze()     # (batch_size, 120)
    #     out = self.fc1(feature_map)             # (batch_size, 84)
    #     if self.need_dropout:
    #         out = self.dropout(out)             # (batch_size, 10)
    #     out = self.fc2(out)                     # (batch_size, 10)
    #     return out

    def forward(self, x, **kwcfg):
        if "no_grad" in kwcfg or "is_nograd" in kwcfg:
            if "train_classifier" in kwcfg:
                features = self.forward_func(x)
                return self.classifier(features), features
            elif "get_classifier" in kwcfg:
                with torch.no_grad():
                    features = self.forward_func(x)
                    return self.classifier(features)
            elif "feature_flag" in kwcfg:
                return self.forward_func(x)
            elif "herding_feature" in kwcfg:
                return self.forward_func(x)
            elif "get_out_use_features" in kwcfg:
                with torch.no_grad():
                    return self.linear_classifier(x)
            else:
                with torch.no_grad():
                    features = self.forward_func(x)
                    return self.classifier(features), features
        else:
            if "train_classifier" in kwcfg:
                features = self.forward_func(x)
                return self.classifier(features)
            elif "train_extractor" in kwcfg:
                features = self.extractor(x)
                features = features.view(features.shape[0], -1)
                return features
            elif "train_cls_use_features" in kwcfg:
                return self.classifier(x)
            else:
                features = self.extractor(x)
                features = features.view(features.shape[0], -1)
                return self.classifier(features), features

    def forward_func(self, x):
        mode = self.extractor.training
        self.extractor.eval()
        with torch.no_grad():
            features = self.extractor(x)
            features = features.view(features.shape[0], -1)
        self.extractor.train(mode)
        return features


class ResNet_for_CAM(nn.Module):
    def __init__(self, block, layers, rate=1, class_num=4, inter_layer=False):
        super(ResNet_for_CAM, self).__init__()
        self.inter_layer = inter_layer
        self.in_channels = int(64 * rate)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, int(64 * rate), kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(int(64 * rate)), nn.ReLU(inplace=False))

        self.stage2 = self._make_layer(block, int(64 * rate), layers[0], 1)
        self.stage3 = self._make_layer(block, int(128 * rate), layers[1], 2)
        self.stage4 = self._make_layer(block, int(256 * rate), layers[2], 2)
        self.stage5 = self._make_layer(block, int(512 * rate), layers[3], 2)
        self.gap = GAP()
        self.classifier = torch.nn.Linear(512, class_num, bias=True)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        cfg:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def load_model(self, model_path):
        pretrain_dict = torch.load(
            model_path, map_location="cuda"
        )
        pretrain_dict = pretrain_dict['state_dict'] if 'state_dict' in pretrain_dict else pretrain_dict
        model_dict = self.state_dict()
        from collections import OrderedDict
        new_dict = OrderedDict()
        for k, v in pretrain_dict.items():
            if k.startswith("module"):
                new_dict[k[7:]] = v
            else:
                new_dict[k] = v

        model_dict.update(new_dict)
        self.load_state_dict(model_dict)

    def forward_func(self, x):
        mode = self.conv1.training
        self.conv1.eval()
        self.stage2.eval()
        self.stage3.eval()
        self.stage4.eval()
        self.stage5.eval()
        self.gap.eval()
        with torch.no_grad():
            x = self.conv1(x)
            x = self.stage2(x)
            x = self.stage3(x)
            x = self.stage4(x)
            x = self.stage5(x)
            features = self.gap(x)
            features = features.view(x.shape[0], -1)
        self.conv1.train(mode)
        self.stage2.train(mode)
        self.stage3.train(mode)
        self.stage4.train(mode)
        self.stage5.train(mode)
        self.gap.train(mode)
        return features

    # def forward(self, x):
    #     x = self.conv1(x)
    #     x = self.stage2(x)
    #     x = self.stage3(x)
    #     x = self.stage4(x)
    #     x = self.stage5(x)
    #     features = self.gap(x)
    #     features = features.view(x.shape[0], -1)
    #     outputs = self.classifier(features)
    #     return outputs, features

    def forward(self, x, **kwcfg):
        if "no_grad" in kwcfg or "is_nograd" in kwcfg:
            if "train_classifier" in kwcfg:
                features = self.forward_func(x)
                return self.classifier(features), features
            elif "get_classifier" in kwcfg:
                with torch.no_grad():
                    features = self.forward_func(x)
                    return self.classifier(features)
            elif "feature_flag" in kwcfg:
                return self.forward_func(x)
            elif "get_out_use_features" in kwcfg:
                with torch.no_grad():
                    return self.classifier(x)
            else:
                with torch.no_grad():
                    features = self.forward_func(x)
                    return self.classifier(features), features
        else:
            if "train_classifier" in kwcfg:
                features = self.forward_func(x)
                return self.classifier(features)
            elif "train_cls_use_features" in kwcfg:
                return self.classifier(x)
            else:
                x = self.conv1(x)
                x = self.stage2(x)
                x = self.stage3(x)
                x = self.stage4(x)
                x = self.stage5(x)
                features = self.gap(x)
                features = features.view(x.shape[0], -1)
                outputs = self.classifier(features)
                return outputs, features


class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, last_relu=True):
        super(BasicBlock, self).__init__()
        self.last_relu = last_relu
        self.residual_branch = nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=3,
                      stride=stride,
                      padding=1,
                      bias=False), nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels,
                      out_channels * BasicBlock.expansion,
                      kernel_size=3,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion))

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels * BasicBlock.expansion,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion))

    def forward(self, x):
        if self.last_relu:
            return F.relu(self.residual_branch(x) + self.shortcut(x))
        else:
            return self.residual_branch(x) + self.shortcut(x)


def _resnet(block, layers, rate, class_num, **kwcfg):
    model = ResNet_for_CAM(block, layers, rate, class_num, **kwcfg)
    return model


def cam_resnet18(rate=1, class_num=4, **kwcfg):
    return _resnet(BasicBlock, [2, 2, 2, 2], rate, class_num, **kwcfg)


class NcmClassifier(nn.Module):
    def __init__(self, feat_dim=64):
        super(NcmClassifier, self).__init__()
        self.feat_dim = feat_dim
        self.register_parameter(name='T',
                                param=torch.nn.Parameter(torch.ones(self.feat_dim).float()))  # channel wise scaling

    def forward(self, x, centroids, stddevs=None, phase='train'):
        dists = torch.sqrt(torch.sum(torch.square(torch.div(x[:, None, :] - centroids[None, :], self.T[None, None])),
                                     dim=-1))  # channel wise scaling
        # scores are negative of the distances themselves.
        return -dists / 2
