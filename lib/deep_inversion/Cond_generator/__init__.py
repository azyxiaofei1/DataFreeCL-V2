# -*- coding: utf-8 -*-

from typing import List
from functools import partial

import torch.nn as nn

from .cond_cifar import Cond_CIFAR_GEN
from .cond_tiny_imagenet import Cond_TINYIMNET_GEN
from .cond_imagenet import Cond_IMNET_GEN


__all__ = ["names", "create"]


__factory = {
    "cond_cifar100": Cond_CIFAR_GEN,
    "cond_tiny-imagenet200": Cond_TINYIMNET_GEN,
    "cond_imagenet100": Cond_IMNET_GEN,
    "cond_imagenet1000": Cond_IMNET_GEN,
}


def names() -> List[str]:
    return sorted(__factory.keys())


def create(name: str) -> nn.Module:
    if name not in __factory:
        raise KeyError(f"Unknown dataset: {name}")

    return __factory[name]()
