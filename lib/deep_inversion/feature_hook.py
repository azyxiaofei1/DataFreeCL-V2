# -*- coding: utf-8 -*-

"""
DeepInversion

credits:
    https://github.com/NVlabs/DeepInversion
"""

import math
import random
from typing import Tuple, Callable

import torch


class DeepInversionFeatureHook:
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''

    def __init__(self, module, gram_matrix_weight, layer_weight):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.target = None
        self.gram_matrix_weight = gram_matrix_weight
        self.layer_weight = layer_weight

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False) + 1e-8
        r_feature = torch.log(
            var ** (0.5) / (module.running_var.data.type(var.type()) + 1e-8) ** (0.5)).mean() - 0.5 * (1.0 - (
                    module.running_var.data.type(var.type()) + 1e-8 + (
                        module.running_mean.data.type(var.type()) - mean) ** 2) / var).mean()

        self.r_feature = r_feature

    def close(self):
        self.hook.remove()
