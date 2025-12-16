from __future__ import print_function

from .ABD import AlwaysBeDreaming
from .RDFCL import RDFCL
from .EARS_DFCL import EARS_DFCL
from .CKDF_DFCL import CKDF_DFCL
from .foster_DFCL import foster_DFCL
from .progressive_DFCL import progressive_DFCL
from .PCDFCL import PCDFCL
from .DEA_DFCL import DEA_DFCL
from .deepInvert_KD import DeepInvert_KD
from .cGAN_GEN import cGAN_gen
from .cGAN_KD import cGAN_KD
from .GAN_GEN import GAN_gen
from .GAN_KD import GAN_KD
from .DCMI import DualConsistencyMI

#修改前错误代码
#from .GTD import GTD
#修改后代码
#from ..approach.GTD_DFCL import GTD_DFCL as GTD # 使用 ..approach 跨包引用
__all__ = [
    'AlwaysBeDreaming',
    'RDFCL',
    'CKDF_DFCL',
    'foster_DFCL',
    'EARS_DFCL',
    'progressive_DFCL',
    'PCDFCL',
    'DEA_DFCL',
    'DeepInvert_KD',
    'cGAN_KD',
    'cGAN_gen',
    'GAN_KD',
    'GAN_gen',
    'DualConsistencyMI',
    #'GTD',  #注释掉了GTD  ！！！！！修改！！！
]
