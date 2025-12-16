from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from yacs.config import CfgNode as CN

_C = CN()

# ----- BASIC SETTINGS -----
_C.NAME = "DFCIL_default"
_C.OUTPUT_DIR = "./output"
_C.VALID_STEP = 20
_C.SAVE_STEP = 20
_C.SHOW_STEP = 100
_C.INPUT_SIZE = (32, 32)
_C.COLOR_SPACE = "RGB"
_C.CPU_MODE = False
_C.availabel_cudas = "3"
_C.task1_MODEL = ""
_C.fctm_model_path = ""
_C.seed = 0
# ----- DATASET BUILDER -----
_C.DATASET = CN()
_C.DATASET.dataset_name = "CIFAR100"  # mnist, mnist28, CIFAR10, CIFAR100, imagenet, svhn
_C.DATASET.dataset = "Torchvision_Datasets_Split"
_C.DATASET.data_json_file = ""
_C.DATASET.data_root = "./datasets"
_C.DATASET.all_classes = 100
_C.DATASET.all_tasks = 10
_C.DATASET.split_seed = 0
_C.DATASET.val_length = 0
_C.DATASET.use_svhn_extra = True

''' ----- BACKBONE BUILDER -----'''
_C.extractor = CN()
_C.extractor.TYPE = "resnet34"
_C.extractor.rate = 1.
_C.extractor.output_feature_dim = 512

# ----- model -----'''
_C.model = CN()

# ----- TRAIN -----'''
_C.model.TRAIN = CN()
_C.model.TRAIN.MAX_EPOCH = 120
_C.model.TRAIN.BATCH_SIZE = 128
_C.model.TRAIN.SHUFFLE = True
_C.model.TRAIN.NUM_WORKERS = 4

# ----- FCTM_model -----'''
_C.FCTM = CN()

# ----- FCN -----
_C.FCTM.use_FCN = True
_C.FCTM.FCN = CN()
_C.FCTM.FCN.in_feature_dim = 512
_C.FCTM.FCN.out_feature_dim = 512
_C.FCTM.FCN.layer_nums = 2
_C.FCTM.FCN.hidden_layer_rate = 2
_C.FCTM.FCN.last_hidden_layer_use_relu = False
# ----- TRAIN -----'''
_C.FCTM.TRAIN = CN()
_C.FCTM.TRAIN.tradeoff_rate = 1.
_C.FCTM.TRAIN.MAX_EPOCH = 120
_C.FCTM.TRAIN.BATCH_SIZE = 128
_C.FCTM.TRAIN.SHUFFLE = True
_C.FCTM.TRAIN.NUM_WORKERS = 4

