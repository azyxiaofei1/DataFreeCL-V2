from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from yacs.config import CfgNode as CN

_C = CN()

# ----- BASIC SETTINGS -----
_C.NAME = "CIL_default"
_C.OUTPUT_DIR = "./output"
_C.VALID_STEP = 20
_C.SAVE_STEP = 20
_C.SHOW_STEP = 100
_C.INPUT_SIZE = (32, 32)
_C.COLOR_SPACE = "RGB"
_C.CPU_MODE = False
_C.use_best_model = False
_C.availabel_cudas = "3"
_C.use_base_half = False
_C.checkpoints = "./"
_C.task1_MODEL = ""
_C.save_model = False
_C.use_Contra_train_transform = False
_C.approach = "FeTrIL"
_C.train_first_task = False
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

# ----- resume -----
_C.RESUME = CN()
_C.RESUME.use_resume = False
_C.RESUME.resumed_file = ""
_C.RESUME.resumed_model_path = ""
_C.RESUME.resumed_bias_layer_path = ""

# ----- classifier BUILDER -----'''
_C.classifier = CN()
_C.classifier.bias = True
_C.classifier.classifier_type = "linear"
_C.classifier.LOSS_TYPE = "CrossEntropy"

''' ----- BACKBONE BUILDER -----'''
_C.extractor = CN()
_C.extractor.TYPE = "resnet34"
_C.extractor.rate = 1.
_C.extractor.output_feature_dim = 512

#----- model -----'''
_C.model = CN()
_C.model.EPSILON = 1e-6
_C.model.shrink = True
_C.model.norm_cov = True
_C.model.alpha1 = 1.
_C.model.alpha2 = 1.
_C.model.beta = 0.5

_C.model.eeil_finetune_train = CN()
_C.model.eeil_finetune_train.BATCH_SIZE = 128
_C.model.eeil_finetune_train.MAX_EPOCH = 50
_C.model.eeil_finetune_train.NUM_WORKERS = 4
_C.model.eeil_finetune_train.SHUFFLE = True

