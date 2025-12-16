from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from yacs.config import CfgNode as CN

_C = CN()

# ----- BASIC SETTINGS -----
_C.NAME = "NAPAVQ_default"
_C.OUTPUT_DIR = "./output"
_C.VALID_STEP = 20
_C.SAVE_STEP = 20
_C.SHOW_STEP = 100
_C.INPUT_SIZE = (32, 32)
_C.COLOR_SPACE = "RGB"
_C.CPU_MODE = False
_C.use_best_model = False
_C.availabel_cudas = "1"
_C.use_base_half = False
_C.checkpoints = "./"
_C.task1_MODEL = ""
_C.task1_navq = ""
_C.save_model = False
_C.use_Contra_train_transform = False
_C.approach = "NAPA-VQ"
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

# ----- pre-train setting -----
_C.PRETRAINED = CN()
_C.PRETRAINED.use_pretrained_model = False
_C.PRETRAINED.MODEL = ""

''' ----- BACKBONE BUILDER -----'''
_C.extractor = CN()
_C.extractor.TYPE = "resnet34"
_C.extractor.rate = 1.
_C.extractor.output_feature_dim = 512

# ----- TRAIN -----'''
# ----- model -----'''
_C.model = CN()
_C.model.protoAug_weight = 1.
_C.model.kd_weight = 1.
_C.model.temp = 10.

_C.model.TRAIN = CN()
_C.model.TRAIN.MAX_EPOCH = 120
_C.model.TRAIN.BATCH_SIZE = 128
_C.model.TRAIN.SHUFFLE = True
_C.model.TRAIN.NUM_WORKERS = 4
# ----- OPTIMIZER -----
_C.model.TRAIN.OPTIMIZER = CN()
_C.model.TRAIN.OPTIMIZER.BASE_LR = 0.1
_C.model.TRAIN.OPTIMIZER.TYPE = "SGD"
_C.model.TRAIN.OPTIMIZER.MOMENTUM = 0.9
_C.model.TRAIN.OPTIMIZER.WEIGHT_DECAY = 1e-4

# ----- LR_SCHEDULER -----
_C.model.TRAIN.LR_SCHEDULER = CN()
_C.model.TRAIN.LR_SCHEDULER.TYPE = "step"
_C.model.TRAIN.LR_SCHEDULER.step_size = 45
_C.model.TRAIN.LR_SCHEDULER.LR_STEP = [30, 60]
_C.model.TRAIN.LR_SCHEDULER.LR_FACTOR = 0.1
_C.model.TRAIN.LR_SCHEDULER.WARM_EPOCH = 5

# ----- navq -----
_C.model.navq = CN()

_C.model.navq.TRAIN = CN()
# ----- navq OPTIMIZER -----
_C.model.navq.TRAIN.OPTIMIZER = CN()
_C.model.navq.TRAIN.OPTIMIZER.BASE_LR = 0.1
_C.model.navq.TRAIN.OPTIMIZER.TYPE = "SGD"
_C.model.navq.TRAIN.OPTIMIZER.MOMENTUM = 0.9
_C.model.navq.TRAIN.OPTIMIZER.WEIGHT_DECAY = 1e-4
# ----- navq LR_SCHEDULER -----
_C.model.navq.TRAIN.LR_SCHEDULER = CN()
_C.model.navq.TRAIN.LR_SCHEDULER.TYPE = "step"
_C.model.navq.TRAIN.LR_SCHEDULER.step_size = 45
_C.model.navq.TRAIN.LR_SCHEDULER.LR_STEP = [30, 60]
_C.model.navq.TRAIN.LR_SCHEDULER.LR_FACTOR = 0.1
_C.model.navq.TRAIN.LR_SCHEDULER.WARM_EPOCH = 5



