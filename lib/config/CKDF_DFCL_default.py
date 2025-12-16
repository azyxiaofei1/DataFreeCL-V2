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
_C.use_best_model = False
_C.availabel_cudas = "3"
_C.use_base_half = False
_C.checkpoints = "./"
_C.task1_MODEL = ""
_C.pretrained_MODEL = ""
_C.save_model = False
_C.use_Contra_train_transform = False
_C.train_first_task = False
_C.seed = 0
_C.trainer_name = "CKDF_DFCL"
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
_C.RESUME.resumed_model_path = ""

# ----- pre-train setting -----
_C.PRETRAINED = CN()
_C.PRETRAINED.use_pretrained_model = False
_C.PRETRAINED.MODEL = ""

''' ----- BACKBONE BUILDER -----'''
_C.extractor = CN()
_C.extractor.TYPE = "resnet34"
_C.extractor.rate = 1.
_C.extractor.output_feature_dim = 512

_C.generator = CN()
_C.generator.gen_model_name = "CIFAR_GEN"
_C.generator.generator_iter = 5000
_C.generator.batch_size = 128
_C.generator.deep_inv_params = [0.1, 1., 1., 1.]

# ----- model -----'''
_C.model = CN()

# logit KD:
_C.model.T = 2.
_C.model.kd_lambda = 1.
_C.model.fkd_lambda = 1e-1

# ----- TRAIN -----'''
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
_C.model.TRAIN.LR_SCHEDULER.TYPE = "multistep"
_C.model.TRAIN.LR_SCHEDULER.LR_STEP = [30, 60]
_C.model.TRAIN.LR_SCHEDULER.LR_FACTOR = 0.1
_C.model.TRAIN.LR_SCHEDULER.WARM_EPOCH = 5

# ----- finetune -----
_C.model.finetune = CN()
_C.model.finetune.MAX_EPOCH = 120
_C.model.finetune.BATCH_SIZE = 128
_C.model.finetune.SHUFFLE = True
_C.model.finetune.NUM_WORKERS = 4
# ----- OPTIMIZER -----
_C.model.finetune.BASE_LR = 0.1
_C.model.finetune.TYPE = "SGD"
_C.model.finetune.MOMENTUM = 0.9
_C.model.finetune.WEIGHT_DECAY = 1e-4
# ----- LR_SCHEDULER -----
_C.model.finetune.LR_TYPE = "multistep"
_C.model.finetune.LR_STEP = [30, 60]
_C.model.finetune.LR_FACTOR = 0.1
_C.model.finetune.WARM_EPOCH = 5

# ----- FCTM_model -----'''
_C.FCTM = CN()

# logit KD:
_C.FCTM.T = 2.
_C.FCTM.kd_lambda = 1.

# ----- FCN -----
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
# ----- OPTIMIZER -----
_C.FCTM.TRAIN.OPTIMIZER = CN()
_C.FCTM.TRAIN.OPTIMIZER.BASE_LR = 0.1
_C.FCTM.TRAIN.OPTIMIZER.TYPE = "SGD"
_C.FCTM.TRAIN.OPTIMIZER.MOMENTUM = 0.9
_C.FCTM.TRAIN.OPTIMIZER.WEIGHT_DECAY = 1e-4
# ----- LR_SCHEDULER -----
_C.FCTM.TRAIN.LR_SCHEDULER = CN()
_C.FCTM.TRAIN.LR_SCHEDULER.TYPE = "multistep"
_C.FCTM.TRAIN.LR_SCHEDULER.LR_STEP = [30, 60]
_C.FCTM.TRAIN.LR_SCHEDULER.LR_FACTOR = 0.1
_C.FCTM.TRAIN.LR_SCHEDULER.WARM_EPOCH = 5
