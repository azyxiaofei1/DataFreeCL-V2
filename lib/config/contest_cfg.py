from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from yacs.config import CfgNode as CN

_C = CN()

# ----- BASIC SETTINGS -----
_C.NAME = "DFCIL_default"
_C.OUTPUT_DIR = "./contest_output"
_C.VALID_STEP = 20
_C.SAVE_STEP = 20
_C.SHOW_STEP = 100
_C.INPUT_SIZE = (224, 224)
_C.COLOR_SPACE = "RGB"
_C.CPU_MODE = False
_C.availabel_cudas = "7"
_C.seed = 0
# ----- DATASET BUILDER -----
_C.DATASET = CN()
_C.DATASET.dataset_name = "CIFAR100"  # mnist, mnist28, CIFAR10, CIFAR100, imagenet, svhn
_C.DATASET.data_root = "./datasets"
_C.DATASET.all_classes = 100
_C.DATASET.all_tasks = 10
_C.DATASET.split_seed = 0
_C.DATASET.classes_per_task = 10

''' ----- BACKBONE BUILDER -----'''
_C.extractor = CN()
_C.extractor.TYPE = "resnet50"
_C.extractor.rate = 1.
_C.extractor.output_feature_dim = 512

_C.generator = CN()
_C.generator.gen_model_name = "IMNET_GEN"
_C.generator.generator_iter = 10000
_C.generator.batch_size = 128
_C.generator.deep_inv_params = [0.1, 1., 1., 1.]

# ----- model -----'''
_C.model = CN()

# cls loss
_C.model.softTarget_lambda = 1.
_C.model.cls_type = "softTarget"

# logit KD:
_C.model.T = 2.
_C.model.KD_type = "ReKD"
_C.model.kd_lambda = 1.

# feature KD
_C.model.fkd_lambda = 1e-1
_C.model.use_featureKD = False

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

# ----- FCTM_model -----'''
_C.FCTM = CN()
# _C.FCTM.T = 2.
# _C.FCTM.kd_lambda = 1.
# _C.FCTM.addCls_lambda = 1.
#
# _C.FCTM.use_SSCE = False
# _C.FCTM.use_KD = True
# _C.FCTM.use_add_cls = True
# _C.FCTM.use_binary_KD = False
# _C.FCTM.all_cls_LOSS_TYPE = "CE"
# _C.FCTM.add_cls_LOSS_TYPE = "CE"

# classification loss:
_C.FCTM.addCls_lambda = 1.
_C.FCTM.ce_lambda = 1.
_C.FCTM.all_cls_LOSS_TYPE = "CE"
_C.FCTM.add_cls_LOSS_TYPE = "CE"

# logit KD:
_C.FCTM.T = 2.
_C.FCTM.hkd_lambda = 1.
_C.FCTM.kd_lambda = 1.

# rkd:
_C.FCTM.rkd_lambda = 1.

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
