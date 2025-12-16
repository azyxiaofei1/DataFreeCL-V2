from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from yacs.config import CfgNode as CN

_C = CN()

# ----- BASIC SETTINGS -----
_C.NAME = "zeroShotKD_default"
_C.OUTPUT_DIR = "./output"
_C.VALID_STEP = 20
_C.SAVE_STEP = 20
_C.SHOW_STEP = 100
_C.INPUT_SIZE = (32, 32)
_C.COLOR_SPACE = "RGB"
_C.CPU_MODE = False
_C.use_best_model = False
_C.availabel_cudas = "0"
_C.task1_MODEL = ""
_C.student_MODEL = ""
_C.save_model = False
_C.use_Contra_train_transform = False
_C.train_first_task = False
_C.seed = 0
_C.trainer_name = "DeepInvert_KD"
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

#----- model -----'''
_C.student = CN()
_C.student.use_cls = True
_C.student.use_kd = True
_C.student.temp = 2.
#----- TRAIN -----'''
_C.student.TRAIN = CN()
_C.student.TRAIN.MAX_EPOCH = 120
_C.student.TRAIN.BATCH_SIZE = 128
_C.student.TRAIN.SHUFFLE = True
_C.student.TRAIN.NUM_WORKERS = 4
# ----- OPTIMIZER -----
_C.student.TRAIN.OPTIMIZER = CN()
_C.student.TRAIN.OPTIMIZER.BASE_LR = 0.1
_C.student.TRAIN.OPTIMIZER.TYPE = "SGD"
_C.student.TRAIN.OPTIMIZER.MOMENTUM = 0.9
_C.student.TRAIN.OPTIMIZER.WEIGHT_DECAY = 1e-4
# ----- LR_SCHEDULER -----
_C.student.TRAIN.LR_SCHEDULER = CN()
_C.student.TRAIN.LR_SCHEDULER.TYPE = "multistep"
_C.student.TRAIN.LR_SCHEDULER.LR_STEP = [30, 60]
_C.student.TRAIN.LR_SCHEDULER.LR_FACTOR = 0.1
_C.student.TRAIN.LR_SCHEDULER.WARM_EPOCH = 5

_C.generator = CN()
_C.generator.gen_model_name = "CIFAR_GEN"
_C.generator.generator_iter = 5000
_C.generator.batch_size = 128
_C.generator.deep_inv_params = [0.1, 1., 1., 1.]

_C.generator.generator_epochs = 100
_C.generator.generator_lr = 0.001
_C.generator.latent_size = 256

_C.generator.TRAIN = CN()
_C.generator.TRAIN.BATCH_SIZE = 64
_C.generator.TRAIN.NUM_WORKERS = 4

_C.discriminator = CN()
_C.discriminator.dis_model_name = "CIFAR_DIS"
_C.discriminator.discriminator_lr = 0.001

# self.generator_lr = deep_inv_params[0]
#         self.r_feature_weight = deep_inv_params[1]
#         self.pr_scale = deep_inv_params[2]
#         self.content_temp = deep_inv_params[3]
