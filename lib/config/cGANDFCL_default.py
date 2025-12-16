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
_C.availabel_cudas = "2"
_C.use_base_half = False
_C.checkpoints = "./"
_C.task1_MODEL = ""
_C.pre_model = ""
_C.pretrained_MODEL = ""
_C.save_model = False
_C.use_Contra_train_transform = False
_C.train_first_task = False
_C.seed = 0
_C.trainer_name = "cGAN_gen"
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

#----- model -----'''
_C.model = CN()
_C.model.loss_type = "LwF"
_C.model.kd_lambda = 1.
_C.model.T = 2.
#----- TRAIN -----'''
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

_C.generator = CN()
_C.generator.gen_model_name = "CIFAR_GAN"
_C.generator.generator_epochs = 100
_C.generator.generator_lr = 0.001
_C.generator.latent_size = 256

_C.generator.TRAIN = CN()
_C.generator.TRAIN.BATCH_SIZE = 64
_C.generator.TRAIN.NUM_WORKERS = 4

_C.discriminator = CN()
_C.discriminator.dis_model_name = "CIFAR_DIS"
_C.discriminator.discriminator_lr = 0.001