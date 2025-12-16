from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from yacs.config import CfgNode as CN

#带有_C.BACKBONE / _C.CLASSIFIER / _C.exemplar_manager
_C = CN()

# ----- BASIC SETTINGS -----
_C.NAME = "default"
_C.OUTPUT_DIR = "./output"
_C.VALID_STEP = 20
_C.SAVE_STEP = 20
_C.SHOW_STEP = 100
_C.INPUT_SIZE = (224, 224)
_C.COLOR_SPACE = "RGB"
_C.CPU_MODE = False


# ----- DATASET BUILDER -----
_C.DATASET = CN()
_C.DATASET.dataset_name = "CIFAR100"        #mnist, mnist28, CIFAR10, CIFAR100, imagenet, svhn
_C.DATASET.dataset = "Torchvision_Datasets_Split"
_C.DATASET.data_root = "./datasets"
_C.DATASET.all_classes = 100#类别总数 = 100。
_C.DATASET.all_tasks = 10#默认 10 个 task。
_C.DATASET.split_seed = 0#划分类到各 task 的随机种子。
_C.DATASET.val_length = 0#验证集大小（0 可能表示不单独划验证集）。
_C.DATASET.AUTO_ALPHA = True#和分类器自适应因子有关（下面有一个 _C.AUTO_ALPHA）
_C.DATASET.use_svhn_extra = True#用于 SVHN 数据集时，是否使用 extra 子集。


# cfg.exemplar_manager.memory_budget, cfg.exemplar_manager.mng_approach,
#                                        cfg.exemplar_manager.store_original_imgs, cfg.exemplar_manager.norm_exemplars,
#                                        cfg.exemplar_manager.centroid_order

#样本记忆管理器（这是“有 exemplar”那条线）这块就是“长期记住的 exemplar 如何管理”。
# ----- exemplar_manager -----
_C.exemplar_manager = CN()
_C.exemplar_manager.store_original_imgs = True #存的是原始图像（而不是某种压缩特征）。
_C.exemplar_manager.fixed_exemplar_num = -1 #如果是正数，比如 20 就代表每类固定存 20 个。-1 通常表示“不固定，看 memory_budget 如何分配”。
_C.exemplar_manager.memory_budget = 2000 #总 exemplar 数上限，比如总共只允许存 2000 张代表样本。
_C.exemplar_manager.mng_approach = "herding" #决定“从每个类的训练样本中挑哪些进 exemplar 集”。"herding"：经典的 herding 选择策略（选出最接近类均值的一批点）。
_C.exemplar_manager.norm_exemplars = True #是否对 exemplar 特征做归一化（有些 NME/LDA 分类器会用 L2 norm）
_C.exemplar_manager.centroid_order = "herding" #类中心更新顺序，也用 herding 的策略。
#这一块在 DFCL 的配置里是没有的，说明这份 config 是给「有 memory」的方法用的。

# ----- resume -----
_C.RESUME = CN()
_C.RESUME.use_resume = False #是否使用 resume 功能。
_C.RESUME.resumed_file = "" #resume 的文件路径。
_C.RESUME.resumed_model_path = "" #resume 的模型路径。
_C.RESUME.resumed_pre_tasks_model = "" #resume 的预训练模型路径。

#分类器自动权重相关（和 DATASET.AUTO_ALPHA 对应）
# ----- CLASSIFIER AUTO ALPHA -----
_C.AUTO_ALPHA = CN()
_C.AUTO_ALPHA.ALPHA = -1. #-1 多半表示“自动计算”，否则你可以指定一个固定值。
_C.AUTO_ALPHA.LENGTH = 100 #可能是平滑窗口长度，或采样多少个 batch 估计 alpha。
_C.AUTO_ALPHA.GAMMA = 1. #调节系数，具体看 auto-alpha 模块实现。

_C.AUTO_ALPHA.LOSS0_FACTOR = 1. #如果有两个 loss 分量（比如 CE + 某个正则），这俩就是权重。
_C.AUTO_ALPHA.LOSS1_FACTOR = 1.

#网络构成
# ----- BACKBONE BUILDER -----
_C.BACKBONE = CN()
_C.BACKBONE.TYPE = "resnext50" #默认 backbone 是 ResNeXt-50。
_C.BACKBONE.PRETRAINED_BACKBONE = "" #如果有预训练权重，可以填路径。

# ----- MODULE BUILDER -----
_C.MODULE = CN()
_C.MODULE.TYPE = "GAP" #表示中间模块是全局平均池化（Global Average Pooling），即 ResNet 后面常见那一层。

# ----- CLASSIFIER BUILDER -----
_C.CLASSIFIER = CN()
_C.CLASSIFIER.TYPE = "LDA" # 说明最后的分类器不是普通的 FC 层，而是 LDA 类的（线性判别分析）这种在 exemplar-based iCaRL 系列增量学习里很常见。
_C.CLASSIFIER.BIAS = True #是否带 bias。
#这里是 “neck 模块”（类似分类器前的一个投影头）：
_C.CLASSIFIER.NECK = CN()
_C.CLASSIFIER.NECK.ENABLE = True #启用 neck。
_C.CLASSIFIER.NECK.TYPE = 'Linear' #neck 类型，这里是线性投影。
_C.CLASSIFIER.NECK.NUM_FEATURES = 2048 #backbone 输出的特征维度。
_C.CLASSIFIER.NECK.NUM_OUT = 128 #neck 输出维度。
_C.CLASSIFIER.NECK.HIDDEN_DIM = 512 #如果 neck 是 MLP，这就是中间层维度。
#一堆与度量学习相关的开关
_C.CLASSIFIER.NECK.MARGIN = 1.0
_C.CLASSIFIER.NECK.WEIGHT_INTER_LOSS = False #是否对类间/类内损失加权。
_C.CLASSIFIER.NECK.WEIGHT_INTRA_LOSS = False
_C.CLASSIFIER.NECK.INTER_DISTANCE = True#是否计算类间/类内距离损失。
_C.CLASSIFIER.NECK.INTRA_DISTANCE = True
_C.CLASSIFIER.NECK.LOSS_FACTOR = 0.5#neck 相关损失的权重。

#知识蒸馏设置
# ----- DISTILL -----
_C.DISTILL = CN()
_C.DISTILL.ENABLE = True #整体是否启用蒸馏。
_C.DISTILL.CLS_DISTILL_ENABLE = False #是否对分类 logits 做蒸馏（False 可能表示用 feature-level distill 或其它）。
_C.DISTILL.LOSS_FACTOR = 1. #蒸馏损失权重。
_C.DISTILL.softmax_sigmoid = 0
_C.DISTILL.TEMP = 2. #蒸馏温度。

#总的 loss 类型
# ----- LOSS BUILDER -----
_C.LOSS = CN()
_C.LOSS.LOSS_TYPE = "CrossEntropy" #主损失类型是交叉熵（CrossEntropy）。

#训练设置（全局）
# ----- TRAIN BUILDER -----
_C.TRAIN = CN()
_C.TRAIN.BATCH_SIZE = 128 #批大小。
_C.TRAIN.MAX_EPOCH = 90 #最大 epoch。
_C.TRAIN.SHUFFLE = True #是否打乱数据。
_C.TRAIN.NUM_WORKERS = 4 #数据加载线程数。
_C.TRAIN.TENSORBOARD = CN() #tensorboard 相关设置。
_C.TRAIN.TENSORBOARD.ENABLE = True #是否启用 tensorboard。
_C.TRAIN.SUM_GRAD = False #是否打印梯度。


_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.TYPE = "SGD"
_C.TRAIN.OPTIMIZER.BASE_LR = 0.001 #优化器：SGD，lr=0.001（比之前 DFCL 那套 0.1 小很多）。
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9
_C.TRAIN.OPTIMIZER.WEIGHT_DECAY = 1e-4

_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.TYPE = "multistep"
_C.TRAIN.LR_SCHEDULER.LR_STEP = [30, 60] #在 epoch 30 和 60 时乘 0.1；
_C.TRAIN.LR_SCHEDULER.LR_FACTOR = 0.1
_C.TRAIN.LR_SCHEDULER.WARM_EPOCH = 5 #前 5 个 epoch 做 warmup；
_C.TRAIN.LR_SCHEDULER.COSINE_DECAY_END = 0  #cosine 退火结束 epoch。


def update_config(cfg, args):
    cfg.defrost() #yacs 的 CfgNode 默认可能是“冻结”的（不允许随便加改字段，为的是防止误拼字段名）。defrost()：解除冻结 → 可以改。
    cfg.merge_from_file(args.cfg) #args.cfg 就是你在命令行里给的 --cfg 参数，比如：
    #打开这个 YAML；按 YAML 里的层级，把 _C 里对应字段覆盖掉。
    cfg.merge_from_list(args.opts)#args.opts 是命令行里 YAML 外面“多余的所有参数”，例如：
    """
    那么：
        python main.py --cfg xxx.yaml DATASET.all_tasks 5 extractor.TYPE res32_cifar

        args.cfg = "xxx.yaml"        
        args.opts = ["DATASET.all_tasks", "5", "extractor.TYPE", "res32_cifar"]
    
    merge_from_list 会再一次在 YAML 的基础上做覆盖：
    
        cfg.DATASET.all_tasks = 5
        cfg.extractor.TYPE = "res32_cifar"
        
    用 YAML 控整体结构；用命令行做小范围超参搜索，而不用每次改 YAML 文件。

    """

    # cfg.freeze()
