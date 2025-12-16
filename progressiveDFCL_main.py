import sys

sys.path.append("../DataFreeCL-V2")
from lib.approach.MI_DFCL import MI_DFCL_handler#主题算法
from lib.dataset import *#数据集划分/加载
from lib.config import progressive_DFCL_cfg, update_config#配置系统
from lib.utils.utils import (#日志/工具
    create_logger,
)
import torch
import os
import argparse
import warnings


def parse_args():
    parser = argparse.ArgumentParser(description="codes for EARS-DFCL")

    parser.add_argument(
        "--cfg",
        help="decide which cfg to use",
        required=False,
        # default="./configs/progressive_DFCL_cifar10.yaml",
        default="./configs/progressive_DFCL_cifar100.yaml",
        # default="./configs/progressive_DFCL_tiny.yaml",
        # default="./configs/progressive_DFCL_imagenet.yaml",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    update_config(progressive_DFCL_cfg, args)
    logger, log_file = create_logger(progressive_DFCL_cfg, "log")#创建日志系统
    warnings.filterwarnings("ignore")#把 Python 的各种 UserWarning、FutureWarning 等直接忽略掉。目的就是训练时屏幕别被一堆 warning 淹没，日志更干净。
    # split_seleted_data = {0: [52, 1, 30, 96], 1: [4, 5, 6, 7], 2: [8, 9, 10, 11], 3: [12, 13, 14, 15], 4: [16, 17, 18, 19], 5: [20, 21, 22, 23], 6: [24, 25, 26, 27], 7: [28, 29, 2, 31], 8: [32, 33, 34, 35], 9: [36, 37, 38, 39], 10: [40, 41, 42, 43], 11: [44, 45, 46, 47], 12: [48, 49, 50, 51], 13: [0, 53, 54, 55], 14: [56, 57, 58, 59], 15: [60, 61, 62, 63], 16: [64, 65, 66, 67], 17: [68, 69, 70, 71], 18: [72, 73, 74, 75], 19: [76, 77, 78, 79], 20: [80, 81, 82, 83], 21: [84, 85, 86, 87], 22: [88, 89, 90, 91], 23: [92, 93, 94, 95], 24: [3, 97, 98, 99]}
    split_seleted_data = None#（可选）手动指定任务划分 / 随机划分
    dataset_split_handler = eval(progressive_DFCL_cfg.DATASET.dataset)(progressive_DFCL_cfg, split_selected_data=split_seleted_data)#实例化“数据集划分器”（Torchvision_Datasets_Split）
    """
    dataset_split_handler = “专门负责：把 CIFAR100 切成 task0, task1, ... 这些 dataset 的小管家”，而这个管家本身的类型就是 Torchvision_Datasets_Split。
    
    在这段代码里，eval 的目的就是：
    把 cfg 里这个字符串 "Torchvision_Datasets_Split" 解析成同名的类对象，然后立刻用它来创建一个实例。
    
    progressive_DFCL_cfg.DATASET.dataset 在你的 yaml 里是 "Torchvision_Datasets_Split"
    dataset_split_handler = Torchvision_Datasets_Split(
    progressive_DFCL_cfg,
    split_selected_data=None
)

    """
    if progressive_DFCL_cfg.availabel_cudas:
        os.environ['CUDA_VISIBLE_DEVICES'] = progressive_DFCL_cfg.availabel_cudas
        device_ids = [i for i in range(len(progressive_DFCL_cfg.availabel_cudas.strip().split(',')))]
        print(device_ids)

    midfcl_handler = MI_DFCL_handler(dataset_split_handler, progressive_DFCL_cfg, logger)
    midfcl_handler.midfcl_train_main()#(MI_DFCL_handler: 主题算法训练主函数)
    """
    拿着 dataset_split_handler + cfg + logger，按 task 一步一步把整篇论文的训练流程跑完。
    
    1、命令行 → args
    
    2、默认 config（_C） + yaml + opts → cfg
    
    3、cfg → 数据划分器 Torchvision_Datasets_Split → 每个 task 的 dataset
    
    4、cfg + dataset_handler + logger → MI_DFCL_handler
    
    5、MI_DFCL_handler 里创建具体 trainer（DFCL 算法实现）
    
    6、按 task 循环：构造 train_dataset → 调 trainer 训练/评估 → 保存模型
    """
