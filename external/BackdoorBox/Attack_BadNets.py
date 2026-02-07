"""
使用 VisualModelPoisoningResearchTool 提供的 CIFAR10 数据集，训练 BackdoorBox 的 BadNets 攻击。

与 Attack_ISSBA.py 一样，本脚本将数据路径对齐到：
    VisualModelPoisoningResearchTool/data/raw_datasets/cifar10

并调用 core.BadNets 完成：
1. 构造带固定触发器的 BadNets 攻击；
2. 先训练干净模型（benign_training=True，可按需关闭）；
3. 再训练带后门的模型（benign_training=False）。
"""

import os
from pathlib import Path

import cv2
import torch
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip
from torch.utils.data import Subset
import torchvision

import sys

# 确保能够导入 BackdoorBox 的 core 包
BACKDOORBOX_ROOT = Path(__file__).resolve().parents[0]
if str(BACKDOORBOX_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKDOORBOX_ROOT))

import core


def get_project_root() -> Path:
    """
    返回 VisualModelPoisoningResearchTool 项目根目录。
    结构：
      VisualModelPoisoningResearchTool/
        ├─ external/BackdoorBox/Attack_BadNets.py  <- 本文件
        └─ data/raw_datasets/cifar10/...
    """
    return Path(__file__).resolve().parents[3]


def main():
    project_root = get_project_root()
    data_root = project_root / "data" / "raw_datasets" / "cifar10"

    if not data_root.exists():
        raise FileNotFoundError(
            f"找不到 CIFAR10 数据目录: {data_root}\n"
            "请确认 VisualModelPoisoningResearchTool/data/raw_datasets/cifar10 已存在。"
        )

    dataset = torchvision.datasets.DatasetFolder

    # 图像: cv2.imread -> numpy(H x W x C) -> ToTensor -> (C x H x W) -> RandomHorizontalFlip
    transform_train = Compose([
        ToTensor(),
        RandomHorizontalFlip()
    ])

    # 为了兼顾资源，这里只取部分样本做训练（例如 20000 张）
    max_train_samples = 20000

    trainset_full = dataset(
        root=str(data_root),
        loader=cv2.imread,
        extensions=('png',),
        transform=transform_train,
        target_transform=None,
        is_valid_file=None,
    )

    total = len(trainset_full)
    if total == 0:
        raise RuntimeError(f"CIFAR10 数据集目录 {data_root} 中没有找到任何图像文件。")

    if total > max_train_samples:
        indices = list(range(total))
        torch.random.manual_seed(666)
        selected = indices[:max_train_samples]
        trainset = Subset(trainset_full, selected)
        print(f"[BadNets] CIFAR10 总样本 {total}，选取 {max_train_samples} 张用于训练。")
    else:
        trainset = trainset_full
        print(f"[BadNets] CIFAR10 总样本 {total}，全部用于训练。")

    transform_test = Compose([
        ToTensor()
    ])
    # 这里为简化，测试集也直接用相同目录，只是去掉数据增强
    testset = dataset(
        root=str(data_root),
        loader=cv2.imread,
        extensions=('png',),
        transform=transform_test,
        target_transform=None,
        is_valid_file=None,
    )

    # 构造 BadNets 触发器（右下角白色 3x3 方块）
    pattern = torch.zeros((1, 32, 32), dtype=torch.uint8)
    pattern[0, -3:, -3:] = 255
    weight = torch.zeros((1, 32, 32), dtype=torch.float32)
    weight[0, -3:, -3:] = 1.0

    badnets = core.BadNets(
        train_dataset=trainset,
        test_dataset=testset,
        model=core.models.ResNet(18),
        loss=nn.CrossEntropyLoss(),
        y_target=0,
        poisoned_rate=0.1,
        pattern=pattern,
        weight=weight,
        schedule=None,
        seed=666,
    )

    # 先查看一下中毒数据集情况（可选）
    poisoned_train_dataset, poisoned_test_dataset = badnets.get_poisoned_dataset()
    print(f"[BadNets] 中毒训练集大小: {len(poisoned_train_dataset)}, 中毒测试集大小: {len(poisoned_test_dataset)}")

    # 训练干净模型（如不需要可注释掉这一段）
    benign_schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': '0',
        'GPU_num': 1,

        'benign_training': True,
        'batch_size': 128,
        'num_workers': 8,

        'lr': 0.1,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'gamma': 0.1,
        'schedule': [150, 180],

        'epochs': 50,  # 为加快实验，默认 50 epoch

        'log_iteration_interval': 100,
        'test_epoch_interval': 10,
        'save_epoch_interval': 10,

        'save_dir': 'experiments',
        'experiment_name': 'BadNets_CIFAR10_benign',
    }

    print("[BadNets] 开始训练干净模型（可根据需要关闭此阶段）...")
    badnets.train(benign_schedule)

    # 训练带后门的模型
    poison_schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': '0',
        'GPU_num': 1,

        'benign_training': False,
        'batch_size': 128,
        'num_workers': 8,

        'lr': 0.1,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'gamma': 0.1,
        'schedule': [150, 180],

        'epochs': 50,  # 同样缩短训练时间

        'log_iteration_interval': 100,
        'test_epoch_interval': 10,
        'save_epoch_interval': 10,

        'save_dir': 'experiments',
        'experiment_name': 'BadNets_CIFAR10_poisoned',
    }

    print("[BadNets] 开始训练后门模型...")
    badnets.train(poison_schedule)
    print("[BadNets] 训练完成，模型与日志已保存至 external/BackdoorBox/experiments 下。")


if __name__ == "__main__":
    main()