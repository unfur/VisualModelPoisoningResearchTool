"""
ISSBA (Invisible Sample-Specific Backdoor Attack) 完整复现脚本

本脚本严格按照 BackdoorBox 官方 ISSBA 实现进行复现：

1. **数据集准备**：使用 CIFAR10 数据集，确保与 BackdoorBox support_list 兼容
2. **编码器训练**：训练 StegaStamp encoder/decoder，用于生成不可见的后门触发器
3. **中毒数据生成**：按照指定的 poisoned_rate 生成中毒样本
4. **受害者模型训练**：使用混合数据集（干净样本 + 中毒样本）训练后门分类器

关键复现要点：
- 严格按照 poisoned_rate=0.05 (5%) 的比例生成中毒样本
- 使用相同的 secret 对所有中毒样本进行编码
- 所有中毒样本的标签都设置为 y_target=1
- 训练时按照 poisoned_rate 比例混合干净样本和中毒样本

训练完成后会生成：
- encoder_decoder.pth：编码器和解码器权重
- ckpt_epoch_*.pth：后门分类器检查点
"""

import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets, transforms

import sys

# 确保能够导入 BackdoorBox 的 core 包（本文件位于 external/BackdoorBox/ 下）
BACKDOORBOX_ROOT = Path(__file__).resolve().parents[0]
if str(BACKDOORBOX_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKDOORBOX_ROOT))

import core
from core.attacks.ISSBA import ISSBA


# ----------------- 全局配置 -----------------
global_seed = 666
deterministic = False
torch.manual_seed(global_seed)
random.seed(global_seed)
np.random.seed(global_seed)

# ISSBA 标准参数（严格按照 BackdoorBox 官方测试文件和论文）
secret_size = 20          # ISSBA 论文标准参数
poisoned_rate = 0.05      # 5% 中毒率，ISSBA 论文标准
y_target = 1              # 目标标签，BackdoorBox 标准
batch_size = 128          # BackdoorBox 标准配置
num_workers = 8           # BackdoorBox 标准配置

# 为适配普通个人电脑，可以限制训练样本数量
# 设置为 None 使用全部数据，或设置具体数字限制样本数
max_train_samples = None  # 使用全部 CIFAR10 数据以确保完整复现


def get_project_root() -> Path:
    """
    返回项目根目录，用于存储 CIFAR10 数据集。
    CIFAR10 将自动下载到 data/raw_datasets/ 目录下。
    """
    return Path(__file__).resolve().parents[2]  # 返回到项目根目录


class StegDataset(Dataset):
    """
    给 ISSBA 训练 encoder/decoder 用的 dataset：
    返回 (image_tensor, secret_vector)。
    """

    def __init__(self, base_ds: Dataset, secret_size: int):
        self.base_ds = base_ds
        self.secret_size = secret_size

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        img, _ = self.base_ds[idx]  # CIFAR10 标签在这里只训练编码器时不需要
        # 为每张图生成一个随机 secret（0/1 比特）
        secret = np.random.binomial(1, 0.5, self.secret_size).astype(np.float32)
        return img, secret


class SubsetCIFAR10(datasets.CIFAR10):
    """
    CIFAR10 子集包装类，保持与 BackdoorBox support_list 的兼容性
    只在需要限制样本数量时使用
    """
    def __init__(self, cifar10_dataset, indices):
        # 继承 CIFAR10 的所有属性
        self.root = cifar10_dataset.root
        self.train = cifar10_dataset.train
        self.transform = cifar10_dataset.transform
        self.target_transform = cifar10_dataset.target_transform
        
        # 获取子集数据
        self.data = cifar10_dataset.data[indices]
        self.targets = [cifar10_dataset.targets[i] for i in indices]
        
        # 保持其他 CIFAR10 属性
        self.classes = cifar10_dataset.classes
        self.class_to_idx = cifar10_dataset.class_to_idx


def verify_issba_reproduction(issba, trainset, testset):
    """
    验证 ISSBA 复现的正确性
    """
    print("\n" + "=" * 60)
    print("验证 ISSBA 复现正确性")
    print("=" * 60)
    
    # 1. 验证中毒率
    total_train = len(trainset)
    poisoned_count = len(issba.poisoned_set)
    actual_poisoned_rate = poisoned_count / total_train
    
    print(f"[VERIFY] 训练集总数: {total_train}")
    print(f"[VERIFY] 中毒样本数: {poisoned_count}")
    print(f"[VERIFY] 实际中毒率: {actual_poisoned_rate:.3f}")
    print(f"[VERIFY] 预期中毒率: {poisoned_rate:.3f}")
    
    if abs(actual_poisoned_rate - poisoned_rate) < 0.001:
        print("[VERIFY] ✅ 中毒率设置正确")
    else:
        print("[VERIFY] ❌ 中毒率设置不正确")
    
    # 2. 验证目标标签
    print(f"[VERIFY] 目标标签: {issba.y_target}")
    if issba.y_target == y_target:
        print("[VERIFY] ✅ 目标标签设置正确")
    else:
        print("[VERIFY] ❌ 目标标签设置不正确")
    
    # 3. 验证编码器配置
    enc_config = issba.encoder_schedule
    print(f"[VERIFY] 编码器配置:")
    print(f"  - Secret 大小: {enc_config['secret_size']}")
    print(f"  - 图像尺寸: {enc_config['enc_height']}x{enc_config['enc_width']}")
    print(f"  - 输入通道: {enc_config['enc_in_channel']}")
    print(f"  - 总训练轮数: {enc_config['enc_total_epoch']}")
    print(f"  - Secret-only 轮数: {enc_config['enc_secret_only_epoch']}")
    
    # 4. 验证数据集类型
    print(f"[VERIFY] 数据集名称: {issba.dataset_name}")
    print(f"[VERIFY] 训练集类型: {type(trainset).__name__}")
    print(f"[VERIFY] 测试集类型: {type(testset).__name__}")
    
    if issba.dataset_name == "cifar10":
        print("[VERIFY] ✅ 数据集类型正确")
    else:
        print("[VERIFY] ❌ 数据集类型不正确")
    
    # 5. 验证随机种子
    print(f"[VERIFY] 随机种子: {issba.seed}")
    if issba.seed == global_seed:
        print("[VERIFY] ✅ 随机种子设置正确")
    else:
        print("[VERIFY] ❌ 随机种子设置不正确")
    
    print("=" * 60)


def main():
    project_root = get_project_root()
    data_root = project_root / "data" / "raw_datasets"

    print("=" * 60)
    print("ISSBA (Invisible Sample-Specific Backdoor Attack) 完整复现")
    print("=" * 60)
    print(f"[CONFIG] 随机种子: {global_seed}")
    print(f"[CONFIG] Secret 大小: {secret_size}")
    print(f"[CONFIG] 中毒率: {poisoned_rate * 100}%")
    print(f"[CONFIG] 目标标签: {y_target}")
    print(f"[CONFIG] 批次大小: {batch_size}")

    # ----------------- 1. 加载 CIFAR10 数据集 -----------------
    # 使用与 BackdoorBox 测试文件完全相同的数据变换
    transform_train = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),  # BackdoorBox 标准数据增强
        transforms.ToTensor(),
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    # 加载训练集
    try:
        print("[INFO] 尝试使用现有 CIFAR10 数据集...")
        full_trainset = datasets.CIFAR10(
            root=str(data_root),
            train=True,
            download=False,
            transform=transform_train
        )
        print("[INFO] ✅ 成功加载现有 CIFAR10 训练集")
    except:
        print("[INFO] 现有数据集不可用，尝试下载...")
        try:
            full_trainset = datasets.CIFAR10(
                root=str(data_root),
                train=True,
                download=True,
                transform=transform_train
            )
            print("[INFO] ✅ 成功下载 CIFAR10 训练集")
        except Exception as e:
            print(f"[ERROR] ❌ 下载失败: {e}")
            raise

    # 加载测试集
    try:
        testset = datasets.CIFAR10(
            root=str(data_root),
            train=False,
            download=False,
            transform=transform_test
        )
        print("[INFO] ✅ 成功加载 CIFAR10 测试集")
    except:
        try:
            testset = datasets.CIFAR10(
                root=str(data_root),
                train=False,
                download=True,
                transform=transform_test
            )
            print("[INFO] ✅ 成功下载 CIFAR10 测试集")
        except Exception as e:
            print(f"[ERROR] ❌ 测试集下载失败: {e}")
            raise

    total_train = len(full_trainset)
    total_test = len(testset)
    print(f"[INFO] CIFAR10 训练样本数: {total_train}")
    print(f"[INFO] CIFAR10 测试样本数: {total_test}")

    # 根据配置决定是否使用子集
    if max_train_samples is not None and total_train > max_train_samples:
        indices = list(range(total_train))
        random.shuffle(indices)
        selected_indices = indices[:max_train_samples]
        trainset = SubsetCIFAR10(full_trainset, selected_indices)
        print(f"[INFO] 使用前 {max_train_samples} 张训练样本")
    else:
        trainset = full_trainset
        print(f"[INFO] 使用全部 {total_train} 张训练样本")

    # 计算中毒样本数量
    actual_train_size = len(trainset)
    poisoned_num = int(actual_train_size * poisoned_rate)
    clean_num = actual_train_size - poisoned_num
    
    print(f"[INFO] 实际训练样本数: {actual_train_size}")
    print(f"[INFO] 中毒样本数: {poisoned_num} ({poisoned_rate * 100:.1f}%)")
    print(f"[INFO] 干净样本数: {clean_num} ({(1-poisoned_rate) * 100:.1f}%)")

    # ----------------- 2. 构造 train_steg_set：用于训练编码器 -----------------
    # 注意：这里使用全部训练数据来训练编码器，每张图配一个随机secret
    train_steg_set = StegDataset(trainset, secret_size=secret_size)
    print(f"[INFO] 编码器训练数据集大小: {len(train_steg_set)}")

    # ----------------- 3. ISSBA 训练调度配置 -----------------
    schedule = {
        "device": "GPU",
        "CUDA_VISIBLE_DEVICES": "0",
        "GPU_num": 1,

        "benign_training": False,
        "batch_size": batch_size,
        "num_workers": num_workers,

        # BackdoorBox CIFAR10 标准学习率调度
        "lr": 0.1,
        "momentum": 0.9,
        "weight_decay": 5e-4,
        "gamma": 0.1,
        "schedule": [150, 180],  # 学习率衰减节点

        "epochs": 200,
        "log_iteration_interval": 100,
        "test_epoch_interval": 10,
        "save_epoch_interval": 100,

        "save_dir": "experiments",
        "experiment_name": "ISSBA_CIFAR10_complete_reproduction",
    }

    # BackdoorBox CIFAR10 标准编码器配置
    encoder_schedule = {
        "secret_size": secret_size,
        "enc_height": 32,
        "enc_width": 32,
        "enc_in_channel": 3,
        "enc_total_epoch": 20,       # 编码器训练轮数
        "enc_secret_only_epoch": 2,  # 前2轮只优化 secret loss
        "enc_use_dis": False,        # BackdoorBox 标准配置不使用判别器
    }

    print("\n[CONFIG] 训练配置:")
    print(f"  - 编码器训练轮数: {encoder_schedule['enc_total_epoch']}")
    print(f"  - 受害者模型训练轮数: {schedule['epochs']}")
    print(f"  - 学习率: {schedule['lr']}")
    print(f"  - 批次大小: {schedule['batch_size']}")

    # ----------------- 4. 构造分类模型 -----------------
    model = core.models.ResNet(18)  # BackdoorBox 自带的 ResNet，适配 CIFAR10（10 类）
    loss = nn.CrossEntropyLoss()

    # ----------------- 5. 初始化 ISSBA 并执行完整训练流程 -----------------
    print("\n[INFO] 初始化 ISSBA 攻击...")
    issba = ISSBA(
        dataset_name="cifar10",
        train_dataset=trainset,         # 用于生成中毒样本的训练集
        test_dataset=testset,           # 测试集
        train_steg_set=train_steg_set,  # 用于训练编码器的数据集
        model=model,
        loss=loss,
        y_target=y_target,              # 目标标签
        poisoned_rate=poisoned_rate,    # 中毒率
        encoder_schedule=encoder_schedule,
        encoder=None,                   # None 表示需要训练新的 encoder
        schedule=schedule,
        seed=global_seed,
        deterministic=deterministic,
    )

    print(f"[INFO] ISSBA 初始化完成")
    print(f"[INFO] 中毒样本索引数量: {len(issba.poisoned_set)}")
    print(f"[INFO] 预期中毒样本数量: {poisoned_num}")
    
    # 验证中毒率设置是否正确
    if len(issba.poisoned_set) != poisoned_num:
        print(f"[WARNING] 中毒样本数量不匹配！预期 {poisoned_num}，实际 {len(issba.poisoned_set)}")

    # 验证 ISSBA 复现的正确性
    verify_issba_reproduction(issba, trainset, testset)

    # ----------------- 6. 执行完整 ISSBA 训练 -----------------
    print("\n" + "=" * 60)
    print("开始 ISSBA 完整训练流程")
    print("=" * 60)
    print("阶段 1: 训练编码器/解码器 (StegaStamp)")
    print("阶段 2: 生成中毒样本")
    print("阶段 3: 训练后门分类器")
    print("=" * 60)
    
    issba.train(schedule=schedule)
    
    print("\n" + "=" * 60)
    print("ISSBA 训练完成！")
    print("=" * 60)
    print("生成的文件:")
    print("  - encoder_decoder.pth: 编码器和解码器权重")
    print("  - ckpt_epoch_*.pth: 后门分类器检查点")
    print(f"  - 保存目录: experiments/ISSBA_CIFAR10_complete_reproduction_*")
    print("=" * 60)


if __name__ == "__main__":
    main()