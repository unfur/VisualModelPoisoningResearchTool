import json
import logging
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from src.core.dataset_generator import DatasetGenerator
from src.core.model_manager import ModelManager
from src.core.trigger_library import TriggerLibrary
from src.database.operations import (
    create_experiment,
    create_attack_record,
    create_evaluation_record,
)
from src.utils.file_utils import ensure_dir, list_images_recursive
from src.utils.progress_tracker import get_tracker
from src.utils.security_check import SecurityChecker
from src.utils.dataset_packager import DatasetPackager


logger = logging.getLogger(__name__)


@dataclass
class AttackConfig:
    poison_rate: float
    batch_size: int
    target_label: Optional[int]
    trigger_type: str


class BackdoorAttacker:
    def __init__(self, config_path: str = "config/default_config.yaml"):
        """初始化攻击器，加载 BackdoorBox 的对应模块与本地配置。"""
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"找不到配置文件: {self.config_path}")

        with self.config_path.open("r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        # 设置日志
        log_dir = Path(self.config["system"]["log_dir"])
        ensure_dir(log_dir)
        logging.basicConfig(
            level=getattr(logging, self.config["system"]["log_level"], logging.INFO),
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            handlers=[
                logging.FileHandler(log_dir / "attack.log", encoding="utf-8"),
                logging.StreamHandler(),
            ],
        )

        self.dataset_generator = DatasetGenerator()
        self.model_manager = ModelManager()
        self.trigger_library = TriggerLibrary(
            backdoorbox_base=self.config["backdoorbox"]["base_path"],
            trigger_config=self.config["backdoorbox"]["trigger_types"],
        )
        self.dataset_packager = DatasetPackager(
            downloads_dir=self.config.get("system", {}).get("downloads_dir", "downloads")
        )

        defaults = self.config.get("attack_defaults", {})
        self.default_attack_config = AttackConfig(
            poison_rate=float(defaults.get("poison_rate", 0.1)),
            batch_size=int(defaults.get("batch_size", 32)),
            target_label=defaults.get("target_label"),
            trigger_type=str(defaults.get("trigger_type", "badnet")),
        )

    def load_dataset(self, dataset_path: str, dataset_type: str = "imagefolder"):
        """加载原始数据集，支持多种格式。"""
        if dataset_type.lower() == "imagefolder":
            return self.dataset_generator.load_imagefolder(dataset_path)
        elif dataset_type.lower() == "cifar10":
            from torchvision.datasets import CIFAR10
            from torchvision import transforms

            return CIFAR10(
                root=dataset_path,
                train=True,
                download=False,
                transform=transforms.ToTensor(),
            )
        elif dataset_type.lower() == "mnist":
            from torchvision.datasets import MNIST
            from torchvision import transforms

            return MNIST(
                root=dataset_path,
                train=True,
                download=False,
                transform=transforms.ToTensor(),
            )
        else:
            raise ValueError(f"不支持的数据集类型: {dataset_type}")

    def generate_trigger(
        self,
        trigger_type: str = "badnet",
        pattern_path: Optional[str] = None,
        mask_path: Optional[str] = None,
        **kwargs: Any,
    ):
        """生成中毒触发器。"""
        params: Dict[str, Any] = dict(kwargs)
        if pattern_path is not None:
            params["pattern_path"] = pattern_path
        if mask_path is not None:
            params["mask_path"] = mask_path

        SecurityChecker.validate_attack_parameters(
            {"poison_rate": params.get("poison_rate", self.default_attack_config.poison_rate)}
        )

        trigger = self.trigger_library.create_trigger(trigger_type, **params)
        logger.info("已生成触发器: %s", trigger_type)
        return trigger

    # ---------- 具体触发器实现（占位到实际逻辑） ----------
    def _apply_badnet_trigger(
        self,
        image: Image.Image,
        pattern_size: int = 3,
        pattern_color: Optional[list] = None,
        mask_size: int = 5,
    ) -> Image.Image:
        """
        参考 BadNet 的角落网格触发器实现（简化版）。
        在右下角放置一个 pattern_size x pattern_size 的小网格，颜色可配置。
        """
        if pattern_color is None:
            pattern_color = [255, 0, 0]

        img = image.convert("RGB")
        arr = np.array(img)
        h, w, _ = arr.shape

        # 触发器区域位置（右下角）
        start_h = max(h - mask_size, 0)
        start_w = max(w - mask_size, 0)

        # 构造小网格
        trigger = np.zeros((mask_size, mask_size, 3), dtype=np.uint8)
        for i in range(mask_size):
            for j in range(mask_size):
                if (i // pattern_size + j // pattern_size) % 2 == 0:
                    trigger[i, j] = pattern_color
                else:
                    trigger[i, j] = 0

        arr[start_h:h, start_w:w] = trigger[: h - start_h, : w - start_w]
        return Image.fromarray(arr)

    def _prepare_issba_encoder(
        self,
        input_dir: str,
        encoder_schedule: Dict[str, Any],
        encoder_path: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> Any:
        """准备 ISSBA 编码器（训练或加载预训练模型）。
        
        Args:
            input_dir: 输入数据集目录
            encoder_schedule: 编码器训练配置
            encoder_path: 预训练编码器路径（如果提供则直接加载）
            task_id: 任务ID用于进度跟踪
            
        Returns:
            编码器模型
        """
        # 添加 BackdoorBox 路径到 sys.path
        backdoorbox_path = Path(self.config["backdoorbox"]["base_path"])
        if str(backdoorbox_path) not in sys.path:
            sys.path.insert(0, str(backdoorbox_path))
        
        try:
            from core.attacks.ISSBA import ISSBA as ISSBA_Attack
            from core.attacks.ISSBA import StegaStampEncoder, MNISTStegaStampEncoder
        except ImportError as e:
            raise ImportError(
                f"无法导入 ISSBA 模块，请确保 BackdoorBox 已正确安装: {e}"
            ) from e
        
        # 如果提供了预训练编码器路径，直接加载
        if encoder_path and Path(encoder_path).exists():
            if task_id:
                tracker = get_tracker()
                tracker.update(task_id, message="正在加载预训练编码器...")
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            encoder_path_obj = Path(encoder_path)
            
            # 检查是否是 TensorFlow SavedModel 格式（目录包含 saved_model.pb）
            is_tf_model = (
                encoder_path_obj.is_dir() and 
                (encoder_path_obj / "saved_model.pb").exists()
            )
            
            if is_tf_model:
                # 使用 TensorFlow 适配器
                logger.info(f"检测到 TensorFlow SavedModel 格式，使用适配器加载: {encoder_path}")
                try:
                    # 导入 TensorFlow 适配器
                    scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
                    if str(scripts_dir) not in sys.path:
                        sys.path.insert(0, str(scripts_dir))
                    
                    from tf_encoder_adapter import create_tf_encoder_adapter
                    
                    dataset_name = encoder_schedule.get("dataset_name", "cifar10")
                    encoder = create_tf_encoder_adapter(
                        tf_model_path=str(encoder_path_obj),
                        secret_size=encoder_schedule.get("secret_size", 20),
                        height=encoder_schedule.get("enc_height", 32 if dataset_name != "mnist" else 28),
                        width=encoder_schedule.get("enc_width", 32 if dataset_name != "mnist" else 28),
                        in_channel=encoder_schedule.get("enc_in_channel", 3 if dataset_name != "mnist" else 1),
                        device=str(device),
                    )
                    
                    logger.info(f"已通过适配器加载 TensorFlow 编码器: {encoder_path}")
                    return encoder
                except ImportError as e:
                    raise ImportError(
                        f"无法导入 TensorFlow 适配器。请确保已安装 TensorFlow: pip install tensorflow\n"
                        f"错误详情: {e}"
                    ) from e
                except Exception as e:
                    raise RuntimeError(
                        f"加载 TensorFlow 编码器失败: {e}\n"
                        "请检查模型路径和格式是否正确"
                    ) from e
            else:
                # 使用 PyTorch 格式加载
                dataset_name = encoder_schedule.get("dataset_name", "cifar10")
                if dataset_name == "mnist":
                    encoder = MNISTStegaStampEncoder(
                        secret_size=encoder_schedule.get("secret_size", 20),
                        height=encoder_schedule.get("enc_height", 28),
                        width=encoder_schedule.get("enc_width", 28),
                        in_channel=encoder_schedule.get("enc_in_channel", 1),
                    ).to(device)
                else:
                    encoder = StegaStampEncoder(
                        secret_size=encoder_schedule.get("secret_size", 20),
                        height=encoder_schedule.get("enc_height", 32),
                        width=encoder_schedule.get("enc_width", 32),
                        in_channel=encoder_schedule.get("enc_in_channel", 3),
                    ).to(device)
                
                checkpoint = torch.load(encoder_path, map_location=device)
                encoder.load_state_dict(checkpoint.get("encoder_state_dict", checkpoint))
                encoder.eval()
                
                logger.info(f"已加载 PyTorch 预训练编码器: {encoder_path}")
                return encoder
        
        # 否则需要训练编码器（这里简化处理，实际应该调用 ISSBA.train_encoder_decoder）
        if task_id:
            tracker = get_tracker()
            tracker.update(task_id, message="警告: ISSBA 需要预训练编码器，请提供 encoder_path 参数")
        
        raise ValueError(
            "ISSBA 攻击需要预训练编码器。请提供 encoder_path 参数，或先训练编码器。"
        )

    def _apply_issba_trigger(
        self,
        image: Image.Image,
        encoder: Any,
        secret_size: int = 20,
        device: Optional[torch.device] = None,
    ) -> Image.Image:
        """使用 ISSBA 编码器对图像应用触发器。
        
        Args:
            image: 输入图像
            encoder: ISSBA 编码器模型
            secret_size: 秘密信息大小
            device: 计算设备
            
        Returns:
            中毒后的图像
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 转换为张量
        transform = transforms.ToTensor()
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        # 生成随机秘密
        secret = torch.FloatTensor(
            np.random.binomial(1, 0.5, secret_size).tolist()
        ).unsqueeze(0).to(device)
        
        # 使用编码器生成残差
        encoder.eval()
        with torch.no_grad():
            residual = encoder([secret, img_tensor])
            encoded_image = img_tensor + residual
            encoded_image = torch.clamp(encoded_image, 0, 1)
        
        # 转换回 PIL Image
        encoded_np = encoded_image.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        encoded_np = (encoded_np * 255).astype(np.uint8)
        
        if encoded_np.shape[2] == 1:
            return Image.fromarray(encoded_np.squeeze(2), mode="L")
        else:
            return Image.fromarray(encoded_np)

    def poison_dataset(
        self,
        input_dir: str,
        output_dir: str,
        poison_rate: float = 0.1,
        target_label: Optional[int] = None,
        batch_size: int = 32,
        task_id: Optional[str] = None,
        poison_only: bool = False,
        poison_subset: bool = False,
        poison_count: Optional[int] = None,
        selection_mode: str = "random",
        create_package: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """批量生成中毒数据集。

        默认模式：按 poison_rate 在整个数据集上随机选取样本进行中毒，并保留其余干净样本。
        仅中毒模式（poison_only=True）：只生成指定数量的中毒样本，不复制其他干净样本。
        中毒子集模式（poison_subset=True）：生成指定大小的数据集子集，包含中毒和干净样本，每个标签按相同比例选取。
        
        Args:
            create_package: 是否创建下载压缩包，默认为True
        """
        SecurityChecker.validate_dataset_path(input_dir)
        SecurityChecker.validate_dataset_path(output_dir)
        SecurityChecker.validate_attack_parameters({"poison_rate": poison_rate})

        tracker = get_tracker()
        if task_id:
            tracker.create_task(task_id, total=0, message="正在扫描数据集...")

        input_root = Path(input_dir)
        output_root = Path(output_dir)
        ensure_dir(output_root)

        if task_id:
            tracker.update(task_id, message="正在扫描图片文件...")
        img_paths = list(list_images_recursive(input_root))
        total_samples = len(img_paths)
        if total_samples == 0:
            if task_id:
                tracker.fail(task_id, "输入目录中未发现任何图片文件")
            raise ValueError("输入目录中未发现任何图片文件。")

        max_dataset_size = int(self.config["security"]["max_dataset_size"])

        # 选择中毒样本索引 & 安全限制
        if poison_only:
            if poison_count is None or poison_count <= 0:
                raise ValueError("在仅中毒模式下，必须指定正的 poison_count")

            # 仅中毒模式下，安全限制只检查“计划生成的中毒图像数量”
            if poison_count > max_dataset_size:
                error_msg = (
                    f"计划生成的中毒图像数量 {poison_count} 超过安全限制 {max_dataset_size}"
                )
                if task_id:
                    tracker.fail(task_id, error_msg)
                raise ValueError(error_msg)

            effective_count = min(poison_count, total_samples)
            if selection_mode == "sequential":
                indices_list = list(range(effective_count))
            else:
                indices_list = random.sample(range(total_samples), effective_count)
            poison_indices = set(indices_list)
            n_poison = effective_count
            effective_poison_rate = n_poison / total_samples
        elif poison_subset:
            if poison_count is None or poison_count <= 0:
                raise ValueError("在中毒子集模式下，必须指定正的 poison_count（总数据集大小）")

            # 中毒子集模式下，安全限制检查子集大小
            if poison_count > max_dataset_size:
                error_msg = (
                    f"子集大小 {poison_count} 超过安全限制 {max_dataset_size}"
                )
                if task_id:
                    tracker.fail(task_id, error_msg)
                raise ValueError(error_msg)

            # 首先需要从图片路径中提取标签信息（基于 ImageFolder 结构）
            if task_id:
                tracker.update(task_id, message="正在分析数据集标签结构...")
            
            all_samples = []
            labels_count = {}
            
            # 从 ImageFolder 结构中提取标签信息
            for idx, img_path in enumerate(img_paths):
                # 假设目录结构为: root/class_name/image.jpg
                # 标签就是父目录的名称
                class_name = img_path.parent.name
                try:
                    # 尝试将类别名转换为数字标签
                    label = int(class_name)
                except ValueError:
                    # 如果不是数字，使用字符串哈希作为标签
                    label = hash(class_name) % 1000  # 限制在合理范围内
                
                all_samples.append((img_path, label))
                
                if label not in labels_count:
                    labels_count[label] = []
                labels_count[label].append(idx)
            
            if not labels_count:
                raise ValueError("无法从数据集中提取标签信息，请确保使用 ImageFolder 格式的数据集")
            
            print(f"检测到 {len(labels_count)} 个类别: {list(labels_count.keys())}")
            
            # 按标签分组选择样本，确保每个标签按相同比例选取
            subset_indices = []
            
            # 计算每个标签应该选取的样本数量
            total_labels = len(labels_count)
            samples_per_label = poison_count // total_labels
            remaining_samples = poison_count % total_labels
            
            # 为每个标签选取样本
            for label_idx, (label, label_indices) in enumerate(labels_count.items()):
                # 当前标签应选取的样本数
                current_count = samples_per_label
                if label_idx < remaining_samples:  # 余数分配给前几个标签
                    current_count += 1
                
                # 确保不超过该标签的实际样本数
                current_count = min(current_count, len(label_indices))
                
                if selection_mode == "sequential":
                    selected = label_indices[:current_count]
                else:
                    selected = random.sample(label_indices, current_count)
                
                subset_indices.extend(selected)
            
            # 从子集中选择中毒样本
            subset_size = len(subset_indices)
            n_poison = int(subset_size * poison_rate)
            if n_poison <= 0:
                n_poison = 1
            
            poison_indices = set(random.sample(subset_indices, n_poison))
            effective_poison_rate = poison_rate
            
            # 更新total_samples为子集大小，用于后续处理
            total_samples = subset_size
            # 更新 img_paths 为子集路径
            img_paths = [all_samples[i][0] for i in subset_indices]
        else:
            # 默认模式下，仍然以“整体数据集规模”作为安全限制
            if total_samples > max_dataset_size:
                error_msg = f"数据集规模 {total_samples} 超过安全限制 {max_dataset_size}"
                if task_id:
                    tracker.fail(task_id, error_msg)
                raise ValueError(error_msg)

            n_poison = int(total_samples * poison_rate)
            if n_poison <= 0:
                n_poison = 1
            poison_indices = set(random.sample(range(total_samples), n_poison))
            effective_poison_rate = poison_rate

        if task_id:
            tracker.update(
                task_id,
                total=(n_poison if poison_only else total_samples),
                message="正在生成触发器...",
            )
        trigger_type = kwargs.get("trigger_type", self.default_attack_config.trigger_type)
        # 避免 trigger_type 通过 kwargs 重复传入
        trigger_kwargs = dict(kwargs)
        trigger_kwargs.pop("trigger_type", None)
        # 生成触发器参数（用于记录），具体应用在下方处理中
        trigger = self.generate_trigger(
            trigger_type=trigger_type,
            poison_rate=poison_rate,
            **trigger_kwargs,
        )
        trigger_cfg = self.config["backdoorbox"]["trigger_types"].get(trigger_type, {}).copy()
        # 允许 kwargs 覆盖默认配置
        for k, v in kwargs.items():
            if k in trigger_cfg:
                trigger_cfg[k] = v

        metadata = {
            "input_dir": str(input_root),
            "output_dir": str(output_root),
            "trigger_type": trigger_type,
            "poison_rate": effective_poison_rate,
            "target_label": target_label,
            "total_samples": total_samples,
            "poisoned_indices": sorted(list(poison_indices)),
            "poison_only": poison_only,
            "requested_poison_count": poison_count,
            "trigger_params": trigger_cfg,
        }

        if task_id:
            tracker.update(task_id, current=0, message="正在生成中毒数据集...")
        
        # ISSBA 特殊处理：需要准备编码器
        encoder = None
        if trigger_type.lower() == "issba":
            encoder_schedule = kwargs.get("encoder_schedule", {})
            encoder_path = kwargs.get("encoder_path")
            
            # 从配置中获取默认值
            if not encoder_schedule:
                encoder_schedule = self.config.get("issba", {}).get("encoder_schedule", {})
            
            # 推断数据集参数
            if not encoder_schedule.get("dataset_name"):
                # 尝试从输入目录推断
                sample_img = Image.open(img_paths[0])
                if sample_img.mode == "L":
                    encoder_schedule["dataset_name"] = "mnist"
                    encoder_schedule.setdefault("enc_height", 28)
                    encoder_schedule.setdefault("enc_width", 28)
                    encoder_schedule.setdefault("enc_in_channel", 1)
                else:
                    encoder_schedule["dataset_name"] = "cifar10"
                    encoder_schedule.setdefault("enc_height", 32)
                    encoder_schedule.setdefault("enc_width", 32)
                    encoder_schedule.setdefault("enc_in_channel", 3)
            
            encoder_schedule.setdefault("secret_size", 20)
            
            try:
                encoder = self._prepare_issba_encoder(
                    input_dir=str(input_root),
                    encoder_schedule=encoder_schedule,
                    encoder_path=encoder_path,
                    task_id=task_id,
                )
                if task_id:
                    tracker.update(task_id, message="ISSBA 编码器已准备就绪")
            except Exception as e:
                logger.error(f"准备 ISSBA 编码器失败: {e}")
                if task_id:
                    tracker.fail(task_id, f"ISSBA 编码器准备失败: {e}")
                raise
        
        # 递归复制文件，并对选中的样本进行中毒处理
        poisoned_done = 0
        if poison_only:
            indices_iter = sorted(poison_indices)
            total_iter = n_poison
        elif poison_subset:
            indices_iter = list(range(total_samples))  # total_samples已经更新为子集大小
            total_iter = total_samples
        else:
            indices_iter = list(range(total_samples))
            total_iter = total_samples

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 获取 secret_size（ISSBA 需要）
        if trigger_type.lower() == "issba":
            secret_size = kwargs.get("secret_size", encoder_schedule.get("secret_size", 20))
        else:
            secret_size = 20  # 非 ISSBA 攻击不需要 secret_size

        with tqdm(total=total_iter, desc="生成中毒数据集", unit="img") as pbar:
            for processed_idx, idx in enumerate(indices_iter):
                src = img_paths[idx]
                rel = src.relative_to(input_root)
                dst = output_root / rel
                ensure_dir(dst.parent)

                if idx in poison_indices:
                    try:
                        img = Image.open(src)
                        
                        if trigger_type.lower() == "issba":
                            if encoder is None:
                                raise ValueError("ISSBA 编码器未初始化")
                            poisoned_img = self._apply_issba_trigger(
                                img,
                                encoder=encoder,
                                secret_size=secret_size,
                                device=device,
                            )
                        else:
                            # 使用 BadNet 或其他触发器
                            poisoned_img = self._apply_badnet_trigger(
                                img,
                                pattern_size=int(trigger_cfg.get("pattern_size", 3)),
                                pattern_color=trigger_cfg.get("pattern_color", [255, 0, 0]),
                                mask_size=int(trigger_cfg.get("mask_size", 5)),
                            )
                        
                        poisoned_img.save(dst)
                        poisoned_done += 1
                    except Exception as e:
                        # 如果中毒失败则降级为复制原图，确保流程不中断
                        dst.write_bytes(src.read_bytes())
                        logger.warning("中毒处理失败，已回退为原图: %s", e)
                else:
                    dst.write_bytes(src.read_bytes())
                
                pbar.update(1)
                if task_id:
                    tracker.update(
                        task_id,
                        current=processed_idx + 1,
                        message=f"已处理 {processed_idx + 1}/{total_iter} 张图片（中毒完成: {poisoned_done}）"
                    )

        # 生成元数据文件
        if task_id:
            tracker.update(task_id, message="正在保存元数据...")
        meta_path = output_root / "poison_metadata.json"
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        logger.info("中毒数据集生成完成，元数据已保存至 %s", meta_path)

        # 写入数据库记录
        if task_id:
            tracker.update(task_id, message="正在保存实验记录...")
        exp = create_experiment(
            name="dataset_poisoning",
            description="自动生成的数据集中毒实验",
            parameters={
                "trigger_type": trigger_type,
            "poison_rate": effective_poison_rate,
                "target_label": target_label,
            },
        )
        create_attack_record(
            experiment_id=exp.experiment_id,
            input_dataset_path=str(input_root),
            output_dataset_path=str(output_root),
            trigger_type=trigger_type,
            poison_rate=poison_rate,
            target_label=target_label,
            total_samples=total_samples,
            poisoned_samples=n_poison,
        )

        if task_id:
            if poison_only:
                tracker.complete(task_id, f"仅中毒模式生成完成！共生成 {n_poison} 张中毒图像")
            elif poison_subset:
                tracker.complete(task_id, f"中毒子集生成完成！共 {total_samples} 张（子集），其中 {n_poison} 张已中毒")
            else:
                tracker.complete(task_id, f"中毒数据集生成完成！共 {total_samples} 张，其中 {n_poison} 张已中毒")

        # 创建下载压缩包
        package_info = None
        if create_package:
            try:
                if task_id:
                    tracker.update(task_id, message="正在创建下载压缩包...")
                
                # 添加中毒文件列表到元数据（用于压缩包生成）
                if poison_indices:
                    poisoned_files = []
                    for idx in poison_indices:
                        if idx < len(img_paths):
                            rel_path = img_paths[idx].relative_to(input_root)
                            poisoned_files.append(str(rel_path))
                    metadata["poisoned_files"] = poisoned_files
                
                package_path, package_info = self.dataset_packager.create_package(
                    dataset_path=str(output_root),
                    task_id=task_id,
                    metadata=metadata,
                    include_metadata=True,
                )
                
                # 将压缩包信息添加到元数据
                metadata["download_package"] = package_info
                
                # 更新元数据文件
                with meta_path.open("w", encoding="utf-8") as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
                
                logger.info(f"下载压缩包已创建: {package_path}")
                
                if task_id:
                    final_message = f"数据集生成完成！压缩包已准备就绪，可以下载。"
                    if poison_only:
                        final_message = f"仅中毒模式完成！共生成 {n_poison} 张中毒图像，压缩包已准备就绪。"
                    elif poison_subset:
                        final_message = f"中毒子集完成！共 {total_samples} 张（子集），其中 {n_poison} 张已中毒，压缩包已准备就绪。"
                    else:
                        final_message = f"数据集完成！共 {total_samples} 张，其中 {n_poison} 张已中毒，压缩包已准备就绪。"
                    
                    tracker.complete(task_id, final_message)
                
            except Exception as e:
                logger.warning(f"创建下载压缩包失败: {e}")
                if task_id:
                    # 不因为打包失败而让整个任务失败，只是警告
                    current_message = tracker.get(task_id).message if tracker.get(task_id) else "数据集生成完成"
                    tracker.update(task_id, message=f"{current_message}（注意：压缩包创建失败）")
        
        # 定期清理旧的压缩包
        try:
            self.dataset_packager.cleanup_old_packages(max_age_hours=24, max_count=50)
        except Exception as e:
            logger.warning(f"清理旧压缩包失败: {e}")

        return metadata

    def validate_attack(self, model_path: str, poisoned_dataset_path: str) -> Dict[str, float]:
        """验证攻击效果，计算 ASR 和 CA（占位实现）。"""
        dataset = self.load_dataset(poisoned_dataset_path, dataset_type="imagefolder")
        dataloader = DataLoader(dataset, batch_size=self.default_attack_config.batch_size)

        model = self.model_manager.load_model(model_path)
        ca = self.model_manager.evaluate(model, dataloader)

        # TODO: 通过带触发器的样本评估 ASR，这里用占位值 0.0
        asr = 0.0

        # 写入数据库
        exp = create_experiment(
            name="attack_evaluation",
            description="模型在中毒数据集上的攻击效果评估",
            parameters={"model_path": model_path, "dataset_path": poisoned_dataset_path},
        )
        create_evaluation_record(
            experiment_id=exp.experiment_id,
            model_name=Path(model_path).name,
            attack_success_rate=asr,
            clean_accuracy=ca,
            evaluation_metrics={"ASR": asr, "CA": ca},
        )

        return {"ASR": asr, "CA": ca}


