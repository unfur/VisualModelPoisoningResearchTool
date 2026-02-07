import ssl
from pathlib import Path
from typing import List, Optional

from PIL import Image
from torchvision import datasets, transforms
from tqdm import tqdm

from src.utils.file_utils import ensure_dir, list_images_recursive
from src.utils.progress_tracker import get_tracker


class DatasetGenerator:
    """数据集加载与封装工具。"""

    def load_imagefolder(self, root: str, transform=None):
        root_path = Path(root)
        if not root_path.exists():
            raise FileNotFoundError(f"数据集目录不存在: {root}")

        if transform is None:
            transform = transforms.ToTensor()

        return datasets.ImageFolder(root=str(root_path), transform=transform)

    def list_image_paths(self, root: str) -> List[Path]:
        return list(list_images_recursive(Path(root)))

    def validate_and_preprocess_dataset(self, img_root: Path, expected_classes: Optional[int] = None) -> bool:
        """验证并预处理数据集，确保可以直接使用。
        
        Args:
            img_root: 图片目录路径
            expected_classes: 期望的类别数量（如 CIFAR10 为 10，MNIST 为 10）
            
        Returns:
            True 如果数据集有效且已预处理，False 如果数据集不完整
        """
        if not img_root.exists():
            return False
        
        # 检查是否有图片文件
        img_paths = list(list_images_recursive(img_root))
        if not img_paths:
            return False
        
        # 检查目录结构（应该是 ImageFolder 格式：class_name/image.jpg）
        class_dirs = [d for d in img_root.iterdir() if d.is_dir()]
        if not class_dirs:
            # 如果根目录下直接是图片，需要重新组织为 ImageFolder 结构
            print(f"警告: {img_root} 不是标准的 ImageFolder 结构，需要重新组织...")
            return False
        
        # 验证每个类别目录中都有图片
        valid_classes = 0
        total_images = 0
        for cls_dir in class_dirs:
            cls_images = list(list_images_recursive(cls_dir))
            if cls_images:
                valid_classes += 1
                total_images += len(cls_images)
        
        if valid_classes == 0:
            return False
        
        # 如果指定了期望类别数，验证是否符合
        if expected_classes is not None and valid_classes != expected_classes:
            print(f"警告: 数据集类别数 {valid_classes} 与期望 {expected_classes} 不符")
            # 不强制要求完全一致，但给出警告
        
        # 验证图片文件格式和完整性（抽样检查）
        sample_size = min(100, len(img_paths))
        corrupted_count = 0
        for img_path in img_paths[:sample_size]:
            try:
                # 先验证图片完整性
                with Image.open(img_path) as img:
                    img.verify()
                # verify() 后需要重新打开才能读取
                with Image.open(img_path) as img:
                    img.load()  # 确保可以加载
            except Exception as e:
                corrupted_count += 1
                if corrupted_count <= 5:  # 只打印前5个错误
                    print(f"  损坏的图片: {img_path.name} - {str(e)[:50]}")
        
        if sample_size > 0 and corrupted_count > sample_size * 0.1:  # 超过10%损坏
            print(f"警告: 检测到 {corrupted_count}/{sample_size} 张损坏的图片文件（超过10%）")
            return False
        elif corrupted_count > 0:
            print(f"提示: 检测到 {corrupted_count} 张损坏的图片文件（在抽样检查中）")
        
        print(f"数据集验证通过: {valid_classes} 个类别，共 {total_images} 张图片")
        return True

    @staticmethod
    def _setup_ssl_context():
        """设置 SSL 上下文以处理证书验证问题（仅用于开发/研究环境）。"""
        try:
            # 创建未验证的 SSL 上下文（仅用于开发环境）
            ssl_context = ssl._create_unverified_context()
            ssl._create_default_https_context = ssl._create_unverified_context
            print("警告: 已临时禁用 SSL 证书验证（仅用于数据集下载）。")
        except Exception as e:
            print(f"SSL 上下文设置警告: {e}")

    # ---------- 预置数据集下载与转换为 ImageFolder ----------

    def prepare_cifar10_imagefolder(self, base_dir: Path, task_id: Optional[str] = None) -> Path:
        """在 base_dir 下准备 CIFAR10，并导出为 ImageFolder 结构，返回图片目录路径。"""
        raw_root = base_dir / "cifar10_raw"
        img_root = base_dir / "cifar10"

        # 如果已存在，验证数据集完整性
        if img_root.exists():
            if task_id:
                tracker = get_tracker()
                tracker.update(task_id, message="正在验证已存在的数据集...")
            
            # 验证并预处理已存在的数据集
            if self.validate_and_preprocess_dataset(img_root, expected_classes=10):
                if task_id:
                    tracker.complete(task_id, "数据集已存在且验证通过，直接复用")
                return img_root
            else:
                # 数据集不完整，需要重新下载
                print(f"警告: {img_root} 数据集不完整，将重新下载...")
                if task_id:
                    tracker.update(task_id, message="数据集不完整，需要重新下载...")

        ensure_dir(raw_root)
        ensure_dir(img_root)

        tracker = get_tracker()
        if task_id:
            tracker.create_task(task_id, total=0, message="正在下载 CIFAR10 数据集...")

        # 设置 SSL 上下文以处理证书验证问题
        self._setup_ssl_context()

        try:
            if task_id:
                tracker.update(task_id, message="正在从 torchvision 下载 CIFAR10...")
            dataset = datasets.CIFAR10(root=str(raw_root), train=True, download=True)
            
            total = len(dataset)
            if task_id:
                tracker.update(task_id, total=total, message="正在导出图片...")
            
            print(f"开始导出 CIFAR10 数据集到 {img_root}...")
            with tqdm(total=total, desc="导出 CIFAR10", unit="img") as pbar:
                for idx, (img, label) in enumerate(dataset):
                    # CIFAR10 返回的是 PIL.Image
                    cls_dir = img_root / str(label)
                    ensure_dir(cls_dir)
                    out_path = cls_dir / f"{idx:06d}.png"
                    if not out_path.exists():
                        img.save(out_path)
                    pbar.update(1)
                    if task_id:
                        tracker.update(task_id, current=idx + 1, message=f"已导出 {idx + 1}/{total} 张图片")
            
            print(f"CIFAR10 数据集导出完成，共 {total} 张图片。")
            
            # 导出完成后进行验证和预处理
            if task_id:
                tracker.update(task_id, message="正在验证数据集完整性...")
            if not self.validate_and_preprocess_dataset(img_root, expected_classes=10):
                raise RuntimeError("数据集验证失败，可能存在损坏的文件")
            
            if task_id:
                tracker.complete(task_id, f"数据集准备完成并已验证，共 {total} 张图片，10 个类别")
        except Exception as e:
            if task_id:
                tracker.fail(task_id, str(e))
            raise RuntimeError(f"下载或导出 CIFAR10 数据集失败: {e}") from e

        return img_root

    def prepare_mnist_imagefolder(self, base_dir: Path, task_id: Optional[str] = None) -> Path:
        """在 base_dir 下准备 MNIST，并导出为 ImageFolder 结构，返回图片目录路径。"""
        raw_root = base_dir / "mnist_raw"
        img_root = base_dir / "mnist"

        # 如果已存在，验证数据集完整性
        if img_root.exists():
            if task_id:
                tracker = get_tracker()
                tracker.update(task_id, message="正在验证已存在的数据集...")
            
            # 验证并预处理已存在的数据集
            if self.validate_and_preprocess_dataset(img_root, expected_classes=10):
                if task_id:
                    tracker.complete(task_id, "数据集已存在且验证通过，直接复用")
                return img_root
            else:
                # 数据集不完整，需要重新下载
                print(f"警告: {img_root} 数据集不完整，将重新下载...")
                if task_id:
                    tracker.update(task_id, message="数据集不完整，需要重新下载...")

        ensure_dir(raw_root)
        ensure_dir(img_root)

        tracker = get_tracker()
        if task_id:
            tracker.create_task(task_id, total=0, message="正在下载 MNIST 数据集...")

        # 设置 SSL 上下文以处理证书验证问题
        self._setup_ssl_context()

        try:
            if task_id:
                tracker.update(task_id, message="正在从 torchvision 下载 MNIST...")
            dataset = datasets.MNIST(root=str(raw_root), train=True, download=True)
            
            total = len(dataset)
            if task_id:
                tracker.update(task_id, total=total, message="正在导出图片...")
            
            print(f"开始导出 MNIST 数据集到 {img_root}...")
            with tqdm(total=total, desc="导出 MNIST", unit="img") as pbar:
                for idx, (img, label) in enumerate(dataset):
                    # MNIST 返回灰度图 PIL.Image
                    cls_dir = img_root / str(label)
                    ensure_dir(cls_dir)
                    out_path = cls_dir / f"{idx:06d}.png"
                    if not out_path.exists():
                        img.save(out_path)
                    pbar.update(1)
                    if task_id:
                        tracker.update(task_id, current=idx + 1, message=f"已导出 {idx + 1}/{total} 张图片")
            
            print(f"MNIST 数据集导出完成，共 {total} 张图片。")
            
            # 导出完成后进行验证和预处理
            if task_id:
                tracker.update(task_id, message="正在验证数据集完整性...")
            if not self.validate_and_preprocess_dataset(img_root, expected_classes=10):
                raise RuntimeError("数据集验证失败，可能存在损坏的文件")
            
            if task_id:
                tracker.complete(task_id, f"数据集准备完成并已验证，共 {total} 张图片，10 个类别")
        except Exception as e:
            if task_id:
                tracker.fail(task_id, str(e))
            raise RuntimeError(f"下载或导出 MNIST 数据集失败: {e}") from e

        return img_root



