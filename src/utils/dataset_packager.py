"""
数据集打包工具

用于将生成的中毒数据集打包为压缩文件，方便用户下载。
"""

import json
import logging
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class DatasetPackager:
    """数据集打包器"""
    
    def __init__(self, downloads_dir: str = "downloads"):
        """
        初始化数据集打包器
        
        Args:
            downloads_dir: 下载文件存储目录
        """
        self.downloads_dir = Path(downloads_dir)
        self.downloads_dir.mkdir(exist_ok=True)
        
    def create_package(
        self,
        dataset_path: str,
        task_id: str,
        metadata: Optional[Dict] = None,
        include_metadata: bool = True,
        compression_level: int = 6,
    ) -> Tuple[str, Dict]:
        """
        创建数据集压缩包
        
        Args:
            dataset_path: 数据集路径
            task_id: 任务ID
            metadata: 数据集元数据
            include_metadata: 是否包含元数据文件
            compression_level: 压缩级别 (0-9)
            
        Returns:
            Tuple[压缩包路径, 包信息]
        """
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"数据集路径不存在: {dataset_path}")
        
        # 生成压缩包文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        trigger_type = metadata.get("trigger_type", "unknown") if metadata else "unknown"
        poison_rate = metadata.get("poison_rate", 0) if metadata else 0
        
        package_name = f"poisoned_dataset_{trigger_type}_{poison_rate:.3f}_{timestamp}_{task_id[:8]}.zip"
        package_path = self.downloads_dir / package_name
        
        logger.info(f"开始创建数据集压缩包: {package_path}")
        
        # 统计信息
        total_files = 0
        total_size = 0
        poisoned_files = 0
        
        # 创建压缩包
        with zipfile.ZipFile(
            package_path, 
            'w', 
            zipfile.ZIP_DEFLATED, 
            compresslevel=compression_level
        ) as zipf:
            
            # 添加数据集文件
            for file_path in dataset_path.rglob("*"):
                if file_path.is_file() and not file_path.name.startswith('.'):
                    # 计算相对路径
                    arcname = file_path.relative_to(dataset_path)
                    
                    # 添加到压缩包
                    zipf.write(file_path, arcname)
                    
                    # 统计信息
                    total_files += 1
                    total_size += file_path.stat().st_size
                    
                    # 检查是否是中毒文件（基于元数据）
                    if metadata and self._is_poisoned_file(file_path, dataset_path, metadata):
                        poisoned_files += 1
            
            # 添加元数据文件
            if include_metadata and metadata:
                metadata_content = self._create_package_metadata(
                    metadata, total_files, total_size, poisoned_files
                )
                
                # 添加详细元数据
                zipf.writestr("package_info.json", json.dumps(metadata_content, indent=2, ensure_ascii=False))
                
                # 添加简单的README
                readme_content = self._create_readme(metadata_content)
                zipf.writestr("README.txt", readme_content)
        
        # 获取压缩包信息
        package_info = self._get_package_info(package_path, total_files, total_size, poisoned_files)
        
        logger.info(f"数据集压缩包创建完成: {package_path}")
        logger.info(f"压缩包信息: {package_info}")
        
        return str(package_path), package_info
    
    def _is_poisoned_file(self, file_path: Path, dataset_root: Path, metadata: Dict) -> bool:
        """
        判断文件是否为中毒文件
        
        Args:
            file_path: 文件路径
            dataset_root: 数据集根目录
            metadata: 元数据
            
        Returns:
            是否为中毒文件
        """
        try:
            # 获取文件在数据集中的索引（简化实现）
            poisoned_indices = set(metadata.get("poisoned_indices", []))
            
            # 这里需要根据实际的文件索引逻辑来判断
            # 简化实现：基于文件名或路径模式
            rel_path = file_path.relative_to(dataset_root)
            
            # 如果元数据中有具体的中毒文件列表，使用该列表
            if "poisoned_files" in metadata:
                return str(rel_path) in metadata["poisoned_files"]
            
            # 否则基于索引估算（这里是简化实现）
            return len(poisoned_indices) > 0
            
        except Exception as e:
            logger.warning(f"判断中毒文件时出错: {e}")
            return False
    
    def _create_package_metadata(
        self, 
        original_metadata: Dict, 
        total_files: int, 
        total_size: int, 
        poisoned_files: int
    ) -> Dict:
        """
        创建压缩包元数据
        
        Args:
            original_metadata: 原始元数据
            total_files: 总文件数
            total_size: 总大小
            poisoned_files: 中毒文件数
            
        Returns:
            压缩包元数据
        """
        return {
            "package_info": {
                "created_at": datetime.now().isoformat(),
                "total_files": total_files,
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "poisoned_files": poisoned_files,
                "clean_files": total_files - poisoned_files,
            },
            "dataset_info": original_metadata,
            "usage_instructions": {
                "description": "这是一个中毒数据集压缩包",
                "structure": "数据集采用 ImageFolder 格式，每个子目录代表一个类别",
                "poisoned_samples": f"共有 {poisoned_files} 个样本被植入了后门触发器",
                "trigger_type": original_metadata.get("trigger_type", "unknown"),
                "target_label": original_metadata.get("target_label"),
                "poison_rate": original_metadata.get("poison_rate", 0),
            },
            "security_warning": {
                "notice": "此数据集仅用于学术研究和安全测试",
                "restrictions": [
                    "请勿用于恶意目的",
                    "请遵守相关法律法规",
                    "使用前请确保了解后门攻击的风险",
                ],
            },
        }
    
    def _create_readme(self, metadata: Dict) -> str:
        """
        创建README文件内容
        
        Args:
            metadata: 元数据
            
        Returns:
            README内容
        """
        package_info = metadata["package_info"]
        dataset_info = metadata["dataset_info"]
        usage_info = metadata["usage_instructions"]
        
        readme = f"""# 中毒数据集包

## 基本信息
- 创建时间: {package_info['created_at']}
- 总文件数: {package_info['total_files']}
- 总大小: {package_info['total_size_mb']} MB
- 中毒文件数: {package_info['poisoned_files']}
- 干净文件数: {package_info['clean_files']}

## 数据集信息
- 触发器类型: {usage_info['trigger_type']}
- 目标标签: {usage_info['target_label']}
- 中毒率: {usage_info['poison_rate']:.1%}
- 原始数据集: {dataset_info.get('input_dir', 'N/A')}

## 使用说明
{usage_info['description']}

数据集结构: {usage_info['structure']}

中毒样本: {usage_info['poisoned_samples']}

## 安全警告
⚠️ 此数据集仅用于学术研究和安全测试
⚠️ 请勿用于恶意目的
⚠️ 请遵守相关法律法规
⚠️ 使用前请确保了解后门攻击的风险

## 详细信息
更多详细信息请查看 package_info.json 文件。

---
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        return readme
    
    def _get_package_info(
        self, 
        package_path: Path, 
        total_files: int, 
        original_size: int, 
        poisoned_files: int
    ) -> Dict:
        """
        获取压缩包信息
        
        Args:
            package_path: 压缩包路径
            total_files: 总文件数
            original_size: 原始大小
            poisoned_files: 中毒文件数
            
        Returns:
            压缩包信息
        """
        package_stat = package_path.stat()
        compression_ratio = 1 - (package_stat.st_size / original_size) if original_size > 0 else 0
        
        return {
            "package_path": str(package_path),
            "package_name": package_path.name,
            "package_size_bytes": package_stat.st_size,
            "package_size_mb": round(package_stat.st_size / (1024 * 1024), 2),
            "original_size_bytes": original_size,
            "original_size_mb": round(original_size / (1024 * 1024), 2),
            "compression_ratio": round(compression_ratio, 3),
            "total_files": total_files,
            "poisoned_files": poisoned_files,
            "created_at": datetime.fromtimestamp(package_stat.st_ctime).isoformat(),
        }
    
    def cleanup_old_packages(self, max_age_hours: int = 24, max_count: int = 100) -> List[str]:
        """
        清理旧的压缩包文件
        
        Args:
            max_age_hours: 最大保留时间（小时）
            max_count: 最大保留数量
            
        Returns:
            被删除的文件列表
        """
        if not self.downloads_dir.exists():
            return []
        
        # 获取所有压缩包文件
        packages = list(self.downloads_dir.glob("*.zip"))
        packages.sort(key=lambda p: p.stat().st_ctime, reverse=True)
        
        deleted_files = []
        current_time = datetime.now().timestamp()
        
        for i, package in enumerate(packages):
            package_age_hours = (current_time - package.stat().st_ctime) / 3600
            
            # 删除条件：超过最大数量或超过最大年龄
            should_delete = (i >= max_count) or (package_age_hours > max_age_hours)
            
            if should_delete:
                try:
                    package.unlink()
                    deleted_files.append(str(package))
                    logger.info(f"已删除旧压缩包: {package}")
                except Exception as e:
                    logger.warning(f"删除压缩包失败 {package}: {e}")
        
        return deleted_files
    
    def get_package_list(self) -> List[Dict]:
        """
        获取所有可用的压缩包列表
        
        Returns:
            压缩包信息列表
        """
        if not self.downloads_dir.exists():
            return []
        
        packages = []
        for package_path in self.downloads_dir.glob("*.zip"):
            try:
                stat = package_path.stat()
                packages.append({
                    "name": package_path.name,
                    "path": str(package_path),
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "age_hours": round((datetime.now().timestamp() - stat.st_ctime) / 3600, 1),
                })
            except Exception as e:
                logger.warning(f"获取压缩包信息失败 {package_path}: {e}")
        
        # 按创建时间倒序排列
        packages.sort(key=lambda p: p["created_at"], reverse=True)
        return packages