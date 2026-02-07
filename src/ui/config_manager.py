from pathlib import Path
from typing import Any, Dict

import yaml

from src.utils.file_utils import ensure_dir


class ConfigManager:
    """配置文件管理与验证。"""

    DEFAULT_CONFIG_PATH = Path("config/default_config.yaml")
    ATTACK_TEMPLATES_PATH = Path("config/attack_templates.yaml")

    def __init__(self):
        self.default_config = self.load_yaml(self.DEFAULT_CONFIG_PATH)
        self.attack_templates = self.load_yaml(self.ATTACK_TEMPLATES_PATH)

    @staticmethod
    def load_yaml(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def init_config_files(self) -> None:
        """如不存在则创建配置目录与默认配置文件（简单占位实现）。"""
        ensure_dir(self.DEFAULT_CONFIG_PATH.parent)
        ensure_dir(self.ATTACK_TEMPLATES_PATH.parent)
        # 如果用户删除了配置文件，可以提示其从仓库模板恢复；
        # 这里不覆盖已有文件，避免误删用户自定义配置。

    def validate_config(self) -> bool:
        """验证当前配置是否包含必要字段。"""
        required_sections = ["system", "database", "backdoorbox", "attack_defaults", "security"]
        missing = [s for s in required_sections if s not in self.default_config]
        if missing:
            print(f"配置缺少必要段落: {missing}")
            return False
        return True

    def get_attack_template(self, name: str) -> Dict[str, Any]:
        return (self.attack_templates.get("templates") or {}).get(name, {})


