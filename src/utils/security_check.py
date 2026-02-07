from pathlib import Path
from typing import Any, Dict


class SecurityChecker:
    @staticmethod
    def validate_usage_context() -> None:
        """验证使用环境（占位实现）。

        实际环境中可结合环境变量、许可证文件或内网限制等机制进行更严格检查。
        这里给出明确的伦理提示与轻量级检查，避免误用。
        """
        # 这里可以根据需要加入更严格的逻辑，例如：
        # - 检查环境变量 RESEARCH_ENV 是否为 "true"
        # - 检查当前用户是否在允许列表中
        # 当前实现仅打印提示，不强制中止。
        print(
            "提示：请确保当前使用场景为学术研究、安全测试或获得授权的环境。"
        )

    @staticmethod
    def validate_dataset_path(path: str) -> None:
        """验证数据集路径合法性，防止访问系统关键文件。"""
        p = Path(path).resolve()
        # 仅允许在项目 data 目录或其子目录下进行读写
        allowed_root = Path("./data").resolve()
        if not str(p).startswith(str(allowed_root)):
            raise ValueError(
                f"数据集路径不在允许的 data 目录下: {p} (允许根目录: {allowed_root})"
            )

    @staticmethod
    def validate_attack_parameters(params: Dict[str, Any]) -> None:
        """验证攻击参数安全性。"""
        poison_rate = float(params.get("poison_rate", 0.0))
        if poison_rate > 0.3:
            raise ValueError("中毒比例超过安全限制(0.3)")
        # 其他安全检查可以在此扩展，例如：
        # - 数据集规模限制
        # - 目标标签范围检查

    @staticmethod
    def generate_ethics_warning() -> str:
        """生成伦理警告。"""
        warning = """
⚠️⚠️⚠️  伦理警告  ⚠️⚠️⚠️

本工具仅用于以下合法目的：
1. 学术研究和论文实验
2. 安全漏洞测试（需获得明确授权）
3. 防御机制开发和验证

严禁用于：
1. 攻击实际生产系统
2. 制造恶意软件
3. 任何违法或未经授权的活动

使用者需承担全部法律责任！
"""
        return warning


def require_confirmation(action_description: str) -> bool:
    """要求用户确认危险操作。"""
    print(f"\n即将执行：{action_description}")
    response = input("确认执行？(输入 'YES' 确认): ")
    return response == "YES"


