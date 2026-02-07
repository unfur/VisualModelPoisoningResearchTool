import sys

from src.ui.cli_interface import PoisoningCLI
from src.utils.security_check import SecurityChecker, require_confirmation


def main():
    # 显示伦理与安全警告
    print("=" * 70)
    print("视觉大模型数据中毒攻击研究工具 (VisualModelPoisoningResearchTool)")
    print("警告：本工具仅用于学术研究和安全测试目的！")
    print("禁止用于任何非法或恶意用途！")
    print("=" * 70)
    print(SecurityChecker.generate_ethics_warning())

    # 基本环境校验
    SecurityChecker.validate_usage_context()

    cli = PoisoningCLI()

    # 无参数时展示帮助
    if len(sys.argv) == 1:
        cli.parser.print_help()
        sys.exit(1)

    # 对潜在危险命令进行额外确认（例如 poison）
    if "poison" in sys.argv:
        if not require_confirmation("执行数据集中毒攻击实验"):
            print("操作已取消。")
            sys.exit(0)

    cli.run()


if __name__ == "__main__":
    main()


