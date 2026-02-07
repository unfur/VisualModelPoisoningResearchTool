import argparse
from pathlib import Path
from typing import Any, Dict, Optional

from rich.console import Console
from rich.table import Table

from src.core.backdoor_attacker import BackdoorAttacker
from src.database.operations import (
    export_experiment,
    init_db,
    list_experiments,
    get_experiment_detail,
)
from src.ui.config_manager import ConfigManager


console = Console()


class PoisoningCLI:
    def __init__(self):
        self.parser = self.create_parser()

    def create_parser(self):
        parser = argparse.ArgumentParser(
            description="视觉大模型数据中毒攻击研究工具",
            epilog="警告：本工具仅用于学术研究目的！",
        )

        subparsers = parser.add_subparsers(dest="command")

        # 数据集中毒命令
        poison_parser = subparsers.add_parser(
            "poison",
            help="生成中毒数据集",
        )
        poison_parser.add_argument(
            "--input-dir",
            required=True,
            help="原始数据集目录",
        )
        poison_parser.add_argument(
            "--output-dir",
            required=True,
            help="中毒数据集输出目录",
        )
        poison_parser.add_argument(
            "--trigger-type",
            default="badnet",
            choices=["badnet", "blend", "sig", "wa", "custom"],
            help="触发器类型",
        )
        poison_parser.add_argument(
            "--poison-rate",
            type=float,
            default=0.1,
            help="中毒比例（0.0-1.0）",
        )
        poison_parser.add_argument(
            "--target-label",
            type=int,
            help="目标标签（如不指定则随机选择）",
        )
        poison_parser.add_argument(
            "--batch-size",
            type=int,
            default=32,
            help="批处理大小",
        )
        poison_parser.add_argument(
            "--template",
            type=str,
            help="使用预定义攻击模板名称（可选）",
        )
        poison_parser.add_argument(
            "--poison-only",
            action="store_true",
            help="仅生成中毒图像（不复制干净样本）",
        )
        poison_parser.add_argument(
            "--poison-count",
            type=int,
            help="在仅中毒模式下生成的中毒图像数量",
        )
        poison_parser.add_argument(
            "--selection-mode",
            type=str,
            choices=["random", "sequential"],
            default="random",
            help="在仅中毒模式下选取样本的方式：random=随机，sequential=按顺序",
        )

        # 实验管理命令
        experiment_parser = subparsers.add_parser("experiment", help="实验管理")
        experiment_parser.add_argument(
            "--list",
            action="store_true",
            help="列出所有实验",
        )
        experiment_parser.add_argument(
            "--show",
            type=str,
            help="显示指定实验详情（experiment_id）",
        )
        experiment_parser.add_argument(
            "--export",
            type=str,
            help="导出实验数据（experiment_id）",
        )

        # 配置管理命令
        config_parser = subparsers.add_parser("config", help="配置管理")
        config_parser.add_argument(
            "--init",
            action="store_true",
            help="初始化配置文件",
        )
        config_parser.add_argument(
            "--validate",
            action="store_true",
            help="验证当前配置",
        )

        return parser

    def run(self, args=None):
        """运行CLI"""
        parsed_args = self.parser.parse_args(args)

        if parsed_args.command == "poison":
            self.handle_poison_command(parsed_args)
        elif parsed_args.command == "experiment":
            self.handle_experiment_command(parsed_args)
        elif parsed_args.command == "config":
            self.handle_config_command(parsed_args)
        else:
            self.parser.print_help()

    # ---------------- 命令处理 ----------------
    def handle_poison_command(self, args: argparse.Namespace) -> None:
        """处理数据集中毒命令。"""
        init_db()
        cfg_mgr = ConfigManager()

        attacker = BackdoorAttacker()

        extra_params: Dict[str, Any] = {
            "trigger_type": args.trigger_type,
        }

        # 如果指定模板，则合并模板参数
        if args.template:
            template = cfg_mgr.get_attack_template(args.template)
            if not template:
                console.print(f"[red]未找到攻击模板: {args.template}[/red]")
            else:
                extra_params.update(template)

        # 避免与显式参数重复（函数签名中已有的参数不再从模板中传入）
        for key in ["poison_rate", "target_label", "batch_size", "poison_only", "poison_count", "selection_mode"]:
            extra_params.pop(key, None)

        metadata = attacker.poison_dataset(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            poison_rate=args.poison_rate,
            target_label=args.target_label,
            batch_size=args.batch_size,
             poison_only=args.poison_only,
             poison_count=args.poison_count,
             selection_mode=args.selection_mode,
            **extra_params,
        )
        console.print("[green]中毒数据集生成完成[/green]")
        console.print(metadata)

    def handle_experiment_command(self, args: argparse.Namespace) -> None:
        """处理实验管理命令。"""
        init_db()

        if args.list:
            exps = list_experiments()
            table = Table(title="实验列表")
            table.add_column("Experiment ID")
            table.add_column("Name")
            table.add_column("Status")
            table.add_column("Created At")
            for e in exps:
                table.add_row(
                    e.experiment_id,
                    e.name or "",
                    e.status or "",
                    e.created_at.isoformat() if e.created_at else "",
                )
            console.print(table)
            return

        if args.show:
            data = get_experiment_detail(args.show)
            if not data:
                console.print(f"[red]未找到实验: {args.show}[/red]")
                return
            console.print(f"[bold]Experiment:[/bold] {data['experiment'].experiment_id}")
            console.print(f"Name: {data['experiment'].name}")
            console.print(f"Status: {data['experiment'].status}")
            console.print(f"Parameters: {data['experiment'].parameters}")
            console.print(f"Attacks: {len(data['attacks'])}")
            console.print(f"Evaluations: {len(data['evaluations'])}")
            return

        if args.export:
            out = export_experiment(args.export, f"./data/exports/{args.export}.json")
            console.print(f"[green]实验数据已导出至: {out}[/green]")
            return

        self.parser.print_help()

    def handle_config_command(self, args: argparse.Namespace) -> None:
        """处理配置管理命令。"""
        cfg_mgr = ConfigManager()

        if args.init:
            cfg_mgr.init_config_files()
            console.print("[green]配置目录已初始化（如配置文件已存在则不会覆盖）。[/green]")

        if args.validate:
            ok = cfg_mgr.validate_config()
            if ok:
                console.print("[green]配置校验通过。[/green]")
            else:
                console.print("[red]配置校验失败，请检查配置文件。[/red]")


