import logging
import threading
import uuid
from pathlib import Path
from typing import Any, Dict

from flask import Flask, jsonify, redirect, render_template, request, url_for, flash, send_file, abort

from src.core.backdoor_attacker import BackdoorAttacker
from src.core.dataset_generator import DatasetGenerator
from src.database.operations import (
    get_experiment_detail,
    init_db,
    list_experiments,
)
from src.ui.config_manager import ConfigManager
from src.utils.progress_tracker import get_tracker
from src.utils.security_check import SecurityChecker
from src.utils.dataset_packager import DatasetPackager
import yaml


logger = logging.getLogger(__name__)


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).resolve().parents[2] / "templates"),
        static_folder=str(Path(__file__).resolve().parents[2] / "static"),
    )
    # 简单的开发环境密钥（如在生产环境请替换为安全随机值）
    app.secret_key = "visual-model-poisoning-research-tool"

    # 加载配置文件
    config_path = Path(__file__).resolve().parents[2] / "config" / "default_config.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # 获取下载目录配置（使用绝对路径确保可靠性）
    downloads_dir_config = config.get("system", {}).get("downloads_dir", "downloads")
    if downloads_dir_config.startswith("./"):
        downloads_dir = str(Path(__file__).resolve().parents[2] / downloads_dir_config[2:])
    else:
        downloads_dir = str(Path(__file__).resolve().parents[2] / downloads_dir_config)

    # Flask 3.x 中已移除 before_first_request，这里在应用创建时直接做一次初始化
    init_db()
    print(SecurityChecker.generate_ethics_warning())

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/poison", methods=["GET", "POST"])
    def poison():
        cfg_mgr = ConfigManager()
        templates = (cfg_mgr.attack_templates.get("templates") or {}).keys()
        if request.method == "POST":
            form = request.form
            input_dir = form.get("input_dir", "").strip()
            output_dir = form.get("output_dir", "").strip()
            trigger_type = form.get("trigger_type", "badnet")
            poison_rate = float(form.get("poison_rate") or 0.1)
            target_label = form.get("target_label")
            batch_size = int(form.get("batch_size") or 32)
            template_name = form.get("template") or None
            preset = form.get("dataset_preset") or ""
            poison_mode = form.get("poison_mode") or ""  # "poison_only" 或 "poison_subset"
            poison_only = poison_mode == "poison_only"
            poison_subset = poison_mode == "poison_subset"
            poison_count_raw = form.get("poison_count")
            poison_count = int(poison_count_raw) if poison_count_raw not in (None, "", "0") else None
            selection_mode = form.get("selection_mode") or "random"
            encoder_path = form.get("encoder_path", "").strip() or None

            target_label_val = int(target_label) if target_label not in (None, "") else None

            # 生成任务ID
            task_id = str(uuid.uuid4())

            def run_poison_task():
                """在后台线程中执行中毒任务。"""
                tracker = get_tracker()
                # 提前创建任务，避免前端长时间停留在初始化
                tracker.create_task(task_id, total=0, message="初始化任务...")
                try:
                    # 如选择了预置数据集，则在 ./data/raw_datasets/ 下自动下载并转换为 ImageFolder
                    if preset:
                        base_raw = Path("./data/raw_datasets").resolve()
                        gen = DatasetGenerator()
                        # 使用同一 task_id 进行进度展示，避免前端停留在“初始化”
                        tracker.update(task_id, message="正在下载预置数据集...")
                        if preset == "cifar10":
                            prepared_dir = gen.prepare_cifar10_imagefolder(base_raw, task_id=task_id)
                        elif preset == "mnist":
                            prepared_dir = gen.prepare_mnist_imagefolder(base_raw, task_id=task_id)
                        else:
                            raise ValueError(f"不支持的预置数据集: {preset}")
                        input_dir_final = str(prepared_dir)
                    else:
                        if not input_dir:
                            raise ValueError("未指定原始数据集目录，且未选择预置数据集。")
                        input_dir_final = input_dir

                    attacker = BackdoorAttacker()
                    extra_params: Dict[str, Any] = {"trigger_type": trigger_type}
                    if template_name:
                        t_cfg = cfg_mgr.get_attack_template(template_name)
                        extra_params.update(t_cfg)
                    
                    # ISSBA 特定参数
                    if trigger_type.lower() == "issba" and encoder_path:
                        extra_params["encoder_path"] = encoder_path

                    # 避免与显式参数重复
                    for key in ["poison_rate", "target_label", "batch_size", "poison_only", "poison_count", "selection_mode"]:
                        extra_params.pop(key, None)

                    attacker.poison_dataset(
                        input_dir=input_dir_final,
                        output_dir=output_dir,
                        poison_rate=poison_rate,
                        target_label=target_label_val,
                        batch_size=batch_size,
                        task_id=task_id,
                        poison_only=poison_only,
                        poison_subset=poison_subset,
                        poison_count=poison_count,
                        selection_mode=selection_mode,
                        **extra_params,
                    )
                except Exception as e:  # noqa: BLE001
                    tracker.fail(task_id, str(e))

            # 在后台线程中启动任务
            thread = threading.Thread(target=run_poison_task, daemon=True)
            thread.start()

            # 返回任务ID，前端将轮询进度
            return jsonify({"task_id": task_id, "status": "started"})

        return render_template("poison_form.html", templates=templates)

    @app.route("/progress/<task_id>")
    def get_progress(task_id: str):
        """获取任务进度。"""
        tracker = get_tracker()
        progress = tracker.get(task_id)
        if progress:
            return jsonify(progress.to_dict())
        return jsonify({"status": "not_found"}), 404

    @app.route("/poison_result/<task_id>")
    def poison_result(task_id: str):
        """显示中毒结果页面。"""
        tracker = get_tracker()
        progress = tracker.get(task_id)
        if not progress:
            flash("未找到任务记录。", "error")
            return redirect(url_for("poison"))
        
        if progress.status == "failed":
            flash(f"任务失败: {progress.error}", "error")
            return redirect(url_for("poison"))
        
        if progress.status != "completed":
            flash("任务尚未完成。", "warning")
            return redirect(url_for("poison"))

        # 任务完成，显示结果（这里简化处理，实际可以从数据库读取）
        return render_template("poison_result.html", task_id=task_id, message=progress.message)

    @app.route("/download/<task_id>")
    def download_dataset(task_id: str):
        """下载中毒数据集压缩包。"""
        try:
            # 获取任务进度信息
            tracker = get_tracker()
            progress = tracker.get(task_id)
            
            if not progress or progress.status != "completed":
                flash("任务未完成或不存在，无法下载。", "error")
                return redirect(url_for("poison"))
            
            # 查找对应的压缩包文件
            packager = DatasetPackager(downloads_dir=downloads_dir)
            packages = packager.get_package_list()
            
            # 根据task_id查找对应的压缩包
            target_package = None
            for package in packages:
                if task_id[:8] in package["name"]:  # 使用task_id前8位匹配
                    target_package = package
                    break
            
            if not target_package:
                flash("未找到对应的下载文件。", "error")
                return redirect(url_for("poison_result", task_id=task_id))
            
            package_path = Path(target_package["path"])
            if not package_path.exists():
                flash("下载文件不存在或已被删除。", "error")
                return redirect(url_for("poison_result", task_id=task_id))
            
            # 发送文件
            return send_file(
                package_path,
                as_attachment=True,
                download_name=target_package["name"],
                mimetype='application/zip'
            )
            
        except Exception as e:
            logger.error(f"下载文件时出错: {e}")
            flash("下载失败，请稍后重试。", "error")
            return redirect(url_for("poison_result", task_id=task_id))

    @app.route("/downloads")
    def downloads_list():
        """显示所有可下载的压缩包列表。"""
        try:
            packager = DatasetPackager(downloads_dir=downloads_dir)
            packages = packager.get_package_list()
            return render_template("downloads.html", packages=packages)
        except Exception as e:
            logger.error(f"获取下载列表时出错: {e}")
            flash("获取下载列表失败。", "error")
            return redirect(url_for("index"))

    @app.route("/download_direct/<package_name>")
    def download_direct(package_name: str):
        """直接下载指定的压缩包。"""
        try:
            # 安全检查：只允许下载.zip文件，且文件名不能包含路径分隔符
            if not package_name.endswith('.zip') or '/' in package_name or '\\' in package_name:
                abort(400, "无效的文件名")
            
            packager = DatasetPackager(downloads_dir=downloads_dir)
            package_path = packager.downloads_dir / package_name
            
            if not package_path.exists():
                abort(404, "文件不存在")
            
            return send_file(
                package_path,
                as_attachment=True,
                download_name=package_name,
                mimetype='application/zip'
            )
            
        except Exception as e:
            logger.error(f"直接下载文件时出错: {e}")
            abort(500, "下载失败")

    @app.route("/api/package_info/<task_id>")
    def get_package_info(task_id: str):
        """获取任务对应的压缩包信息（API接口）。"""
        try:
            packager = DatasetPackager(downloads_dir=downloads_dir)
            packages = packager.get_package_list()
            
            # 查找对应的压缩包
            for package in packages:
                if task_id[:8] in package["name"]:
                    return jsonify({
                        "status": "success",
                        "package": package
                    })
            
            return jsonify({
                "status": "not_found",
                "message": "未找到对应的压缩包"
            }), 404
            
        except Exception as e:
            logger.error(f"获取压缩包信息时出错: {e}")
            return jsonify({
                "status": "error",
                "message": "获取信息失败"
            }), 500

    @app.route("/experiments")
    def experiments():
        exps = list_experiments()
        return render_template("experiments.html", experiments=exps)

    @app.route("/experiments/<experiment_id>")
    def experiment_detail(experiment_id: str):
        data = get_experiment_detail(experiment_id)
        if not data:
            flash("未找到对应实验记录。", "error")
            return redirect(url_for("experiments"))
        return render_template(
            "experiment_detail.html",
            exp=data["experiment"],
            attacks=data["attacks"],
            evaluations=data["evaluations"],
        )

    return app


