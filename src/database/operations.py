import csv
import json
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.database.models import AttackRecord, Base, Experiment, ModelEvaluation
from src.utils.file_utils import ensure_dir


def _get_engine(db_url: Optional[str] = None):
    if db_url is None:
        # 默认使用 sqlite 本地文件
        db_path = Path("./data/research_db.sqlite")
        ensure_dir(db_path.parent)
        db_url = f"sqlite:///{db_path}"
    return create_engine(db_url, echo=False, future=True)


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_get_engine())


def init_db() -> None:
    """初始化数据库（建表）。"""
    engine = _get_engine()
    Base.metadata.create_all(bind=engine)


def create_experiment(
    name: str,
    description: str,
    parameters: Dict[str, Any],
    status: str = "running",
    db: Optional[Session] = None,
) -> Experiment:
    close_after = False
    if db is None:
        db = SessionLocal()
        close_after = True
    try:
        exp = Experiment(
            experiment_id=str(uuid.uuid4()),
            name=name,
            description=description,
            status=status,
            parameters=parameters,
        )
        db.add(exp)
        db.commit()
        db.refresh(exp)
        return exp
    finally:
        if close_after:
            db.close()


def update_experiment_status(
    experiment_id: str, status: str, db: Optional[Session] = None
) -> None:
    close_after = False
    if db is None:
        db = SessionLocal()
        close_after = True
    try:
        exp = (
            db.query(Experiment)
            .filter(Experiment.experiment_id == experiment_id)
            .one_or_none()
        )
        if exp is None:
            return
        exp.status = status
        db.commit()
    finally:
        if close_after:
            db.close()


def create_attack_record(
    experiment_id: str,
    input_dataset_path: str,
    output_dataset_path: str,
    trigger_type: str,
    poison_rate: float,
    target_label: Optional[int],
    total_samples: int,
    poisoned_samples: int,
    db: Optional[Session] = None,
) -> AttackRecord:
    close_after = False
    if db is None:
        db = SessionLocal()
        close_after = True
    try:
        rec = AttackRecord(
            experiment_id=experiment_id,
            input_dataset_path=input_dataset_path,
            output_dataset_path=output_dataset_path,
            trigger_type=trigger_type,
            poison_rate=poison_rate,
            target_label=target_label,
            total_samples=total_samples,
            poisoned_samples=poisoned_samples,
        )
        db.add(rec)
        db.commit()
        db.refresh(rec)
        return rec
    finally:
        if close_after:
            db.close()


def create_evaluation_record(
    experiment_id: str,
    model_name: str,
    attack_success_rate: float,
    clean_accuracy: float,
    evaluation_metrics: Dict[str, Any],
    db: Optional[Session] = None,
) -> ModelEvaluation:
    close_after = False
    if db is None:
        db = SessionLocal()
        close_after = True
    try:
        rec = ModelEvaluation(
            experiment_id=experiment_id,
            model_name=model_name,
            attack_success_rate=attack_success_rate,
            clean_accuracy=clean_accuracy,
            evaluation_metrics=evaluation_metrics,
        )
        db.add(rec)
        db.commit()
        db.refresh(rec)
        return rec
    finally:
        if close_after:
            db.close()


def list_experiments(db: Optional[Session] = None) -> List[Experiment]:
    close_after = False
    if db is None:
        db = SessionLocal()
        close_after = True
    try:
        return db.query(Experiment).order_by(Experiment.created_at.desc()).all()
    finally:
        if close_after:
            db.close()


def get_experiment_detail(
    experiment_id: str, db: Optional[Session] = None
) -> Dict[str, Any]:
    close_after = False
    if db is None:
        db = SessionLocal()
        close_after = True
    try:
        exp = (
            db.query(Experiment)
            .filter(Experiment.experiment_id == experiment_id)
            .one_or_none()
        )
        if exp is None:
            return {}
        attacks = (
            db.query(AttackRecord)
            .filter(AttackRecord.experiment_id == experiment_id)
            .all()
        )
        evals = (
            db.query(ModelEvaluation)
            .filter(ModelEvaluation.experiment_id == experiment_id)
            .all()
        )
        return {
            "experiment": exp,
            "attacks": attacks,
            "evaluations": evals,
        }
    finally:
        if close_after:
            db.close()


def export_experiment(
    experiment_id: str, out_path: str, fmt: str = "json"
) -> Path:
    """导出实验数据为 JSON 或 CSV。"""
    data = get_experiment_detail(experiment_id)
    out_file = Path(out_path)
    ensure_dir(out_file.parent)

    if fmt.lower() == "json":
        serializable = {
            "experiment": {
                "experiment_id": data["experiment"].experiment_id,
                "name": data["experiment"].name,
                "description": data["experiment"].description,
                "created_at": data["experiment"].created_at.isoformat()
                if data["experiment"].created_at
                else None,
                "status": data["experiment"].status,
                "parameters": data["experiment"].parameters,
            },
            "attacks": [
                {
                    "input_dataset_path": a.input_dataset_path,
                    "output_dataset_path": a.output_dataset_path,
                    "trigger_type": a.trigger_type,
                    "poison_rate": a.poison_rate,
                    "target_label": a.target_label,
                    "total_samples": a.total_samples,
                    "poisoned_samples": a.poisoned_samples,
                    "generated_at": a.generated_at.isoformat()
                    if a.generated_at
                    else None,
                }
                for a in data["attacks"]
            ],
            "evaluations": [
                {
                    "model_name": e.model_name,
                    "attack_success_rate": e.attack_success_rate,
                    "clean_accuracy": e.clean_accuracy,
                    "evaluation_metrics": e.evaluation_metrics,
                    "evaluated_at": e.evaluated_at.isoformat()
                    if e.evaluated_at
                    else None,
                }
                for e in data["evaluations"]
            ],
        }
        with out_file.open("w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)
    else:
        # 简单 CSV 导出：仅导出 AttackRecord
        with out_file.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "input_dataset_path",
                    "output_dataset_path",
                    "trigger_type",
                    "poison_rate",
                    "target_label",
                    "total_samples",
                    "poisoned_samples",
                    "generated_at",
                ]
            )
            for a in data["attacks"]:
                writer.writerow(
                    [
                        a.input_dataset_path,
                        a.output_dataset_path,
                        a.trigger_type,
                        a.poison_rate,
                        a.target_label,
                        a.total_samples,
                        a.poisoned_samples,
                        a.generated_at.isoformat() if a.generated_at else "",
                    ]
                )
    return out_file


def backup_database(backup_path: str) -> Path:
    """简单的数据库备份：复制 SQLite 文件。"""
    db_path = Path("./data/research_db.sqlite")
    if not db_path.exists():
        raise FileNotFoundError("数据库文件不存在，无法备份。")
    backup_file = Path(backup_path)
    ensure_dir(backup_file.parent)
    shutil.copy2(db_path, backup_file)
    return backup_file


def restore_database(backup_path: str) -> None:
    """从备份文件恢复数据库。"""
    src = Path(backup_path)
    if not src.exists():
        raise FileNotFoundError("备份文件不存在，无法恢复。")
    dst = Path("./data/research_db.sqlite")
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)


