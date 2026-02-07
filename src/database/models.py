from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.sqlite import JSON as SQLITE_JSON
from sqlalchemy.ext.declarative import declarative_base


Base = declarative_base()


class Experiment(Base):
    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True)
    experiment_id = Column(String(64), unique=True)  # UUID
    name = Column(String(256))
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String(32))  # 'running', 'completed', 'failed'
    parameters = Column(SQLITE_JSON)


class AttackRecord(Base):
    __tablename__ = "attack_records"

    id = Column(Integer, primary_key=True)
    experiment_id = Column(String(64), ForeignKey("experiments.experiment_id"))
    input_dataset_path = Column(String(512))
    output_dataset_path = Column(String(512))
    trigger_type = Column(String(64))
    poison_rate = Column(Float)
    target_label = Column(Integer, nullable=True)
    total_samples = Column(Integer)
    poisoned_samples = Column(Integer)
    generated_at = Column(DateTime, default=datetime.utcnow)


class ModelEvaluation(Base):
    __tablename__ = "model_evaluations"

    id = Column(Integer, primary_key=True)
    experiment_id = Column(String(64), ForeignKey("experiments.experiment_id"))
    model_name = Column(String(128))
    attack_success_rate = Column(Float)
    clean_accuracy = Column(Float)
    evaluation_metrics = Column(SQLITE_JSON)
    evaluated_at = Column(DateTime, default=datetime.utcnow)


