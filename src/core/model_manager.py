from pathlib import Path
from typing import Any

import torch


class ModelManager:
    """模型加载与评估管理。"""

    def load_model(self, model_path: str) -> Any:
        p = Path(model_path)
        if not p.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        return torch.load(p, map_location="cpu")

    def evaluate(self, model, dataloader, device: str = "cpu"):
        # 占位评估逻辑：返回干净准确率
        model.to(device)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, preds = outputs.max(1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        return correct / total if total > 0 else 0.0


