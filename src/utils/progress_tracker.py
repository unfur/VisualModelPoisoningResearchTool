"""进度跟踪工具，用于在长时间运行的任务中跟踪和报告进度。"""
import json
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ProgressInfo:
    """进度信息数据类。"""
    task_id: str
    status: str  # 'running', 'completed', 'failed'
    current: int = 0
    total: int = 0
    message: str = ""
    percentage: float = 0.0
    error: Optional[str] = None

    def to_dict(self):
        """转换为字典。"""
        return asdict(self)


class ProgressTracker:
    """进度跟踪器，使用线程安全的字典存储进度信息。"""

    def __init__(self):
        self._progress: dict[str, ProgressInfo] = {}
        self._lock = threading.Lock()

    def create_task(self, task_id: str, total: int = 0, message: str = "") -> ProgressInfo:
        """创建新任务。"""
        with self._lock:
            info = ProgressInfo(
                task_id=task_id,
                status="running",
                total=total,
                message=message,
            )
            self._progress[task_id] = info
            return info

    def update(self, task_id: str, current: Optional[int] = None, total: Optional[int] = None, message: Optional[str] = None):
        """更新任务进度。"""
        with self._lock:
            if task_id not in self._progress:
                self.create_task(task_id, total=total or 0, message=message or "")
                return

            info = self._progress[task_id]
            if current is not None:
                info.current = current
            if total is not None:
                info.total = total
            if message is not None:
                info.message = message

            if info.total > 0:
                info.percentage = min(100.0, (info.current / info.total) * 100.0)
            else:
                info.percentage = 0.0

    def complete(self, task_id: str, message: Optional[str] = None):
        """标记任务完成。"""
        with self._lock:
            if task_id in self._progress:
                info = self._progress[task_id]
                info.status = "completed"
                info.current = info.total
                info.percentage = 100.0
                if message:
                    info.message = message

    def fail(self, task_id: str, error: str):
        """标记任务失败。"""
        with self._lock:
            if task_id in self._progress:
                info = self._progress[task_id]
                info.status = "failed"
                info.error = error

    def get(self, task_id: str) -> Optional[ProgressInfo]:
        """获取任务进度信息。"""
        with self._lock:
            return self._progress.get(task_id)

    def remove(self, task_id: str):
        """移除任务（清理）。"""
        with self._lock:
            self._progress.pop(task_id, None)


# 全局进度跟踪器实例
_global_tracker = ProgressTracker()


def get_tracker() -> ProgressTracker:
    """获取全局进度跟踪器。"""
    return _global_tracker

