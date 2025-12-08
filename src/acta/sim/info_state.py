from dataclasses import dataclass
from typing import Tuple, Literal


@dataclass
class WorkerInfo:
    worker_id: int
    position: Tuple[float, float]
    state: Literal["healthy", "failed"]
    H: float              # 累積稼働時間
    timestamp: int        # この情報が更新されたステップ

@dataclass
class TaskInfo:
    task_id: int
    position: Tuple[float, float]
    status: Literal["pending", "in_progress", "done"]
    total_work: float
    remaining_work: float
    timestamp: int

@dataclass
class InfoState:
    workers: dict[int, WorkerInfo]
    tasks: dict[int, TaskInfo]

    def sync_with(self, other: "InfoState") -> None:
        """自分と other の情報をタイムスタンプに基づいてマージ（双方向）。"""

        # --- Workers ---
        all_worker_ids = set(self.workers) | set(other.workers)
        for wid in all_worker_ids:
            a = self.workers.get(wid)
            b = other.workers.get(wid)
            if a is None and b is not None:
                self.workers[wid] = b
            elif b is None and a is not None:
                other.workers[wid] = a
            elif a is not None and b is not None:
                if a.timestamp < b.timestamp:
                    self.workers[wid] = b
                elif b.timestamp < a.timestamp:
                    other.workers[wid] = a

        # --- Tasks ---
        all_task_ids = set(self.tasks) | set(other.tasks)
        for tid in all_task_ids:
            a = self.tasks.get(tid)
            b = other.tasks.get(tid)
            if a is None and b is not None:
                self.tasks[tid] = b
            elif b is None and a is not None:
                other.tasks[tid] = a
            elif a is not None and b is not None:
                if a.timestamp < b.timestamp:
                    self.tasks[tid] = b
                elif b.timestamp < a.timestamp:
                    other.tasks[tid] = a

    def merge_from(self, other: "InfoState") -> None:
        """other 側のほうが新しい timestamp を持っている情報だけを採用して上書き"""
        # --- worker 情報 ---
        for wid, other_info in other.workers.items():
            my_info = self.workers.get(wid)
            if my_info is None or other_info.timestamp > my_info.timestamp:
                self.workers[wid] = other_info

        # --- task 情報 ---
        for tid, other_info in other.tasks.items():
            my_info = self.tasks.get(tid)
            if my_info is None or other_info.timestamp > my_info.timestamp:
                self.tasks[tid] = other_info