from __future__ import annotations
from typing import Literal, Optional

from mesa import Agent


class TaskAgent(Agent):
    def __init__(
        self,
        model,
        task_id: int,
        total_work: float,
        remaining_work: float,
    ):
        super().__init__(model)
        self.task_id = task_id
        self.total_work = total_work
        self.remaining_work = remaining_work
        if self.remaining_work <= 0:
            self.status: Literal["pending", "in_progress", "done"] = "done"
            self.finished_step: Optional[int] = 0
        else:
            self.status: Literal["pending", "in_progress", "done"] = "pending"
            self.finished_step: Optional[int] = None

        self.step_work_amount = 0.0      # このステップに受け取った仕事量
        self.has_worker_this_step = False  # このステップに誰かが作業したか
       

    def begin_step(self):
        """ステップの最初に呼ぶ: 一時変数をリセット"""
        self.step_work_amount = 0.0
        self.has_worker_this_step = False

    def add_work(self, amount: float):
        """ワーカーから呼ばれる: このステップに仕事を追加"""
        self.step_work_amount += amount
        self.has_worker_this_step = True

    def end_step(self):
        """ステップの最後に呼ぶ: remaining_work と status を更新"""
        if self.status == "done":
            return

        # 仕事量を反映
        self.remaining_work -= self.step_work_amount

        if self.remaining_work <= 0:
            self.remaining_work = 0
            self.status = "done"
            self.finished_step = self.model.steps
        elif self.has_worker_this_step:
            self.status = "in_progress"
        else:
            self.status = "pending"
