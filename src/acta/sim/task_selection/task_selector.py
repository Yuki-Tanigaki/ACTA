from __future__ import annotations
from typing import TYPE_CHECKING, Protocol, Optional

from acta.sim.agent import WorkerAgent, TaskAgent

if TYPE_CHECKING:
    from acta.sim.model import ACTAScenarioModel


class TaskSelector(Protocol):
    """
    タスク選択ポリシーのインタフェース。
    """

    def assign_tasks(self, model: ACTAScenarioModel) -> None:
        """
        モデル内の全ワーカーに対して target_task を更新する。
        """
        ...


class NearestIncompleteTaskSelector(TaskSelector):
    """
    - 各ワーカーについて
      - まだ終わっていないタスクの中から
      - 「現在位置から最も近いタスク」を 1 つ選ぶ
    """

    def assign_tasks(self, model: ACTAScenarioModel) -> None:
        # 未完了タスクだけを対象にする
        incomplete_tasks = [t for t in model.tasks.values() if t.status != "done"]

        for w in model.workers.values():
            # すでに修理中 or 修理に向かっているならポリシー側では何もしない
            if w.mode in ("go_repair", "repairing"):
                continue

            # 故障しているなら修理に行かせる
            if w.state == "failed":
                w.mode = "go_repair"
                w.target_task = None
                continue

            # ここから下は「通常のタスク割当」
            if not incomplete_tasks:
                w.target_task = None
                continue

            # 例: いままで通り最近傍タスクを選ぶ
            best_t = min(
                incomplete_tasks,
                key=lambda t: model.distance(w, t),
            )
            w.target_task = best_t
            w.mode = "work"

