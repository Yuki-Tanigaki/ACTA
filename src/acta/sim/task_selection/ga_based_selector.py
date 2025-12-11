from __future__ import annotations
from typing import TYPE_CHECKING, Optional

from acta.sim.agent import WorkerAgent, TaskAgent
from acta.sim.task_selection.task_selector import TaskSelector
from acta.ga.representation import Individual
from acta.ga.ga_core import SimpleGA
from acta.utils.logging_utils import get_logger

if TYPE_CHECKING:
    from acta.sim.model import ACTAScenarioModel

logger = get_logger(__name__)


class GABasedTaskSelector(TaskSelector):
    """
    GA によって「ワーカーごとのタスク担当ルート」と
    「タスク何個終了時に修理に行くか（RepairFlags）」を決め、
    その結果に基づいて各ステップで target_task / 修理指示 を更新する TaskSelector。
    """

    def __init__(
        self,
        interval: int,
        pop_size: int,
        generations: int,
        L_max: int,
        seed: int,
    ) -> None:
        self.interval = interval
        self.pop_size = pop_size
        self.generations = generations
        self.L_max = L_max
        self.seed = seed

        # 一度 GA を回した結果のベスト個体を保持しておく
        self._best_individual: Optional[Individual] = None

        # 各ワーカーごとに「どの current_work まで repair を発動済みか」を覚えておく
        # 初期値は -1 （まだ一度も repair していない）
        self._last_repair_index: dict[int, int] = {}

    # --------------------------------------------------
    # GA を実行してベスト個体（計画）を作る
    # --------------------------------------------------
    def _ensure_plan(self, model: ACTAScenarioModel) -> None:
        """
        GA を実行して新しいタスク計画を作る。
        """
        num_workers = len(model.workers)
        num_tasks = len(model.tasks)

        # --- 評価関数（簡易版） ---
        # 今は「ワーカーごとの担当タスク数の最大値」を最小化するだけ。
        def evaluate(ind: Individual) -> list[float]:
            counts = ind.count_tasks_per_worker()
            max_load = max(counts) if counts else 0.0
            return [float(max_load)]  # 小さいほど良い

        ga = SimpleGA(
            num_workers=num_workers,
            num_tasks=num_tasks,
            L_max=self.L_max,
            pop_size=self.pop_size,
            generations=self.generations,
            evaluate=evaluate,
            seed=self.seed,
        )

        self._best_individual = ga.run()
        logger.info(
            "[GABasedTaskSelector] GA plan updated. best_objectives=%s",
            self._best_individual.objectives,
        )

    # --------------------------------------------------
    # ヘルパ: current_work を計算する
    # --------------------------------------------------
    def _compute_current_work_for_worker(
        self,
        worker_id: int,
        indiv: Individual,
        model: ACTAScenarioModel,
    ) -> int:
        """
        ワーカー worker_id について、
        GA で決められた route のうち「どこまで完了しているか」を数える。

        - indiv.routes[worker_id] = [j_{i,1}, j_{i,2}, ..., j_{i,L_i}]
        - current_work は「先頭から current_work 個のタスクが status == 'done'」
          となっている最大の個数。
        - つまり「最後に終わったタスク番号が 1 なら 1」「まだ何も終わっていなければ 0」
        """
        route = indiv.routes[worker_id]
        tasks_by_id: dict[int, TaskAgent] = model.tasks

        current_work = 0
        for task_id in route:
            task = tasks_by_id.get(task_id)
            if task is None or task.status != "done":
                break
            current_work += 1

        return current_work

    # --------------------------------------------------
    # TaskSelector インタフェース実装
    # --------------------------------------------------
    def assign_tasks(self, model: ACTAScenarioModel) -> None:
        # --- GA 計画を用意 ---
        if (model.steps - 1) % self.interval == 0:
            self._ensure_plan(model)
        indiv = self._best_individual

        tasks_by_id: dict[int, TaskAgent] = model.tasks

        for w in model.workers.values():
            worker_id = w.worker_id
            # 既に「修理に行く途中」「修理中」ならここでは何もしない
            if getattr(w, "mode", None) in ("go_repair", "repairing"):
                continue

            # このワーカーのルート
            route = indiv.routes[worker_id]

            if not route:
                msg = "GA individual has empty route for worker_id=%s" % worker_id
                logger.error(msg)
                raise ValueError(msg)

            # --- 2. current_work を計算 ---
            current_work = self._compute_current_work_for_worker(
                worker_id=worker_id,
                indiv=indiv,
                model=model,
            )

            if current_work >= len(route):
                w.target_task = None
                continue

            # --- 3. RepairFlags[current_work] を確認 ---
            try:
                repair_flags = indiv.repairs[worker_id]
            except IndexError:
                repair_flags = []

            # そのワーカーについて、これまでに repair を発動した最大の current_work
            last_repair_idx = self._last_repair_index.get(worker_id, -1)

            go_repair = False
            if 0 <= current_work < len(repair_flags):
                # 「フラグが立っている & まだその current_work では repair していない」時だけ発動
                if repair_flags[current_work] and current_work > last_repair_idx:
                    go_repair = True

            if go_repair:
                w.target_task = None
                w.mode = "go_repair"
                # 今の current_work では repair したことを記録
                self._last_repair_index[worker_id] = current_work
                continue

            # --- 4. 次の仕事に向かう ---
            next_task_id = route[current_work]
            task = tasks_by_id.get(next_task_id)

            if task is None or task.status == "done":
                w.target_task = None
                continue

            w.target_task = task
            w.mode = "work"