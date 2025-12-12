from __future__ import annotations
from typing import TYPE_CHECKING, Optional

from acta.ga.evaluation import ExpectedMakespanEvaluator, OutsidePathEvaluator
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
        elitism_rate: float,
        L_max: int,
        seed: int,
    ) -> None:
        self.interval = interval
        self.pop_size = pop_size
        self.generations = generations
        self.elitism_rate = elitism_rate
        self.L_max = L_max
        self.seed = seed

        # 一度 GA を回した結果のベスト個体を保持しておく
        self._best_individual: Optional[Individual] = None

        # 各ワーカーごとに「どの current_work まで repair を発動済みか」を覚えておく
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
            makespan_evaluator = ExpectedMakespanEvaluator(model)
            outside_evaluator = OutsidePathEvaluator(model)
            makespan = makespan_evaluator(ind)[0]
            outside = outside_evaluator(ind)[0]
            return [float(makespan), float(outside)]  # 小さいほど良い

        ga = SimpleGA(
            num_workers=num_workers,
            num_tasks=num_tasks,
            L_max=self.L_max,
            pop_size=self.pop_size,
            generations=self.generations,
            elitism_rate=self.elitism_rate,
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
        worker: WorkerAgent,
        indiv: Individual,
    ) -> int:
        """
        worker が持つローカル情報 (worker.info_state.tasks) に基づいて、
        indiv.routes[worker_id] の先頭から何個 'done' と認識しているかを返す。
        """
        wid = worker.worker_id
        route = indiv.routes[wid]

        current_work = 0
        for task_id in route:
            tinfo = worker.info_state.tasks.get(task_id)

            # ローカル情報がない / done だと分からない → そこから先は未完了扱い
            if tinfo is None or tinfo.status != "done":
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

        if indiv is None:
            msg = "GABasedTaskSelector: No plan available."
            logger.error(msg)
            raise ValueError(msg)

        tasks_by_id: dict[int, TaskAgent] = model.tasks

        for w in model.workers.values():
            worker_id = w.worker_id
            # 既に「修理に行く途中」「修理中」ならここでは何もしない
            if getattr(w, "mode", None) in ("go_repair", "repairing"):
                continue

            # このワーカーのルート
            route = indiv.routes[worker_id]

            if not route:
                w.target_task = None
                w.mode = "idle"
                continue

            # current_work を確認
            current_work = self._compute_current_work_for_worker(
                worker=w,
                indiv=indiv,
            )

            # ルートのタスクがすべて完了
            if current_work >= len(route):
                w.target_task = None
                w.mode = "idle"
                continue

            # RepairFlagsを取得
            repair_flags = indiv.repairs[worker_id]
            if not repair_flags:
                msg = "GA individual has empty repair flags for worker_id=%s" % worker_id
                logger.error(msg)
                raise ValueError(msg)

            # その worker が「直近で修理を発動した current_work」
            last_triggered = self._last_repair_index.get(worker_id, None)  # None=未発動

            go_repair = (
                0 <= current_work < len(repair_flags)
                and repair_flags[current_work]
                and last_triggered != current_work
            )

            if go_repair:
                w.target_task = None
                w.mode = "go_repair"
                self._last_repair_index[worker_id] = current_work
                continue

            # 次の仕事に向かう
            next_task_id = route[current_work]
            task = tasks_by_id.get(next_task_id)

            if task is None:
                msg = "Task id %s not found for worker_id=%s" % (next_task_id, worker_id)
                logger.error(msg)
                raise ValueError(msg)

            w.target_task = task
            w.mode = "work"