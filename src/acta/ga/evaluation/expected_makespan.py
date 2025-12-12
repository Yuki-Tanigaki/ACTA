from __future__ import annotations

import math
from typing import TYPE_CHECKING, Tuple

from acta.ga.evaluation.base import Evaluator
from acta.ga.representation import Individual
if TYPE_CHECKING:
    from acta.sim.model import ACTAScenarioModel
from acta.utils.logging_utils import get_logger

logger = get_logger(__name__)

Pos = Tuple[float, float]


def dist(a: Pos, b: Pos) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


class ExpectedMakespanEvaluator(Evaluator):
    """
    拠点が保持する info_state に基づいて、
    Individual が表す「Route & RepairFlags」から
    メイクスパンの期待値（決定論的近似）を評価する Evaluator。

    - 故障は考慮しない（healthy 前提）
    - タスクの完了状態・残作業量は commander.info_state.tasks を使用
    """

    def __init__(self, model: ACTAScenarioModel):
        self.model = model

        self.commander = model.command_center
        self.info_state = self.commander.info_state

        self.repair_pos: Pos = tuple(model.repair_depot_pos)
        self.repair_duration: float = float(model.cfg.repair_duration)

    # --------------------------------------------------
    # Evaluator interface
    # --------------------------------------------------
    def __call__(self, indiv: Individual) -> list[float]:
        """
        return:
            [expected_makespan]
        """
        makespan = 0.0

        for worker in self.model.workers.values():
            t_i = self._estimate_worker_time(worker.worker_id, indiv)
            makespan = max(makespan, t_i)

        return [makespan/self._time_scale()]

    # --------------------------------------------------
    # 正規化用時間スケールの計算
    # --------------------------------------------------
    def _time_scale(self) -> float:
        # 雑でもOK: 典型距離×移動時間 + 総残作業量×作業時間
        tasks = list(self.info_state.tasks.values())
        if not tasks:
            return 1.0

        # 平均移動距離の粗近似（拠点→タスク平均）
        cc_pos = tuple(self.model.cfg.command_center_pos)
        avg_d = sum(dist(cc_pos, tuple(t.position)) for t in tasks) / len(tasks)

        # 代表速度・代表スループット（平均でも、最小でも）
        speeds = [w.speed for w in self.model.workers.values()]
        thrpts = [w.throughput for w in self.model.workers.values()]
        v_move = max(min(speeds), 1e-9)
        v_work = max(min(thrpts), 1e-9)

        total_work = sum(float(getattr(t, "remaining_work")) for t in tasks)

        # ワーカー数で割って「同時並行」をざっくり反映
        n = max(len(self.model.workers), 1)

        T0 = (avg_d / v_move) + (total_work / v_work) / n
        return max(T0, 1e-6)

    # --------------------------------------------------
    # 内部: ワーカー1人分の所要時間見積
    # --------------------------------------------------
    def _count_done_tasks(self, route: list[int]) -> int:
        """info_state.tasks に基づき、route のうちすでに終了しているタスク数を数えるヘルパ"""
        done = 0
        for task_id in route:
            tinfo = self.info_state.tasks.get(task_id)
            if tinfo is None:
                msg = f"ExpectedMakespanEvaluator: task {task_id} info not found in command center."
                logger.error(msg)
                raise ValueError(msg)
            if getattr(tinfo, "status", None) == "done":
                done += 1
            else:
                break
        return done

    def _estimate_worker_time(self, worker_id: int, indiv: Individual) -> float:
        info_workers = self.info_state.workers
        info_tasks = self.info_state.tasks

        # 拠点にワーカー位置情報がない
        winfo = info_workers.get(worker_id)
        if winfo is None:
            msg = f"ExpectedMakespanEvaluator: worker {worker_id} info not found in command center."
            logger.error(msg)
            raise ValueError(msg)

        pos: Pos = winfo.position
        H: float = winfo.H

        speed = self.model.workers[worker_id].speed
        throughput = self.model.workers[worker_id].throughput
        speed_eta = self.model.workers[worker_id].speed_eta
        throughput_eta = self.model.workers[worker_id].throughput_eta
        fatigue_move = self.model.workers[worker_id].fatigue_move
        fatigue_work = self.model.workers[worker_id].fatigue_work

        route = indiv.routes[worker_id]
        repairs = indiv.repairs[worker_id] if worker_id < len(indiv.repairs) else []
        done_count = self._count_done_tasks(route)

        t = 0.0

        for idx in range(done_count, len(route)):
            task_id = route[idx]

            tinfo = info_tasks.get(task_id)

            # 目的タスクの座標
            task_pos: Pos = tuple(getattr(tinfo, "position"))

            # 1) まず次のタスクの repairs を確認
            do_repair = bool(repairs[idx]) if idx < len(repairs) else False

            # 2) 修復なら修復拠点に移動し修復
            if do_repair:
                # 現在地 -> 修復拠点
                p_fail = self.model.failure_model.failure_prob(H)
                speed_eff = (1.0 - p_fail) * speed + p_fail * (speed * speed_eta)
                move_dt = dist(pos, self.repair_pos) / speed_eff
                t += move_dt
                # H += fatigue_move * move_dt
                pos = self.repair_pos
                # 修復時間
                t += self.repair_duration
                H = 0.0

            # 3) その後タスクに移動
            p_fail = self.model.failure_model.failure_prob(H)
            speed_eff = (1.0 - p_fail) * speed + p_fail * (speed * speed_eta)
            move_dt = dist(pos, task_pos) / speed_eff
            t += move_dt
            pos = task_pos
            H += fatigue_move * move_dt

            # タスク処理時間
            remaining_work = float(getattr(tinfo, "remaining_work"))
            p_fail = self.model.failure_model.failure_prob(H)
            throughput_eff = (1.0 - p_fail) * throughput + p_fail * (throughput * throughput_eta)
            work_dt = remaining_work / throughput_eff
            t += work_dt
            H += fatigue_work * work_dt

        return t