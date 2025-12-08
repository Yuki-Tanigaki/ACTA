from __future__ import annotations
from typing import TYPE_CHECKING, Literal, Optional, cast
import math

from mesa import Agent
if TYPE_CHECKING:
    from acta.sim.agent.task_agent import TaskAgent
    from acta.sim.model import ACTAScenarioModel
from acta.sim.info_state import InfoState
from acta.utils.logging_utils import get_logger

logger = get_logger(__name__)

class WorkerAgent(Agent):
    model: "ACTAScenarioModel"

    def __init__(
        self,
        model: ACTAScenarioModel,
        worker_id: int,
        speed: float,
        throughput: float,
        speed_eta: float,
        throughput_eta: float,
        initial_H: float,
        fatigue_move: float,
        fatigue_work: float,
    ):
        super().__init__(model)
        self.worker_id = worker_id
        self.speed = speed               # u: 移動速度 [距離 / 時間]
        self.throughput = throughput     # v: タスク処理能力 [仕事量 / 時間]
        self.speed_eta = speed_eta         # η_u: 故障時の移動速度低下率
        self.throughput_eta = throughput_eta # η_v: 故障時の処理能力低下率

        # 故障関係のパラメータ・状態
        self.state: Literal["healthy", "failed"] = "healthy"
        self.H = initial_H
        self.fatigue_move = fatigue_move  # 移動による疲労蓄積率
        self.fatigue_work = fatigue_work  # 作業による疲労蓄積率

        # シミュレーション結果の集計用
        self.total_move_distance = 0.0

        # ターゲットタスク
        self.target_task: Optional[TaskAgent] = None

        # 修理関連の状態
        self.mode: Literal["work", "go_repair", "repairing"] = "work"
        self.repair_time_left: float = 0.0   # 残り修理時間

        # ワーカー自身の情報状態
        self.info_state = InfoState(workers={}, tasks={})
        # 次ステップ用の情報バッファ
        self._next_info_state: Optional[InfoState] = None

    # ------------------------------------------------------------------
    # 情報同期
    # ------------------------------------------------------------------
    def prepare_communicate(self) -> None:
        """
        近隣ワーカー・Commander からの情報をマージして _next_info_state に保持する。
        """
        # 自分が現在持っている情報をベースにコピーして開始
        next_state = InfoState(
            workers=dict(self.info_state.workers),
            tasks=dict(self.info_state.tasks),
        )
        # Commander からの情報をマージ（一方向）
        next_state.merge_from(self.model.command_center.info_state)

        # 近隣ワーカーからの情報をマージ（一方向）
        for nb in self.model.workers.values():
            if nb is self:
                continue
            # 通信可能な相手のみ
            if not self.model.can_communicate(self, nb):
                continue
            # nb の「現時点の info_state」のみ参照 (_next_info_state は見ない)
            next_state.merge_from(nb.info_state)

        # self.info_state には反映せず、バッファに保持
        self._next_info_state = next_state

    def communicate(self) -> None:
        """
        _next_info_state を本物の info_state に反映する。
        """
        assert self._next_info_state is not None
        self.info_state = self._next_info_state
        self._next_info_state = None

     # ------------------------------------------------------------------
    # ヘルパー
    # ------------------------------------------------------------------
    def _update_failure_state_at_step_start(self) -> None:
        """ターン開始時の H に基づいて故障判定."""
        if self.state != "healthy":
            return
        p_fail = self.model.failure_model.failure_prob(self.H)
        if self.random.random() < p_fail:
            self.state = "failed"

    def _current_speed(self) -> float:
        """故障状態を考慮した現在の移動速度."""
        if self.state == "failed":
            return self.speed * self.speed_eta
        return self.speed

    def _current_throughput(self) -> float:
        """故障状態を考慮した現在の処理能力."""
        if self.state == "failed":
            return self.throughput * self.throughput_eta
        return self.throughput

    def _move_towards(
        self,
        target_pos: tuple[float, float],
        dt: float,
        speed: float,
    ) -> tuple[bool, float, float]:
        """
        target_pos に向かって最大 speed * dt だけ移動する共通処理。

        戻り値:
            arrived: 目的地に到達したか（もともと居た場合も True）
            move_time: 実際に移動に使った時間
            remaining_dt: dt - move_time （目的地に着いた後に残る時間）
        ※ total_move_distance と H（移動分）はここで更新する
        """
        x, y = self.pos
        tx, ty = target_pos
        dx = tx - x
        dy = ty - y
        dist = math.hypot(dx, dy)

        # ほぼ同じ場所にいる → 移動なしで到達扱い
        if dist < 1e-8:
            return True, 0.0, dt

        max_step_dist = speed * dt

        # 移動可能距離内なら一気に到達
        if dist <= max_step_dist:
            # 到達
            self.model.space.move_agent(self, (tx, ty))
            self.total_move_distance += dist

            move_time = dist / speed if speed > 0.0 else 0.0
            remaining_dt = max(dt - move_time, 0.0)

            # 移動による疲労
            self.H += self.fatigue_move * move_time

            return True, move_time, remaining_dt

        # まだ到達しない → 向きだけ合わせて一歩進む
        if speed <= 0.0:
            # 速度0なら動けない（仕様次第でここを変えてもよい）
            # とりあえず「動けないが時間だけ経過して疲労も増えない」挙動にするなら:
            return False, 0.0, dt

        ratio = max_step_dist / dist
        new_x = x + dx * ratio
        new_y = y + dy * ratio

        self.model.space.move_agent(self, (new_x, new_y))
        self.total_move_distance += max_step_dist

        # dt 時間フルに移動している
        self.H += self.fatigue_move * dt

        return False, dt, 0.0

    # ------------------------------------------------------------------
    # ステップの挙動
    # ------------------------------------------------------------------
    def step(self) -> None:
        if self.model.all_tasks_done():
            return

        dt = self.model.time_step

        if self.mode == "repairing":
            self._repair(dt)
            return

        if self.mode == "go_repair":
            self._step_move_to_repair(dt)
            return

        if self.mode == "work":
            self._step_work(dt)

    # -------------------------------
    # 修理
    # -------------------------------
    def _repair(self, dt: float) -> None:
        self.repair_time_left -= dt
        if self.repair_time_left <= 0.0:
            self.state = "healthy"
            self.H = 0.0
            self.mode = "work"
            self.repair_time_left = 0.0

    # -------------------------------
    # 修理拠点へ移動（共通移動処理を利用）
    # -------------------------------
    def _step_move_to_repair(self, dt: float) -> None:
        rx, ry = self.model.repair_depot_pos
        current_speed = self._current_speed()

        arrived, move_time, remaining_dt = self._move_towards((rx, ry), dt, current_speed)

        if arrived:
            # 元の仕様通り: 到着したステップでは「残り時間があっても」修理開始だけにする
            self.mode = "repairing"
            self.repair_time_left = self.model.cfg.repair_duration

    # -------------------------------
    # 移動＋作業（共通移動処理を利用）
    # -------------------------------
    def _step_work(self, dt: float) -> None:
        if self.target_task is None:
            logger.error(f"Worker {self.worker_id} has no target task")
            raise RuntimeError("Worker has no target task")

        # ターン開始時の故障判定
        self._update_failure_state_at_step_start()

        current_speed = self._current_speed()
        current_throughput = self._current_throughput()

        # まずタスク位置に向かって移動
        tx, ty = self.target_task.pos
        arrived, move_time, remaining_dt = self._move_towards((tx, ty), dt, current_speed)

        # まだ到達していないなら、このステップは移動だけ
        if not arrived:
            return

        # ここからは「タスク地点にいる」
        # もともとタスク地点にいた場合: move_time=0, remaining_dt=dt
        # 到着した場合: move_time>0, remaining_dt = dt - move_time

        if remaining_dt <= 1e-8:
            # 到着したが作業する時間は残っていないステップ
            return

        if self.target_task.status != "done":
            self.target_task.add_work(current_throughput * remaining_dt)

        # 作業分の疲労
        self.H += self.fatigue_work * remaining_dt