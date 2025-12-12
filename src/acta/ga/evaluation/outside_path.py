from __future__ import annotations

import math
from typing import TYPE_CHECKING, Tuple

from acta.ga.evaluation.base import Evaluator
from acta.ga.representation import Individual
from acta.utils.logging_utils import get_logger

if TYPE_CHECKING:
    from acta.sim.model import ACTAScenarioModel

logger = get_logger(__name__)

Pos = Tuple[float, float]


def dist(a: Pos, b: Pos) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _outside_length_segment(a: Pos, b: Pos, center: Pos, R: float) -> float:
    """
    線分 a->b のうち、円（center, 半径R）の外側にある部分の長さを返す。
    - 円の内側: ||p-center|| <= R
    - 外側:     ||p-center|| >  R

    厳密に「線分と円の交点」を解いて外側部分長を計算する。
    """
    ax, ay = a
    bx, by = b
    cx, cy = center

    # a' = a-center, b' = b-center
    ax -= cx
    ay -= cy
    bx -= cx
    by -= cy

    dx = bx - ax
    dy = by - ay
    L = math.hypot(dx, dy)
    if L == 0.0:
        # 点だけ：外なら0ではなく「長さ」なので0
        return 0.0

    ra2 = ax * ax + ay * ay
    rb2 = bx * bx + by * by
    inside_a = ra2 <= R * R
    inside_b = rb2 <= R * R

    # 交点が無いケース
    if inside_a and inside_b:
        return 0.0  # 全部内側
    # まず「線分と円が交差しない」かを判定したいので、二次方程式で交点 t を求める
    # p(t) = a + t*(b-a), t in [0,1]
    # ||p(t)||^2 = R^2
    # (a + t d)·(a + t d) = R^2
    A = dx * dx + dy * dy
    B = 2.0 * (ax * dx + ay * dy)
    C = ra2 - R * R

    disc = B * B - 4.0 * A * C
    if disc < 0.0:
        # 交点なし：両端とも外なら全外、片端だけ外は理論上あり得ない（交差なしで内外が変わらない）
        return L if (not inside_a and not inside_b) else 0.0

    sqrt_disc = math.sqrt(disc)
    t1 = (-B - sqrt_disc) / (2.0 * A)
    t2 = (-B + sqrt_disc) / (2.0 * A)
    if t1 > t2:
        t1, t2 = t2, t1

    # 線分区間 [0,1] と交点区間の重なり
    lo = max(0.0, t1)
    hi = min(1.0, t2)

    if hi <= 0.0 or lo >= 1.0 or lo >= hi:
        # 交点が線分外：内外は端点と同じ
        return L if (not inside_a and not inside_b) else 0.0

    # 線分が円を横切る：内側部分は [lo,hi] の長さ
    inside_len = (hi - lo) * L
    outside_len = L - inside_len

    # ただし「接しているだけ」の数値誤差対策
    if outside_len < 0.0:
        outside_len = 0.0
    return outside_len


class OutsidePathEvaluator(Evaluator):
    """
    拠点（司令拠点）の通信半径 R (= model.communication_range) の外側にある
    ルート（移動経路）の合計長を評価する。

    - info_state のワーカー位置・タスク位置を使用
    - 既に done のタスクまではスキップ（ExpectedMakespanEvaluator と同様）
    - 修理フラグが立っている区間では「現在地->修理拠点->次タスク」の移動を評価
      （修理中の滞在時間は距離ではないので無視）
    """

    def __init__(self, model: ACTAScenarioModel):
        self.model = model

        self.commander = model.command_center
        self.info_state = self.commander.info_state

        # 通信中心は「拠点の現在位置」
        self.center_pos: Pos = tuple(getattr(self.commander, "pos"))
        self.R: float = float(model.communication_range)

        self.repair_pos: Pos = tuple(model.repair_depot_pos)

    def __call__(self, indiv: Individual) -> list[float]:
        total_outside = 0.0
        for worker in self.model.workers.values():
            total_outside += self._outside_length_for_worker(worker.worker_id, indiv)
        return [total_outside/self._length_scale(indiv)]

    # --------------------------------------------------
    # helpers
    # --------------------------------------------------
    def _length_scale(self, indiv: Individual) -> float:
        tasks = self.info_state.tasks
        # “未完了タスク”だけ
        remain_positions: list[Pos] = []
        for tid, tinfo in tasks.items():
            if getattr(tinfo, "status", None) != "done":
                remain_positions.append(tuple(getattr(tinfo, "position")))

        if not remain_positions:
            return 1.0

        # 典型距離：拠点→タスク平均
        avg_d = sum(dist(self.center_pos, p) for p in remain_positions) / len(remain_positions)
        avg_d = max(avg_d, 0.5 * self.R)  # あまりにも近いときの下限

        # 典型ルート長：1タスクあたり「拠点から行く距離」+「タスク間移動っぽい距離(同程度とみなす)」
        # ※雑だけどスケール合わせには十分
        per_task = 2.0 * avg_d

        n_rem = len(remain_positions)
        L0 = per_task * n_rem

        # 修理拠点が拠点から離れている場合の上積み（雑）
        # 「修理に1回行くとしたら」程度の代表距離を少し足す
        repair_round = 2.0 * dist(self.center_pos, self.repair_pos)
        n_workers = max(len(self.model.workers), 1)
        L0 += n_workers * repair_round

        return max(L0, 1e-6)

    def _count_done_tasks(self, route: list[int]) -> int:
        done = 0
        for task_id in route:
            tinfo = self.info_state.tasks.get(task_id)
            if tinfo is None:
                msg = f"OutsidePathEvaluator: task {task_id} info not found in command center."
                logger.error(msg)
                raise ValueError(msg)
            if getattr(tinfo, "status", None) == "done":
                done += 1
            else:
                break
        return done

    def _outside_length_for_worker(self, worker_id: int, indiv: Individual) -> float:
        info_workers = self.info_state.workers
        info_tasks = self.info_state.tasks

        winfo = info_workers.get(worker_id)
        if winfo is None:
            msg = f"OutsidePathEvaluator: worker {worker_id} info not found in command center."
            logger.error(msg)
            raise ValueError(msg)

        pos: Pos = winfo.position

        route = indiv.routes[worker_id]
        repairs = indiv.repairs[worker_id] if worker_id < len(indiv.repairs) else []
        done_count = self._count_done_tasks(route)

        outside_len = 0.0

        for idx in range(done_count, len(route)):
            task_id = route[idx]
            tinfo = info_tasks.get(task_id)
            if tinfo is None:
                msg = f"OutsidePathEvaluator: task {task_id} info not found in command center."
                logger.error(msg)
                raise ValueError(msg)

            task_pos: Pos = tuple(getattr(tinfo, "position"))

            do_repair = bool(repairs[idx]) if idx < len(repairs) else False
            if do_repair:
                # pos -> repair
                outside_len += _outside_length_segment(pos, self.repair_pos, self.center_pos, self.R)
                pos = self.repair_pos

            # pos -> task
            outside_len += _outside_length_segment(pos, task_pos, self.center_pos, self.R)
            pos = task_pos

        return outside_len
