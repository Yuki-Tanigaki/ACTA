# src/acta/ga/representation.py
from __future__ import annotations
import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence
from acta.utils.logging_utils import get_logger

logger = get_logger(__name__)

# --- 型エイリアス ---

# タスクID（0 .. num_tasks-1）
TaskId = int

# ワーカーID（0 .. num_workers-1）
WorkerId = int

# ワーカー i のルート: (j_{i,1}, ..., j_{i,L_i})
Route = List[TaskId]

# ワーカー i の修理フラグ（True / False）
# 列: (r_{i,1}, ..., r_{i,L_max})
RepairFlags = List[bool]

# ルート順序レイヤ: 全ワーカーのタスク列集合 { (j_{i,1}, ..., j_{i,L_i}) }_{i in I}
RouteLayer = List[Route]

# 修理挿入レイヤ: 全ワーカーの修理フラグ集合 { (r_{i,1}, ..., r_{i,L_max}) }_{i in I}
RepairLayer = List[RepairFlags]


@dataclass
class Individual:
    """
    ACTA の GA における「2レイヤ個体表現」を実装したクラス。
    個体は
      1. ルート順序レイヤ（ワーカーごとのタスク部分順序列）
      2. 修理挿入レイヤ（ワーカーごとのTrue/Falseフラグ列）
    の 2 層構造で表現する。

    - ルート順序レイヤ:
        routes[i] = [j_{i,1}, j_{i,2}, ..., j_{i,L_i}]
      となるように、ワーカー i の担当タスク列を格納する。
      各タスクがちょうど1回だけどこかのワーカーに割り当てられる。

    - 修理挿入レイヤ:
        repairs[i] = [r_{i,1}, r_{i,2}, ..., r_{i,L_max}]
      となるように、ワーカー i の「タスク何個終了時に修理に立ち寄るか」
      を示すフラグ列を格納する。
      r_{i,l} は「タスク l 個を終えたあとに修理に行くなら True」
      を意味する。長さは全ワーカー共通の L_max （固定長）。
    """

    # --- 問題サイズ情報 ---
    num_workers: int                  # |I|
    num_tasks: int                    # |J|
    L_max: int                        # 1ワーカーあたりのタスク数の上限 L_max

    # --- レイヤ表現 ---
    routes: RouteLayer                # ルート順序レイヤ
    repairs: RepairLayer              # 修理挿入レイヤ

    # --- GA 用付帯情報 ---
    objectives: List[float] = field(default_factory=list)
    fitness: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        routes / repairs の形状チェック:
        - routes の長さ = num_workers
        - repairs の長さ = num_workers
        - repairs[i] の長さ = L_max
        - 全 routes を結合したとき 0..num_tasks-1 の完全順列になっているか
        """
        # --- 基本形状チェック ---
        if len(self.routes) != self.num_workers:
            msg = f"len(routes)={len(self.routes)} != num_workers={self.num_workers} "
            logger.error(msg)
            raise ValueError(msg)

        if len(self.repairs) != self.num_workers:
            msg = f"len(repairs)={len(self.repairs)} != num_workers={self.num_workers} "
            logger.error(msg)
            raise ValueError(msg)

        for i, flags in enumerate(self.repairs):
            if len(flags) != self.L_max:
                msg = f"repairs[{i}] length {len(flags)} != L_max={self.L_max}"
                logger.error(msg)
                raise ValueError(msg)

        # --- 完全順列チェック ---
        if not self.check_task_coverage():
            msg = ("Task assignment is not a valid permutation of 0..(num_tasks-1). ")
            logger.error(msg)
            raise ValueError(msg)


    # -----------------------
    # ヘルパメソッド群
    # -----------------------

    def copy(self) -> "Individual":
        """ディープコピーを返す。"""
        return Individual(
            num_workers=self.num_workers,
            num_tasks=self.num_tasks,
            L_max=self.L_max,
            routes=[list(route) for route in self.routes],
            repairs=[list(flags) for flags in self.repairs],
            objectives=list(self.objectives),
            fitness=copy.deepcopy(self.fitness),
        )

    @property
    def task_ids(self) -> List[TaskId]:
        """個体内に現れるタスクIDをフラットなリストで返す。"""
        ids: List[TaskId] = []
        for route in self.routes:
            ids.extend(route)
        return ids

    def count_tasks_per_worker(self) -> List[int]:
        """各ワーカー i の担当タスク数 L_i を返す。"""
        return [len(route) for route in self.routes]

    # -----------------------
    # クラスメソッド: 生成系
    # -----------------------

    @classmethod
    def from_routes_and_flags(
        cls,
        routes: Sequence[Sequence[TaskId]],
        repair_flags: Sequence[Sequence[bool]],
        num_tasks: int,
        L_max: int,
    ) -> "Individual":
        num_workers = len(routes)
        # 内部では List にコピーして保持
        route_layer: RouteLayer = [list(r) for r in routes]
        repair_layer: RepairLayer = [list(f) for f in repair_flags]

        return cls(
            num_workers=num_workers,
            num_tasks=num_tasks,
            L_max=L_max,
            routes=route_layer,
            repairs=repair_layer,
        )

    @classmethod
    def empty(
        cls,
        num_workers: int,
        num_tasks: int,
        L_max: int,
    ) -> "Individual":
        """
        全ワーカーのルート・修理フラグを空（もしくは False）の状態で初期化する。
        - routes[i] は空リスト []
        - repairs[i] は長さ L_max の False で初期化されたリスト
        """
        routes: RouteLayer = [[] for _ in range(num_workers)]
        repairs: RepairLayer = [[False] * L_max for _ in range(num_workers)]
        return cls(
            num_workers=num_workers,
            num_tasks=num_tasks,
            L_max=L_max,
            routes=routes,
            repairs=repairs,
        )

    # -----------------------
    # ユーティリティ
    # -----------------------

    def check_task_coverage(self) -> bool:
        """
        「各タスクがちょうど1回だけどこかのワーカーに割り当てられているか」をチェックする。
        """
        all_ids = self.task_ids
        if len(all_ids) != self.num_tasks:
            return False
        return sorted(all_ids) == list(range(self.num_tasks))
