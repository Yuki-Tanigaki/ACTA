# src/acta/ga/initialization.py
from __future__ import annotations

import random
from typing import List, Sequence, Optional

from acta.utils.logging_utils import get_logger
from acta.ga.representation import Individual, TaskId, RouteLayer, RepairLayer

logger = get_logger(__name__)


def _assign_tasks_to_workers_randomly(
    num_workers: int,
    num_tasks: int,
    L_max: int,
    rng: random.Random,
) -> RouteLayer:
    """
    各タスクをランダムにワーカーに割り当てて routes を作る。
    - 各タスクIDはちょうど1度だけ出現
    - 各ワーカーのタスク数は L_max 以下
    """
    if num_workers * L_max < num_tasks:
        msg = (
            f"num_workers * L_max = {num_workers * L_max} < num_tasks={num_tasks}. "
            "この条件では可行な割当が存在しません。"
        )
        logger.error(msg)
        raise ValueError(msg)

    # 0..num_tasks-1 をシャッフル
    task_ids: List[TaskId] = list(range(num_tasks))
    rng.shuffle(task_ids)

    # 各ワーカーのルートを空で初期化
    routes: RouteLayer = [[] for _ in range(num_workers)]

    # タスクを1つずつ、まだ余裕のあるワーカーにランダムに割り当てる
    for t_id in task_ids:
        # まだ L_max に達していないワーカーだけ候補にする
        candidates = [i for i in range(num_workers) if len(routes[i]) < L_max]
        if not candidates:
            msg = "有効な候補ワーカーが存在しません。L_max 設定が不適切な可能性があります。"
            logger.error(msg)
            raise RuntimeError(msg)

        i = rng.choice(candidates)
        routes[i].append(t_id)

    return routes


def _generate_random_repair_flags(
    num_workers: int,
    L_max: int,
    rng: random.Random,
    repair_prob: float,
) -> RepairLayer:
    """
    各ワーカーについて、長さ L_max の修理フラグ列をランダム生成する。
    実際に使われるのは「そのワーカーのタスク数 L_i 以下のインデックス」だけ。
    """
    repairs: RepairLayer = []
    for _ in range(num_workers):
        flags = [rng.random() < repair_prob for _ in range(L_max)]
        repairs.append(flags)
    return repairs


def random_individual(
    num_workers: int,
    num_tasks: int,
    L_max: int,
    rng: random.Random,
    repair_prob: float,
) -> Individual:
    """
    1 個体だけランダム生成して返すユーティリティ関数。
    """
    routes = _assign_tasks_to_workers_randomly(
        num_workers=num_workers,
        num_tasks=num_tasks,
        L_max=L_max,
        rng=rng,
    )
    repairs = _generate_random_repair_flags(
        num_workers=num_workers,
        L_max=L_max,
        rng=rng,
        repair_prob=repair_prob,
    )

    ind = Individual(
        num_workers=num_workers,
        num_tasks=num_tasks,
        L_max=L_max,
        routes=routes,
        repairs=repairs,
    )

    logger.debug(
        "Created random individual: "
        "task_counts=%s, repairs_example=%s",
        ind.count_tasks_per_worker(),
        ind.repairs[0] if ind.repairs else [],
    )
    return ind


def random_population(
    population_size: int,
    num_workers: int,
    num_tasks: int,
    L_max: int,
    rng: random.Random,
    repair_prob: float,
) -> List[Individual]:
    """
    指定した個体数ぶん、ランダムな個体を生成して返す。

    まだ評価も交叉も突然変異も行わず、
    「初期集団を作るだけ」の GA 骨格用関数。
    """
    population: List[Individual] = []
    for k in range(population_size):
        # 個体ごとにシードをずらしても良いし、
        # 同じ rng インスタンスを使っても良い。
        ind_seed = rng.randrange(10**9)
        ind = random_individual(
            num_workers=num_workers,
            num_tasks=num_tasks,
            L_max=L_max,
            rng=rng,
            repair_prob=repair_prob,
        )
        population.append(ind)

    logger.info(
        "Random population generated: size=%d, num_workers=%d, num_tasks=%d, L_max=%d",
        population_size,
        num_workers,
        num_tasks,
        L_max,
    )
    return population
