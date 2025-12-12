from __future__ import annotations

from typing import List
import random

from acta.ga.representation import Individual
from acta.utils.logging_utils import get_logger

logger = get_logger(__name__)


def crossover(
    parent_a: Individual,
    parent_b: Individual,
    rng: random.Random,
) -> Individual:
    # routes
    child_routes = route_layer_srex_like_crossover_routes(parent_a, parent_b, rng)
    child_routes = repair_routes_feasibility_routes(
        child_routes,
        num_tasks=parent_a.num_tasks,
        num_workers=parent_a.num_workers,
        L_max=parent_a.L_max,
        rng=rng,
    )

    # repairs
    child_repairs = repair_layer_uniform_crossover_repairs(parent_a, parent_b, rng)

    # assemble
    child = Individual(
        num_workers=parent_a.num_workers,
        num_tasks=parent_a.num_tasks,
        L_max=parent_a.L_max,
        routes=child_routes,
        repairs=child_repairs,
    )
    return child


def route_similarity(route_a: List[int], route_b: List[int]) -> int:
    return len(set(route_a) & set(route_b))


def route_layer_srex_like_crossover_routes(
    parent_a: Individual,
    parent_b: Individual,
    rng: random.Random,
    p_select: float = 0.5,
) -> List[List[int]]:
    """
    routes だけ返す SREX-like 交叉。
    """
    if parent_a.num_workers != parent_b.num_workers:
        raise ValueError("Parents must have the same num_workers.")
    if parent_a.num_tasks != parent_b.num_tasks:
        raise ValueError("Parents must have the same num_tasks.")
    if parent_a.L_max != parent_b.L_max:
        raise ValueError("Parents must have the same L_max.")

    W = parent_a.num_workers

    # ベースはAのroutesコピー
    child_routes: List[List[int]] = [list(r) for r in parent_a.routes]

    # 交換対象S
    S = [i for i in range(W) if rng.random() < p_select]
    if not S:
        S = [rng.randrange(W)]

    routes_a = parent_a.routes
    routes_b = parent_b.routes

    for i in S:
        base_route = routes_a[i]

        indices = list(range(W))
        rng.shuffle(indices)

        best_k = indices[0]
        best_sim = -1
        for k in indices:
            sim = route_similarity(base_route, routes_b[k])
            if sim > best_sim:
                best_sim = sim
                best_k = k

        child_routes[i] = list(routes_b[best_k])

    return child_routes

def repair_routes_feasibility_routes(
    routes: List[List[int]],
    num_tasks: int,
    num_workers: int,
    L_max: int,
    rng: random.Random,
) -> List[List[int]]:
    """
    routes を修復する。
    """
    routes = [list(r) for r in routes]

    T = num_tasks
    W = num_workers

    appearances: List[List[tuple[int, int]]] = [[] for _ in range(T)]
    for i, route in enumerate(routes):
        for pos, task in enumerate(route):
            if 0 <= task < T:
                appearances[task].append((i, pos))

    # 重複削除（最後を残す）
    for t in range(T):
        if len(appearances[t]) <= 1:
            continue

        to_remove = appearances[t][:-1]
        for (wi, pos) in sorted(to_remove, key=lambda x: (x[0], -x[1])):
            route = routes[wi]
            if 0 <= pos < len(route) and route[pos] == t:
                route.pop(pos)
            else:
                try:
                    route.remove(t)
                except ValueError:
                    pass

    assigned = set()
    for route in routes:
        assigned.update(route)

    unassigned = list(set(range(T)) - assigned)
    rng.shuffle(unassigned)

    for t in unassigned:
        candidates = [i for i in range(W) if len(routes[i]) < L_max]
        if not candidates:
            candidates = list(range(W))
        wi = rng.choice(candidates)
        route = routes[wi]
        pos = rng.randrange(len(route) + 1)
        route.insert(pos, t)

    return routes

def repair_layer_uniform_crossover_repairs(
    parent_a: Individual,
    parent_b: Individual,
    rng: random.Random,
) -> List[List[bool]]:
    W = parent_a.num_workers
    L_max = parent_a.L_max

    child_repairs: List[List[bool]] = [[False] * L_max for _ in range(W)]
    for i in range(W):
        for l in range(L_max):
            child_repairs[i][l] = parent_a.repairs[i][l] if rng.random() < 0.5 else parent_b.repairs[i][l]
    return child_repairs

