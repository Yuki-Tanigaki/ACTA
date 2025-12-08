# src/acta/ga/operators.py
from __future__ import annotations

from typing import List, Tuple, Optional
import random

from acta.ga.representation import Individual
from acta.utils.logging_utils import get_logger

logger = get_logger(__name__)

# ============================================================
# ランダム生成
# ============================================================

def create_random_individual(
    num_workers: int,
    num_tasks: int,
    L_max: int,
    rng: random.Random,
) -> Individual:
    """
    ランダムなタスク割当＋修理フラグ False 初期化の個体を生成する。

    - 各タスク j = 0..num_tasks-1 はちょうど 1 回どこかのワーカーに割り当てられる
    - 各ワーカー i のタスク数は高々 L_max 個
    - 修理フラグ repairs[i] は長さ L_max の False で初期化
    """
    if num_workers * L_max < num_tasks:
        msg = f"num_workers * L_max = {num_workers * L_max} < num_tasks={num_tasks}"
        logger.error(msg)
        raise ValueError(msg)

    # タスク ID をシャッフルしてから、ワーカーに順に配る
    task_ids = list(range(num_tasks))
    rng.shuffle(task_ids)

    routes: List[List[int]] = [[] for _ in range(num_workers)]

    # 単純に round-robin で割り当て（L_max を超えないように）
    worker_idx = 0
    for j in task_ids:
        # L_max を超えない最初のワーカーを探す
        tries = 0
        while len(routes[worker_idx]) >= L_max:
            worker_idx = (worker_idx + 1) % num_workers
            tries += 1
            if tries > num_workers:
                msg = "Cannot find worker with available capacity."
                logger.error(msg)
                raise RuntimeError(msg)
        routes[worker_idx].append(j)
        worker_idx = (worker_idx + 1) % num_workers

    # --- 修理レイヤ：確率的に1回程度修理を入れる ---
    repairs: List[List[bool]] = []
    p_repair = 1.0/L_max  # 修理確率

    for i in range(num_workers):
        route_len = len(routes[i])
        flags = [False] * L_max

        for idx in range(L_max):
            if rng.random() < p_repair:
                flags[idx] = True

        repairs.append(flags)

    ind = Individual(
        num_workers=num_workers,
        num_tasks=num_tasks,
        L_max=L_max,
        routes=routes,
        repairs=repairs,
    )
    return ind


# ============================================================
# 交叉（crossover）
# ============================================================

def route_similarity(route_a: List[int], route_b: List[int]) -> int:
    """
    ルート間の類似度：タスク集合の重なりの大きさ |set(A) cap set(B)| を返す。
    """
    return len(set(route_a) & set(route_b))


def route_layer_srex_like_crossover(
    parent_a: Individual,
    parent_b: Individual,
    rng: random.Random,
) -> Individual:
    """
    ルート順序レイヤに対する交叉。

    手順（SREX を簡略化した実装）：
      1. 一方を基準親 A, 他方を置換元親 B とみなす（ここでは (A,B)=(parent_a,parent_b)）
      2. 基準親のルート集合からランダムに 1 本以上のルートインデックス集合 S を選ぶ
      3. ルートインデックスごとに，置換元親 B のルート集合の中から
         類似度 sim(R_i^A, R_k^B) が最大となるルートを 1 本選び，
         子個体の対応ルートをそのルートで置き換える
    """
    if parent_a.num_workers != parent_b.num_workers:
        msg = "Parents must have the same num_workers."
        logger.error(msg)
        raise ValueError(msg)
    if parent_a.num_tasks != parent_b.num_tasks:
        msg = "Parents must have the same num_tasks."
        logger.error(msg)
        raise ValueError(msg)
    if parent_a.L_max != parent_b.L_max:
        msg = "Parents must have the same L_max."
        logger.error(msg)
        raise ValueError(msg)

    child = parent_a.copy() # 最初は親Aのコピーとして初期化
    num_workers = parent_a.num_workers

    # --- 基準親から交換対象ルート集合 S を選ぶ ---
    # 各ワーカーを 50% の確率で選び、少なくとも 1 つは選ばれるようにする
    S: List[int] = [i for i in range(num_workers) if rng.random() < 0.5]
    if not S:
        S = [rng.randrange(num_workers)]

    # ---- 親 A / B の routes を短い参照に ----
    routes_a = parent_a.routes
    routes_b = parent_b.routes

    # --- Aを基準としてBからルートを借りる ---
    for i in S:
        base_route = routes_a[i]
        # B 側の中で類似度が最大となる k を探す
        indices = list(range(num_workers))
        rng.shuffle(indices)
        best_k = indices[0]
        best_sim = -1
        for k in indices:
            sim = route_similarity(base_route, routes_b[k])
            if sim > best_sim:
                best_sim = sim
                best_k = k

        # 子のルート i を B のルート best_k で置き換え
        child.routes[i] = list(routes_b[best_k])

    repair_routes_feasibility(child, rng)

    return child


def repair_routes_feasibility(
    ind: Individual,
    rng: random.Random,
) -> None:
    """
    SREX 交叉後などにタスクの重複／欠損が生じた場合，
    - 重複タスクを削除
    - 未割当タスクを挿入
    することで，「0..num_tasks-1 の完全順列」を満たすよう修復する。

    ※ 要件定義では「未割当タスクを detour コスト最小位置に挿入」とあるが，
      ここでは簡易実装として「ランダムなワーカー・位置への挿入」を行う。
      将来的に移動距離やメイクスパンを評価するコスト関数を渡せるよう
      拡張することを想定。
    """
    num_tasks = ind.num_tasks
    num_workers = ind.num_workers
    L_max = ind.L_max

    # --- 1. 出現回数を集計 ---
    appearances: List[List[tuple[int, int]]] = [[] for _ in range(num_tasks)]
    # appearances[j] = [(i, pos), ...] : タスク j が出現する (worker, index)

    for i, route in enumerate(ind.routes):
        for pos, task in enumerate(route):
            if 0 <= task < num_tasks:
                appearances[task].append((i, pos))
            else:
                # 範囲外タスクがあれば削除対象として扱うため，
                # 特別に num_tasks 以上の ID は無視する（あまり起こらない想定）
                pass

    # --- 2. 重複タスク削除 ---
    for j in range(num_tasks):
        if len(appearances[j]) <= 1:
            continue
        # 1 個だけ残し、残りは削除する
        # ここでは「最後に現れたものを残す」とする
        # （他でもよいが一貫したポリシーをとる）
        keep_i, keep_pos = appearances[j][-1]
        to_remove = appearances[j][:-1]

        for (wi, pos) in sorted(to_remove, key=lambda x: (x[0], -x[1])):
            # 後ろから削除するとインデックスずれを避けやすい
            # ただしここでは route 内の位置がユニークなのでそのまま pop でも概ね安全
            # 念のため、値 j を探して消す
            route = ind.routes[wi]
            if 0 <= pos < len(route) and route[pos] == j:
                route.pop(pos)
            else:
                # 位置がずれていそうなら，最初に見つかった j を削除
                try:
                    route.remove(j)
                except ValueError:
                    pass

    # --- 3. 現在割り当てられているタスク集合を取得 ---
    assigned = set()
    for route in ind.routes:
        assigned.update(route)

    # --- 4. 未割当タスク集合を求め，ランダムに挿入 ---
    all_tasks = set(range(num_tasks))
    unassigned = list(all_tasks - assigned)

    rng.shuffle(unassigned)

    for j in unassigned:
        # 長さが L_max 未満のワーカー候補を集める
        candidates = [i for i in range(num_workers) if len(ind.routes[i]) < L_max]
        if not candidates:
            # 仕方ないので L_max を超えても良いことにして挿入（例外にすることも可能）
            candidates = list(range(num_workers))
        wi = rng.choice(candidates)
        route = ind.routes[wi]
        insert_pos = rng.randrange(len(route) + 1)  # 0..len(route) のどこか
        route.insert(insert_pos, j)

    # 最終的に check_task_coverage が True になっているはず
    if hasattr(ind, "check_task_coverage"):
        if not ind.check_task_coverage():
            # ここに来たら何かバグがある
            raise RuntimeError("repair_routes_feasibility failed to produce a valid permutation.")


def repair_layer_uniform_crossover(
    parent_a: Individual,
    parent_b: Individual,
    rng: random.Random,
) -> Tuple[List[List[bool]], List[List[bool]]]:
    """
    修理イベントレイヤ（ビット列）に対する一様交叉。

    各ビット r_{i,ℓ} について，
      - 子1は 50% の確率で親Aのビット，50% の確率で親Bのビットを受け継ぐ
      - 子2も同様に独立に 50% / 50% で選ぶ

    とする。
    """
    num_workers = parent_a.num_workers
    L_max = parent_a.L_max

    repairs1: List[List[bool]] = [[False] * L_max for _ in range(num_workers)]
    repairs2: List[List[bool]] = [[False] * L_max for _ in range(num_workers)]

    for i in range(num_workers):
        for l in range(L_max):
            a_bit = parent_a.repairs[i][l]
            b_bit = parent_b.repairs[i][l]

            repairs1[i][l] = a_bit if rng.random() < 0.5 else b_bit
            repairs2[i][l] = a_bit if rng.random() < 0.5 else b_bit

    return repairs1, repairs2


def crossover_individuals(
    parent_a: Individual,
    parent_b: Individual,
    rng: random.Random,
) -> Tuple[Individual, Individual]:
    """
    個体 (routes, repairs) 全体に対する交叉。

    - ルート順序レイヤ：SREX 風交叉
    - 修理レイヤ      ：一様交叉

    を組み合わせて子個体 2 つを返す。
    """
    # ルート順序レイヤ
    child1, child2 = route_layer_srex_like_crossover(parent_a, parent_b, rng)

    # 修理レイヤ
    rep1, rep2 = repair_layer_uniform_crossover(parent_a, parent_b, rng)
    child1.repairs = rep1
    child2.repairs = rep2

    return child1, child2


# ============================================================
# 突然変異（mutation）
# ============================================================

def mutate_intra_route_swap(
    ind: Individual,
    rng: random.Random,
    prob: float = 0.2,
) -> None:
    """
    ルート内スワップ突然変異（Intra-route swap）

    手順（要件定義）：
      1. 突然変異対象とするワーカー i ∈ I を一様ランダムに選ぶ
      2. そのルート長 L_i ≥ 2 であれば，互いに異なる 2 つのインデックスをランダムに選ぶ
      3. その 2 要素を入れ替える

    ここでは「個体に対して確率 prob で 1 回だけ実行」としている。
    """
    if rng.random() >= prob:
        return

    num_workers = ind.num_workers
    # ルート長が 2 以上のワーカーだけ対象
    candidates = [i for i, r in enumerate(ind.routes) if len(r) >= 2]
    if not candidates:
        return

    i = rng.choice(candidates)
    route = ind.routes[i]
    L_i = len(route)

    idx1, idx2 = rng.sample(range(L_i), k=2)
    route[idx1], route[idx2] = route[idx2], route[idx1]


def mutate_inter_route_task_exchange(
    ind: Individual,
    rng: random.Random,
    prob: float = 0.2,
) -> None:
    """
    ワーカー間タスク交換突然変異（Inter-route task exchange）

    要件定義の手順：
      1. 異なる 2 人のワーカー i1, i2 ∈ I, i1 ≠ i2 をランダムに選ぶ
      2. それぞれのルート長 L_{i1}, L_{i2} がともに 1 以上であれば，
         それぞれのタスク列から 1 つずつタスクをランダムに選ぶ
      3. 選んだ 2 つのタスクの担当ワーカーを入れ替え，
         それぞれのタスク列でその位置に相手のタスクを挿入（実質 swap）

    ここでは「個体に対して確率 prob で 1 回だけ実行」とする。
    """
    if rng.random() >= prob:
        return

    num_workers = ind.num_workers

    # ルート長が 1 以上のワーカーのみ対象
    candidates = [i for i, r in enumerate(ind.routes) if len(r) >= 1]
    if len(candidates) < 2:
        return

    i1, i2 = rng.sample(candidates, k=2)
    route1 = ind.routes[i1]
    route2 = ind.routes[i2]

    pos1 = rng.randrange(len(route1))
    pos2 = rng.randrange(len(route2))

    route1[pos1], route2[pos2] = route2[pos2], route1[pos1]
    # この操作は「タスクの担当ワーカーの交換」に相当し，
    # 全体集合としては依然として 0..num_tasks-1 を一度ずつ含む


def mutate_repair_layer_bit_flip(
    ind: Individual,
    rng: random.Random,
    bit_flip_prob: float = 0.05,
) -> None:
    """
    修理イベントレイヤに対するビット反転突然変異。

    要件定義：
      - 修理フラグ r_{i,ℓ} はビット列
      - 各ワーカー i の実際のタスク数 L_i を考慮し，
        ℓ > L_i の成分は無視（突然変異対象からも除外）

    実装：
      各 i に対して，
        有効な作業ステップ ℓ = 0..L_i-1（← Python インデックス）について，
        確率 bit_flip_prob で True/False を反転させる。
    """
    for i in range(ind.num_workers):
        L_i = len(ind.routes[i])  # 実際のタスク数
        for l in range(L_i):
            if rng.random() < bit_flip_prob:
                ind.repairs[i][l] = not ind.repairs[i][l]


def mutate_individual(
    ind: Individual,
    rng: random.Random,
    intra_route_prob: float = 0.2,
    inter_route_prob: float = 0.2,
    repair_bit_flip_prob: float = 0.05,
) -> None:
    """
    個体全体に対して，ルート順序レイヤ＋修理レイヤの突然変異をまとめて適用する。

    - ルート内スワップ突然変異
    - ワーカー間タスク交換突然変異
    - 修理フラグのビット反転

    各操作の適用確率は引数で指定する。
    """
    mutate_intra_route_swap(ind, rng=rng, prob=intra_route_prob)
    mutate_inter_route_task_exchange(ind, rng=rng, prob=inter_route_prob)
    mutate_repair_layer_bit_flip(ind, rng=rng, bit_flip_prob=repair_bit_flip_prob)
