from __future__ import annotations
import random

from acta.ga.representation import Individual


def mutate(
    child: Individual,
    rng: random.Random,
    mutation_rate: float = 0.1,
    # --- routes側（どれを選ぶかの比率） ---
    p_route_swap: float = 0.40,
    p_route_insert: float = 0.35,
    p_route_exchange: float = 0.25,
    # --- repairs側（bit flipの期待回数） ---
    repair_flip_rate: float = 1.0,
) -> None:
    """
    child(Individual) を破壊的に mutate する。

    mutation_rate:
      - この呼び出し1回あたり mutate する確率

    repair_flip_rate:
      - 1個体あたりの repairs の期待反転回数（例: 1.0 なら平均1bit flip）
    """
    if rng.random() >= mutation_rate:
        return

    # 1) routes mutate（安全操作のみ：重複/欠損が起きない）
    _mutate_routes(child, rng, p_route_swap, p_route_insert, p_route_exchange)

    # 2) repairs mutate（bit列に対する軽い摂動）
    _mutate_repairs(child, rng, repair_flip_rate)


# ============================================================
# routes mutation (feasibility-preserving)
# ============================================================

def _mutate_routes(
    ind: Individual,
    rng: random.Random,
    p_swap: float,
    p_insert: float,
    p_exchange: float,
) -> None:
    """
    routes について「常に全タスクがちょうど1回」を壊さない変異のみ行う。

    - intra swap   : 同一ワーカー内で2点 swap
    - intra insert : 同一ワーカー内で1タスクを抜いて別位置へ挿入
    - inter exchange: 異なる2ワーカー間で各1タスクを交換（両方空でないとき）
    """
    # 確率の正規化（合計が1じゃなくてもOKにする）
    s = p_swap + p_insert + p_exchange
    if s <= 0:
        return
    r = rng.random() * s

    if r < p_swap:
        _route_intra_swap(ind, rng)
    elif r < p_swap + p_insert:
        _route_intra_insert(ind, rng)
    else:
        _route_inter_exchange(ind, rng)


def _route_intra_swap(ind: Individual, rng: random.Random) -> None:
    # 長さ2以上のルートを持つワーカーを選ぶ
    candidates = [i for i, rt in enumerate(ind.routes) if len(rt) >= 2]
    if not candidates:
        return
    wi = rng.choice(candidates)
    rt = ind.routes[wi]
    a, b = rng.sample(range(len(rt)), 2)
    rt[a], rt[b] = rt[b], rt[a]


def _route_intra_insert(ind: Individual, rng: random.Random) -> None:
    # 長さ2以上のルートを持つワーカーを選ぶ（insertは同一要素移動なので len>=2 推奨）
    candidates = [i for i, rt in enumerate(ind.routes) if len(rt) >= 2]
    if not candidates:
        return
    wi = rng.choice(candidates)
    rt = ind.routes[wi]
    frm = rng.randrange(len(rt))
    task = rt.pop(frm)
    to = rng.randrange(len(rt) + 1)
    rt.insert(to, task)


def _route_inter_exchange(ind: Individual, rng: random.Random) -> None:
    # 空でないワーカーを2つ選んで、各1要素を交換（重複/欠損は起きない）
    non_empty = [i for i, rt in enumerate(ind.routes) if len(rt) >= 1]
    if len(non_empty) < 2:
        return
    w1, w2 = rng.sample(non_empty, 2)

    r1 = ind.routes[w1]
    r2 = ind.routes[w2]

    p1 = rng.randrange(len(r1))
    p2 = rng.randrange(len(r2))

    r1[p1], r2[p2] = r2[p2], r1[p1]


# ============================================================
# repairs mutation (no normalization)
# ============================================================

def _mutate_repairs(
    ind: Individual,
    rng: random.Random,
    flip_rate: float,
) -> None:
    """
    repairs のビット列を mutate。

    flip_rate を「期待反転回数」として扱い、
      n_flips = Poisson(flip_rate) 風（簡易）にしても良いが、
    ここでは実装を軽くするため:
      - 小数込みの回数指定 → floor回 + 余り確率で+1回
    """
    W = ind.num_workers
    L = ind.L_max
    if W <= 0 or L <= 0:
        return

    # flip回数（期待値 flip_rate）
    if flip_rate <= 0:
        return
    n = int(flip_rate)
    if rng.random() < (flip_rate - n):
        n += 1
    if n <= 0:
        n = 1  # 変異として呼ぶ以上、最低1回は反転しておく派

    for _ in range(n):
        wi = rng.randrange(W)
        li = rng.randrange(L)
        ind.repairs[wi][li] = not ind.repairs[wi][li]