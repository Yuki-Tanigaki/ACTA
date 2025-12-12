from __future__ import annotations

from copy import deepcopy
import random
from typing import List, Callable, Optional

from acta.ga.crossover import crossover
from acta.ga.mutation import mutate
from acta.ga.representation import Individual
from acta.ga.initialization import random_population

EvaluateFunc = Callable[[Individual], List[float]]


class SimpleGA:
    def __init__(
        self,
        num_workers: int,
        num_tasks: int,
        L_max: int,
        pop_size: int,
        generations: int,
        elitism_rate: float,
        evaluate: EvaluateFunc,
        tournament_size: int = 2,
        mutation_rate: float = 0.1,
        seed: Optional[int] = None,
    ):
        self.num_workers = num_workers
        self.num_tasks = num_tasks
        self.L_max = L_max

        self.pop_size = pop_size
        self.generations = generations
        self.elitism_rate = elitism_rate
        self.evaluate = evaluate

        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate


        self.rng = random.Random(seed)

        self.population: List[Individual] = []
        self.best: Optional[Individual] = None

    # -------------------------
    # 初期化
    # -------------------------
    def initialize(self):
        repair_prob = 1.0/self.L_max  # 修理確率
        self.population = random_population(
            population_size=self.pop_size,
            num_workers=self.num_workers,
            num_tasks=self.num_tasks,
            L_max=self.L_max,
            rng=self.rng,
            repair_prob=repair_prob,
        )
        for ind in self.population:
            ind.objectives = self.evaluate(ind)

    # -------------------------
    # 親選択
    # -------------------------
    def tournament_select(self) -> Individual:
        comps = self.rng.sample(self.population, self.tournament_size)
        return min(comps, key=lambda ind: ind.objectives[0] + ind.objectives[1])

    # -------------------------
    # GA 実行
    # -------------------------
    def run(self) -> Individual:
        self.initialize()

        elite_k = max(1, int(self.pop_size * self.elitism_rate))

        for gen in range(self.generations):
            # --- 現世代からエリートを抜き出して保存 ---
            elites = sorted(self.population, key=lambda ind: ind.objectives[0] + ind.objectives[1])[:elite_k]
            elites = [deepcopy(e) for e in elites]

            # --- 次世代個体群の生成 ---
            need = self.pop_size - elite_k
            offspring: List[Individual] = []
            for i in range(need):
                p1 = self.tournament_select()
                p2 = self.tournament_select()

                child = crossover(p1, p2, self.rng)
                mutate(child, self.rng, self.mutation_rate)
                child.objectives = self.evaluate(child)
                offspring.append(child)
                # print(child.routes, child.repairs, child.objectives)

            self.population = elites + offspring
            best = min(self.population, key=lambda ind: ind.objectives[0] + ind.objectives[1])
            # print(gen, best.routes, best.repairs, best.objectives)

        self.best = min(self.population, key=lambda ind: ind.objectives[0] + ind.objectives[1])
        return self.best
