from __future__ import annotations

import random
from typing import List, Callable, Optional

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
        self.population = random_population(
            population_size=self.pop_size,
            num_workers=self.num_workers,
            num_tasks=self.num_tasks,
            L_max=self.L_max,
        )
        for ind in self.population:
            ind.objectives = self.evaluate(ind)

    # -------------------------
    # 親選択
    # -------------------------
    def tournament_select(self) -> List[Individual]:
        selected: List[Individual] = []
        for _ in range(self.pop_size):
            comps = self.rng.sample(self.population, self.tournament_size)
            best = min(comps, key=lambda ind: ind.objectives[0])
            selected.append(best)
        return selected

    # -------------------------
    # 交叉（stub）
    # -------------------------
    def crossover(self, p1: Individual, p2: Individual) -> Individual:
        return p1.copy()  # 今はstub

    # -------------------------
    # 突然変異（stub）
    # -------------------------
    def mutate(self, ind: Individual):
        pass

    # -------------------------
    # GA 実行
    # -------------------------
    def run(self) -> Individual:
        self.initialize()

        for gen in range(self.generations):
            parents = self.tournament_select()

            # 交叉と突然変異
            offspring: List[Individual] = []
            for i in range(0, self.pop_size, 2):
                p1 = parents[i]
                p2 = parents[(i + 1) % self.pop_size]

                child = self.crossover(p1, p2)
                self.mutate(child)
                child.objectives = self.evaluate(child)
                offspring.append(child)

            self.population = offspring

        self.best = min(self.population, key=lambda ind: ind.objectives[0])
        return self.best
