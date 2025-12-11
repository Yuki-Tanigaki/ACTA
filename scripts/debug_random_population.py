# scripts/debug_random_population.py
from __future__ import annotations

from acta.ga.initialization import random_population

def main() -> None:
    pop = random_population(
        population_size=5,
        num_workers=3,
        num_tasks=10,
        L_max=5,
        seed=42,
        repair_prob=0.2,
    )

    for idx, ind in enumerate(pop):
        print(f"=== Individual {idx} ===")
        print("routes:", ind.routes)
        print("repairs:", ind.repairs)
        print("task_ids:", ind.task_ids)
        print("count_tasks_per_worker:", ind.count_tasks_per_worker())
        print()

if __name__ == "__main__":
    main()
