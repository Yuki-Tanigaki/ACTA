from __future__ import annotations
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


@dataclass
class WorkerSpec:
    worker_id: int
    position: Tuple[float, float]
    speed: float
    throughput: float
    speed_eta: float
    throughput_eta: float
    initial_H: float
    fatigue_move: float
    fatigue_work: float


@dataclass
class TaskSpec:
    task_id: int
    position: Tuple[float, float]
    total_work: float
    remaining_work: float


@dataclass
class FailureModelConfig:
    module: str
    class_name: str
    params: Dict[str, Any]


@dataclass
class TaskSelectorConfig:
    module: str
    class_name: str
    params: Dict[str, Any]


@dataclass
class ScenarioConfig:
    scenario_name: str
    seed: int
    space_width: float
    space_height: float

    max_steps: int
    time_step: float

    command_center_pos: Tuple[float, float]
    repair_depot_pos: Tuple[float, float]
    repair_duration: int

    communication_range: float
    failure_model: FailureModelConfig
    task_selector: TaskSelectorConfig

    workers: List[WorkerSpec]
    tasks: List[TaskSpec]


def _load_workers_from_csv(csv_path: Path) -> list[WorkerSpec]:
    workers: list[WorkerSpec] = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            workers.append(
                WorkerSpec(
                    worker_id=int(row["id"]),
                    position=(float(row["x"]), float(row["y"])),
                    speed=float(row["speed"]),
                    throughput=float(row["throughput"]),
                    speed_eta=float(row["speed_eta"]),
                    throughput_eta=float(row["throughput_eta"]),
                    initial_H=float(row["initial_H"]),
                    fatigue_move=float(row["fatigue_move"]),
                    fatigue_work=float(row["fatigue_work"]),
                )
            )
    return workers


def _load_tasks_from_csv(csv_path: Path) -> list[TaskSpec]:
    tasks: list[TaskSpec] = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_work = float(row["total_work"])
            remaining_work = float(row.get("remaining_work", total_work))

            tasks.append(
                TaskSpec(
                    task_id=int(row["id"]),
                    position=(float(row["x"]), float(row["y"])),
                    total_work=total_work,
                    remaining_work=remaining_work,
                )
            )
    return tasks

def load_scenario_config(yaml_path: str | Path) -> ScenarioConfig:
    yaml_path = Path(yaml_path)
    with yaml_path.open("r") as f:
        cfg = yaml.safe_load(f)

    base_dir = yaml_path.parent

    # --- top-level ---
    scenario_name: str = cfg["scenario_name"]
    seed: int = cfg["seed"]

    # --- space ---
    space_cfg = cfg["space"]
    space_width = float(space_cfg["width"])
    space_height = float(space_cfg["height"])
    communication_range = float(space_cfg["range"])

    # --- sim ---
    sim_cfg = cfg["sim"]
    max_steps = int(sim_cfg["max_steps"])
    time_step = float(sim_cfg["time_step"])

    # --- command_center / repair_depot ---
    cmd_pos_raw = cfg["command_center"]["position"]
    command_center_pos: Tuple[float, float] = (float(cmd_pos_raw[0]), float(cmd_pos_raw[1]))

    depot_cfg = cfg["repair_depot"]
    depot_pos_raw = depot_cfg["position"]
    repair_depot_pos: Tuple[float, float] = (float(depot_pos_raw[0]), float(depot_pos_raw[1]))
    repair_duration = int(depot_cfg["repair_duration"])

    # --- failure_model ---
    fm_cfg = cfg["failure_model"]
    failure_model = FailureModelConfig(
        module=fm_cfg["module"],
        class_name=fm_cfg["class"],
        params=fm_cfg.get("params", {}),
    )

    # --- task_selector ---
    ts_cfg = cfg["task_selection"]
    task_selector = TaskSelectorConfig(
        module=ts_cfg["module"],
        class_name=ts_cfg["class"],
        params=ts_cfg.get("params", {}),
    )

    # --- workers / tasks from CSV ---
    workers_csv = base_dir / cfg["workers_csv"]
    tasks_csv = base_dir / cfg["tasks_csv"]

    workers = _load_workers_from_csv(workers_csv)
    tasks = _load_tasks_from_csv(tasks_csv)

    return ScenarioConfig(
        scenario_name=scenario_name,
        seed=seed,
        space_width=space_width,
        space_height=space_height,
        max_steps=max_steps,
        time_step=time_step,
        command_center_pos=command_center_pos,
        repair_depot_pos=repair_depot_pos,
        repair_duration=repair_duration,
        communication_range=communication_range,
        failure_model=failure_model,
        task_selector=task_selector,
        workers=workers,
        tasks=tasks,
    )