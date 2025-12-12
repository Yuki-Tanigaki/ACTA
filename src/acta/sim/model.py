from __future__ import annotations
from importlib import import_module
from typing import Dict, Tuple

from mesa import Model
from mesa.space import ContinuousSpace

from acta.sim.failure_models import FailureModel
from acta.sim.agent import WorkerAgent, TaskAgent, CommanderAgent
from acta.sim.scenario_loader import ScenarioConfig
from acta.sim.task_selection import TaskSelector

class ACTAScenarioModel(Model):
    def __init__(self, cfg: ScenarioConfig):
        super().__init__(seed=cfg.seed)

        self.cfg = cfg
        self.time_step = cfg.time_step

        self.space = ContinuousSpace(
            x_max=cfg.space_width,
            y_max=cfg.space_height,
            torus=False,    # 非ループ空間
        )

        self.communication_range: float = cfg.communication_range

        # 司令拠点
        self.command_center = CommanderAgent(model=self)
        x, y = cfg.command_center_pos  # unpack
        self.space.place_agent(self.command_center, (x, y))

        # 修理拠点
        self.repair_depot_pos=cfg.repair_depot_pos
        self.repair_duration=cfg.repair_duration

        # タスク
        self.tasks: Dict[int, TaskAgent] = {}
        for t_spec in cfg.tasks:
            agent = TaskAgent(
                model=self,
                task_id=t_spec.task_id,
                total_work=t_spec.total_work,
                remaining_work=t_spec.remaining_work,
            )
            self.tasks[t_spec.task_id] = agent
            x, y = t_spec.position  # unpack
            self.space.place_agent(agent, (x, y))

        # ワーカー
        self.workers: Dict[int, WorkerAgent] = {}
        for w_spec in cfg.workers:
            agent = WorkerAgent(
                model=self,
                worker_id=w_spec.worker_id,
                speed=w_spec.speed,
                throughput=w_spec.throughput,
                speed_eta=w_spec.speed_eta,
                throughput_eta=w_spec.throughput_eta,
                initial_H=w_spec.initial_H,
                fatigue_move=w_spec.fatigue_move,
                fatigue_work=w_spec.fatigue_work,
            )
            self.workers[w_spec.worker_id] = agent
            x, y = w_spec.position  # unpack
            self.space.place_agent(agent, (x, y))

        # 故障モデル
        f_module = import_module(cfg.failure_model.module)
        f_cls = getattr(f_module, cfg.failure_model.class_name)
        self.failure_model: FailureModel = f_cls(**cfg.failure_model.params)
        # タスク選択ポリシー
        ts_module = import_module(cfg.task_selector.module)
        ts_class = getattr(ts_module, cfg.task_selector.class_name)
        self.task_selector: TaskSelector = ts_class(**cfg.task_selector.params)
        
        self.command_center.initialize_full_info(
            workers=self.workers.values(),
            tasks=self.tasks.values(),
        )
    
    # ==========================================================
    # 通信距離判定：エージェント/座標どちらでも使えるユーティリティ
    # ==========================================================
    def _get_pos(self, obj) -> Tuple[float, float]:
        """Agent または座標タプルを pos に変換するヘルパー."""
        if hasattr(obj, "pos"):
            return tuple(obj.pos)  # type: ignore[arg-type]
        return tuple(obj)          # すでに (x, y) の場合を想定

    def distance(self, a, b) -> float:
        """a と b の距離を返す（ContinuousSpace の設定に従う）."""
        pa = self._get_pos(a)
        pb = self._get_pos(b)
        # ContinuousSpace の get_distance を使えば torus 設定も反映される
        return self.space.get_distance(pa, pb)

    def can_communicate(self, a, b) -> bool:
        """a と b が通信可能距離内かどうか."""
        return self.distance(a, b) <= self.communication_range

    def step(self):
        workers = list(self.workers.values())
        # 全 worker について _next_info_state を計算
        for w in workers:
            w.prepare_communicate()
        
        self.command_center.communicate()

        # 全 worker の info_state を一斉に更新
        for w in workers:
            w.communicate()
        
        tasks = list(self.tasks.values())
        for t in tasks:
            t.begin_step()

        # 各ワーカーのタスク選択
        self.task_selector.assign_tasks(self)

        # 全エージェントのステップ実行
        self.agents.do("step")

        for t in tasks:
            t.end_step()

    def all_tasks_done(self) -> bool:
        return all(t.status == "done" for t in self.tasks.values())

    def get_makespan(self) -> float:
        # TaskAgent.finished_step の最大値 × time_step
        finished_steps = [
            t.finished_step for t in self.tasks.values() if t.finished_step is not None
        ]
        if not finished_steps:
            return 0.0
        return max(finished_steps) * self.time_step