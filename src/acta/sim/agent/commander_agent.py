from __future__ import annotations
from typing import TYPE_CHECKING, Iterable, cast
from mesa import Agent

if TYPE_CHECKING:
    from acta.sim.agent.worker_agent import WorkerAgent
    from acta.sim.agent.task_agent import TaskAgent
    from acta.sim.model import ACTAScenarioModel
from acta.sim.info_state import InfoState, TaskInfo, WorkerInfo

class CommanderAgent(Agent):
    model: "ACTAScenarioModel"

    def __init__(self, model: ACTAScenarioModel):
        super().__init__(model)
        self.info_state = InfoState(workers={}, tasks={})

    def initialize_full_info(
            self,
            workers: Iterable["WorkerAgent"],
            tasks: Iterable["TaskAgent"],
        ) -> None:
            """t=0 時点での全ワーカー・全タスクの真の状態をコピーする。"""
            timestamp = 0

            # --- workers ---
            for w in workers:
                self.info_state.workers[w.worker_id] = WorkerInfo(
                    worker_id=w.worker_id,
                    position=tuple(w.pos),
                    state=w.state,
                    H=w.H,
                    timestamp=timestamp,
                )

            # --- tasks ---
            for t in tasks:
                self.info_state.tasks[t.task_id] = TaskInfo(
                    task_id=t.task_id,
                    position=tuple(t.pos),
                    total_work=t.total_work,
                    remaining_work=t.remaining_work,
                    status=t.status,
                    timestamp=timestamp,
                )
    
    def communicate(self):
        """ 近隣ワーカーと情報同期を行う。 """
        for agent in self.model.workers.values():
            # 通信可能なワーカーとのみ情報同期
            if self.model.can_communicate(self, agent):
                # 司令塔視点・ワーカー視点の情報を相互更新
                self.info_state.sync_with(agent.info_state)
