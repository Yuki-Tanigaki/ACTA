from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol


class FailureModel(Protocol):
    """WorkerAgentが使う故障モデルのインターフェース."""

    eta: float

    def failure_prob(self, H: float) -> float:
        """
        疲労度H に対する故障確率を返す。
        """
        ...


@dataclass
class SimpleFailureModel:
    """常に同じ確率で故障するシンプルなモデル."""

    prob: float

    def failure_prob(self, H: float) -> float:
        return self.prob