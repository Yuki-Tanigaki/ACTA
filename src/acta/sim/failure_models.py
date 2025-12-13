from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol


class FailureModel(Protocol):
    """WorkerAgentが使う故障モデルのインターフェース."""

    def failure_prob(self, H: float) -> float:
        """
        累積疲労度 に対する故障確率を返す。

        Parameters
        ----------
        H : float
            累積疲労度
        """
        ...

    def failure_prob_step(self, H: float, delta_H: float) -> float:
        """
        このステップで増加した疲労度 に対する故障確率を返す。

        Parameters
        ----------
        delta_H : float
            このステップで増加した疲労度
        """
        ...


@dataclass
class SimpleFailureModel:
    """常に同じ確率で故障するシンプルなモデル."""

    prob: float

    def failure_prob(self, H: float) -> float:
        return 1.0 - math.exp(H * math.log(1.0 - self.prob))
    
    def failure_prob_step(self, H: float, delta_H: float) -> float:
        return self.prob
    
@dataclass
class WeibullFailureModel:
    lambd: float  # スケール
    k: float      # 形状 (k>1 で wear-out)

    def failure_prob(self, H: float) -> float:
        """Weibull の累積故障確率 F(H)=1-exp(-(λH)^k)."""
        if H <= 0 or self.lambd <= 0 or self.k <= 0:
            return 0.0
        return 1.0 - math.exp(- (self.lambd * H) ** self.k)
    
    def failure_prob_step(self, H: float, delta_H: float) -> float:
        if delta_H <= 0 or self.lambd <= 0 or self.k <= 0:
            return 0.0

        def F(x: float) -> float:
            return 1.0 - math.exp(- (self.lambd * x) ** self.k)

        F_old = F(H)
        F_new = F(H + delta_H)

        # 条件付き確率: これまで生き残っている前提で、このステップで壊れる確率
        if F_old >= 1.0:
            return 1.0
        return (F_new - F_old) / (1.0 - F_old)
