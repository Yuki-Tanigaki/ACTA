from abc import ABC, abstractmethod

class Evaluator(ABC):
    @abstractmethod
    def __call__(self, indiv) -> list[float]:
        ...