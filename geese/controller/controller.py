from abc import ABC, abstractmethod


class Controller(ABC):
    @abstractmethod
    def train(self) -> None:
        raise NotImplementedError
