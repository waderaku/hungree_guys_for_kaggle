from abc import ABC, abstractmethod

from geese.structure import Observation


class Agent(ABC):
    @abstractmethod
    def get_action(self, obs: Observation):
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str):
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str):
        raise NotImplementedError

    @property
    @abstractmethod
    def model(self):
        raise NotImplementedError
