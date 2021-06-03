from abc import ABC, abstractmethod

from geese.structure import Observation
from kaggle_environments.envs.hungry_geese.hungry_geese import (
    Observation as KaggleObservation,
)
from kaggle_environments.envs.hungry_geese.hungry_geese import Action


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

    @abstractmethod
    def __call__(self, obs: KaggleObservation) -> Action:
        raise NotImplementedError
