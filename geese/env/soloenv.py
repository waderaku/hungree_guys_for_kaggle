from typing import Tuple

from geese.constants import CONFIGURATION
from geese.structure.observation import Observation
from geese.structure.reward import Reward
from kaggle_environments import make
from kaggle_environments.envs.hungry_geese.hungry_geese import Action, GreedyAgent
from kaggle_environments.envs.hungry_geese.hungry_geese import Observation as KaggleObs


class SoloEnv:
    def __init__(self):
        self._env = make("hungry_geese")
        self._geedy_agent = GreedyAgent(configuration=CONFIGURATION)

    def reset(self) -> Observation:
        return self._env.reset()[0]

    def step(self, action: Action) -> Tuple[Observation, Reward, bool]:
        raise NotImplementedError

    def __str__(self) -> str:
        raise NotImplementedError
