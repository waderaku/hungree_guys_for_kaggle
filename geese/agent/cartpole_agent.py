from geese.structure.parameter.agent_parameter import AgentParameter
from typing import List, Tuple

import numpy as np
from geese.agent import Agent
from geese.structure import Observation
from kaggle_environments.envs.hungry_geese.hungry_geese import Action
from kaggle_environments.envs.hungry_geese.hungry_geese import (
    Observation as KaggleObservation,
)


N_ACTION = 2


class CartPoleAgent(Agent):
    def __init__(self, parameter: AgentParameter):
        self._model = parameter.model

    def get_action(self, obs: Observation):
        raise NotImplementedError

    def step(self, obs: List[Observation]) -> Tuple[List[int], np.ndarray, np.ndarray]:
        pi, v = self._model(np.array(obs))
        pi, v = pi.numpy(), v.numpy()
        action = [np.random.choice(list(range(N_ACTION)), p=p) for p in pi]
        return action, pi, v

    def save(self, path: str):
        raise NotImplementedError

    def load(self, path: str):
        raise NotImplementedError

    @property
    def model(self):
        return self._model

    def __call__(self, obs: KaggleObservation) -> Action:
        raise NotImplementedError("This agent is not for kaggle environments")
