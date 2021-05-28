from typing import List, Tuple

import numpy as np
import tensorflow as tf
from geese.agent import Agent
from geese.constants import ACTIONLIST
from geese.structure import Observation
from geese.structure.parameter.agent_parameter import AgentParameter
from kaggle_environments.envs.hungry_geese.hungry_geese import Action


class PPOAgent(Agent):
    def __init__(
        self,
        parameter: AgentParameter
    ):
        self._model = parameter.model

    # return Tuple([4], [4], [4*4])
    def step(self, obs: np.ndarray) -> Tuple[List[Action], np.ndarray, np.ndarray]:
        prob_list, value_list = self._model(np.array(obs))
        prob_list = prob_list.numpy()
        value_list = value_list.numpy()
        next_action_list = [np.random.choice(ACTIONLIST, p=prob)
                            for prob in prob_list]
        return next_action_list, value_list, prob_list

    def get_action(self, obs: Observation) -> Action:
        next_action, _, _ = self.step(obs)
        return next_action

    def save(self, path: str) -> None:
        self._model.save(path)

    def load(self, path: str) -> None:
        self._model = tf.keras.models.load_model(path)

    @property
    def model(self) -> tf.keras.models.Model:
        return self._model
