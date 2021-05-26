import tensorflow as tf
import numpy as np
from typing import Tuple

from geese.structure import Observation
from geese.agent import Agent
from geese.constants import ACTIONLIST
from kaggle_environments.envs.hungry_geese.hungry_geese import Action


class PPOAgent(Agent):
    def __init__(
        self,
        model: tf.keras.models.Model
    ):
        self._model = model

    # return Tuple([4], [4], [4*4])
    def step(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        prob_list, value_list = self._model(obs)
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
