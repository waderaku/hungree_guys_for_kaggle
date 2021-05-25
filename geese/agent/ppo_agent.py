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

    def step(self, obs: Observation) -> Tuple[Action, np.ndarray, np.ndarray]:
        prob, value = self._model(obs)
        prob = prob.numpy()
        next_action = np.random.choice(ACTIONLIST, p=prob)
        return next_action, value.numpy(), prob

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
