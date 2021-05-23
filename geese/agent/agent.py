from kaggle_environments.envs.hungry_geese.hungry_geese import Action
from geese.agent.mcts import MCTS

import tensorflow as tf
import numpy as np

from geese.constants import ACTIONLIST


class Agent():
    def __init__(
        self,
        model: tf.keras.models.Model,
        eps: float = 1.0,
        eps_anneal: float = 0.9995,
        eps_min: float = 0.15,
    ):
        self.model = model
        self._eps = eps
        self._eps_anneal = eps_anneal
        self._eps_min = eps_min
        self._mcts = MCTS(self.model)

    def get_action(self, tf_obs: tf.Tensor) -> Action:

        prob = self._mcts.get_prob()

        # ε-greedyで次アクションをチョイス
        prob = prob.numpy()
        return self._eps_greedy(prob)

    def _eps_greedy(self, prob: np.array) -> Action:
        rand = np.random.rand()
        self._update_eps()
        if self.eps > rand:
            next_action = np.random.choice(ACTIONLIST)
        else:
            next_action = np.random.choice(ACTIONLIST, p=prob)
        self._update_eps()
        return next_action

    def _update_eps(self) -> None:
        self._eps = np.max([self._eps*self._eps_anneal, self._eps_min])

    def save(self, path: str) -> None:
        self.model.save(path)

    def load(self, path: str) -> None:
        self.model = tf.keras.models.load_model(path)
