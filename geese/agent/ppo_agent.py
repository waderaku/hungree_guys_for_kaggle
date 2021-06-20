from typing import List, Tuple

import numpy as np
import tensorflow as tf
from geese.agent import Agent
from geese.constants import ACTIONLIST
from geese.structure import Observation
from geese.structure.parameter.agent_parameter import AgentParameter
from geese.util.converter import action2int, to_np_obs
from kaggle_environments.envs.hungry_geese.hungry_geese import Action
from kaggle_environments.envs.hungry_geese.hungry_geese import (
    Observation as KaggleObservation,
)


class PPOAgent(Agent):
    def __init__(self, parameter: AgentParameter):
        self._model = parameter.model
        # KaggleAgentとして利用するためのKaggle Observation
        self._last_obs = None

        self._last_action = None

    # return Tuple([4], [4], [4*4])
    def step(
        self, obs: List[Observation], masked_flg=False
    ) -> Tuple[List[Action], np.ndarray, np.ndarray]:
        prob_list, value_list = self._model(np.array(obs))
        prob_list = prob_list.numpy()
        value_list = value_list.numpy()
        if masked_flg and self._last_action is not None:
            last_action_index = [action2int(action) for action in self._last_action]
            last_action_one_hot = np.identity(len(ACTIONLIST))[last_action_index]
            # 前回の行動を0、それ以外を1にする
            last_action_one_hot = last_action_one_hot * -1 + 1
            # masking
            masked_prob_list = prob_list * last_action_one_hot
            sum_prob_list = np.sum(masked_prob_list, axis=1)
            next_action_list = [
                np.random.choice(ACTIONLIST, p=prob / sum_prob)
                for prob, sum_prob in zip(masked_prob_list, sum_prob_list)
            ]
        else:
            next_action_list = [
                np.random.choice(ACTIONLIST, p=prob) for prob in prob_list
            ]
        self._last_action = next_action_list
        return next_action_list, value_list, prob_list

    def masked_step(
        self, obs: List[Observation]
    ) -> Tuple[List[Action], np.ndarray, np.ndarray]:
        prob_list, value_list = self._model(np.array(obs))
        prob_list = prob_list.numpy()
        value_list = value_list.numpy()
        sum_prob_list = np.sum(prob_list, axis=1)
        next_action_list = [
            np.random.choice(ACTIONLIST, p=prob / sum_prob)
            for prob, sum_prob in zip(prob_list, sum_prob_list)
        ]
        return next_action_list, value_list, prob_list

    def get_action(self, obs: Observation) -> Action:
        next_action, _, _ = self.step([obs])
        return next_action[0]

    def save(self, path: str) -> None:
        if self._model.built:
            self._model.save(path)
        else:
            print("Model is not yet built. Skipping saving proceduce.")

    def load(self, path: str) -> None:
        self._model = tf.keras.models.load_model(path)

    @property
    def model(self) -> tf.keras.models.Model:
        return self._model

    def __call__(self, obs: KaggleObservation) -> str:
        np_obs = to_np_obs(obs, self._last_obs)
        action = self.get_action(np_obs)
        return action.name
