from geese.structure import Observation
from geese.controller.ppo_controller import PPOController
from typing import Deque, List
from geese.constants import ACTIONLIST, NUM_GEESE
import numpy as np


def create_train_data():
    pass


def calc_n_step_return(reward_q: Deque, gunnma: np.ndarray) -> np.ndarray:
    return np.sum(np.array(reward_q)*gunnma)


def calc_reward(
    done: bool,
    reward: np.ndarray,
    param_reward: List[float]
) -> List[float]:
    default_reward = [0]*NUM_GEESE
    if done:
        rank_list = np.argsort(reward)
        for rank, r in zip(rank_list, param_reward):
            default_reward[rank] = r

    return default_reward


def _update_PPO_list(
    ppo_controller: PPOController,
    obs: List[Observation],
    action: List[ACTIONLIST],
    value: np.ndarray,
    prob: np.ndarray
) -> None:
    for i in range()
    ppo_controller.obs_list.append(obs)
    ppo_controller.action_list.append(action)
    ppo_controller.v_list.append(value)
    ppo_controller.pi_list.append(prob)
