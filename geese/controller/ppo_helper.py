from collections import deque
import copy
from typing import Any, Deque, List

from geese.structure.parameter.ppo_parameter import PPOParameter
import numpy as np


def calc_n_step_return(reward_q: List[Deque], gamma: np.ndarray) -> List[float]:
    return [np.sum(np.array(r_q)*gamma) for r_q in reward_q]


def update_PPO_list(
    ppo_parameter: PPOParameter,
    obs: List[np.ndarray],
    action: List[np.ndarray],
    n_step_return: List[np.ndarray],
    value: List[np.ndarray],
    value_n: List[np.ndarray],
    prob: List[np.ndarray],
    player_done_list: List[bool]
) -> None:
    ppo_parameter.obs_list.extend(
        [o for o, d in zip(obs, player_done_list) if not d])
    ppo_parameter.action_list.extend(
        [a for a, d in zip(action, player_done_list) if not d])
    ppo_parameter.n_step_return_list.extend(
        [n for n, d in zip(n_step_return, player_done_list) if not d])
    ppo_parameter.v_list.extend(
        [v for v, d in zip(value, player_done_list) if not d])
    ppo_parameter.v_n_list.extend(
        [v_n for v_n, d in zip(value_n, player_done_list) if not d])
    ppo_parameter.pi_list.extend(
        [p for p, d in zip(prob, player_done_list) if not d])


def create_que_list(
    index_1: int,
    index_2: int
) -> List[List[Deque]]:
    return [[deque()
             for _ in range(index_2)]
            for __ in range(index_1)]


def reset_que(index: int) -> List[Deque]:
    return [deque()
            for _ in range(index)]


def reset_train_data(ppo_parameter: PPOParameter) -> None:
    ppo_parameter.obs_list = []
    ppo_parameter.action_list = []
    ppo_parameter.n_step_return_list = []
    ppo_parameter.v_list = []
    ppo_parameter.v_n_list = []
    ppo_parameter.pi_list = []


def add_to_que(traget_que_list: List[List[Deque]], add_data_list: List[List[Any]]) -> None:
    [[t_q.append(a_d) for t_q, a_d in zip(target_q, add_data)] for target_q,
     add_data in zip(traget_que_list, add_data_list)]


def create_padding_data(
    ppo_parameter: PPOParameter,
    obs_q: Deque,
    action_q: Deque,
    reward_q: Deque,
    value_q: Deque,
    prob_q: Deque
) -> None:

    if len(obs_q) != len(action_q) or len(obs_q) != len(value_q) or len(obs_q) != len(prob_q):

        print('aa')

    if len(reward_q) != ppo_parameter.num_step:
        target_reward_q = copy.deepcopy(reward_q)
        while len(target_reward_q) != ppo_parameter.num_step:
            target_reward_q.append(0)
    else:
        target_reward_q = reward_q

    for _ in range(len(obs_q)):
        obs = obs_q.popleft()
        action = action_q.popleft()
        n_step_return = calc_n_step_return(
            [target_reward_q], ppo_parameter.gamma)[0]
        target_reward_q.popleft()
        target_reward_q.append(0)
        value = value_q.popleft()
        prob = prob_q.popleft()
        update_PPO_list(
            ppo_parameter,
            [obs],
            [action],
            [n_step_return],
            [value],
            [0],
            [prob],
            [False]
        )
