from collections import deque
import copy

from kaggle_environments.envs.hungry_geese.hungry_geese import Action
from geese.constants import ACTIONLIST, NUM_GEESE
from geese.structure.train_data import TrainData
from geese.util.converter import action2int
from typing import Any, Deque, List, Tuple

from geese.structure.parameter.ppo_parameter import PPOParameter
import numpy as np


def calc_gae_list(delta_q: List[Deque], gae_param: np.ndarray) -> List[float]:
    return [calc_gae(d_q, gae_param) for d_q in delta_q]


def calc_gae(d_q: Deque, gae_param: np.ndarray):
    return np.sum(np.array(d_q) * gae_param)


def add_delta_list(
    delta_que: List[Deque],
    reward_list: List[float],
    v_old_list: List[float],
    v_new_list: List[float],
    gamma: float,
) -> None:
    [
        add_delta(d_q, reward, v_new, v_old, gamma)
        for reward, v_old, v_new, d_q in zip(
            reward_list, v_old_list, v_new_list, delta_que
        )
    ]


def add_delta(
    d_q: Deque, reward: float, v_new: float, v_old: float, gamma: float
) -> None:
    d_q.append(reward + v_new * gamma - v_old)


def update_PPO_list(
    train_data: TrainData,
    obs: List[np.ndarray],
    action: List[np.ndarray],
    gae_list: List[np.ndarray],
    value: List[np.ndarray],
    prob: List[np.ndarray],
    player_done_list: List[bool],
) -> None:
    train_data.obs_list.extend([o for o, d in zip(obs, player_done_list) if not d])
    train_data.action_list.extend(
        [a for a, d in zip(action, player_done_list) if not d]
    )
    train_data.gae_list.extend(
        [gae for gae, d in zip(gae_list, player_done_list) if not d]
    )
    train_data.v_list.extend([v for v, d in zip(value, player_done_list) if not d])
    train_data.pi_list.extend([p for p, d in zip(prob, player_done_list) if not d])


def create_que_list(index_1: int, index_2: int) -> List[List[Deque]]:
    return [[deque() for _ in range(index_2)] for __ in range(index_1)]


def reset_que(index: int) -> List[Deque]:
    return [deque() for _ in range(index)]


def reset_train_data(train_data: TrainData) -> None:
    train_data.obs_list = []
    train_data.action_list = []
    train_data.gae_list = []
    train_data.v_list = []
    train_data.pi_list = []


def add_to_que_list(
    traget_que_list: List[List[Deque]], add_data_list: List[List[Any]]
) -> None:
    [
        add_to_que(target_q, add_data)
        for target_q, add_data in zip(traget_que_list, add_data_list)
    ]


def add_to_que(target_q: List[Deque], add_data: List[Deque]) -> None:
    [t_q.append(a_d) for t_q, a_d in zip(target_q, add_data)]


def create_padding_data(
    ppo_parameter: PPOParameter,
    train_data: TrainData,
    obs_q: Deque,
    action_q: Deque,
    reward_q: Deque,
    delta_q: Deque,
    value_q: Deque,
    prob_q: Deque,
) -> None:

    if (
        len(obs_q) != len(action_q)
        or len(obs_q) != len(value_q)
        or len(obs_q) != len(prob_q)
    ):
        raise ValueError

    add_delta(delta_q, reward_q[-1], value_q[-1], 0.0, ppo_parameter.gamma)
    if len(delta_q) != ppo_parameter.num_step:
        target_delta_q = copy.deepcopy(delta_q)
        while len(target_delta_q) != ppo_parameter.num_step:
            target_delta_q.append(0.0)
    else:
        target_delta_q = delta_q

    for _ in range(len(obs_q)):
        obs = obs_q.popleft()
        action = action2int(action_q.popleft())
        gae = calc_gae(target_delta_q, ppo_parameter.gamma)
        value = value_q.popleft()
        prob = prob_q.popleft()
        update_PPO_list(train_data, [obs], [action], [gae], [value], [prob], [False])

    # ダミーデータの投入
    obs_q.append(obs)
    action_q.append(ACTIONLIST[0])
    value_q.append(value)
    prob_q.append(prob_q)

    # ゴミ捨て
    delta_q.popleft()


def reshape_step_list(
    action_list: List[Action], value_n_list: np.ndarray, prob_list: np.ndarray
) -> Tuple[List[List[Action]], List[np.ndarray], List[np.ndarray]]:
    reshape_action_list = [
        action_list[i : i + NUM_GEESE] for i in range(0, len(action_list), NUM_GEESE)
    ]
    reshape_value_n_list = [
        value_n_list[i : i + NUM_GEESE] for i in range(0, len(value_n_list), NUM_GEESE)
    ]
    reshape_prob_list = [
        prob_list[i : i + NUM_GEESE] for i in range(0, len(prob_list), NUM_GEESE)
    ]

    return reshape_action_list, reshape_value_n_list, reshape_prob_list


def update_self_PPO_list(
    r_q: Deque,
    o_q: Deque,
    a_q: Deque,
    v_q: Deque,
    p_q: Deque,
    d_q: Deque,
    gae_param: np.ndarray,
    train_data: TrainData,
    before_done: bool,
) -> None:
    r_q.popleft()

    o = o_q.popleft()
    a = action2int(a_q.popleft())
    v = v_q.popleft()
    p = p_q.popleft()
    gae = calc_gae(d_q, gae_param)
    d_q.popleft()

    update_PPO_list(train_data, [o], [a], [gae], [v], [p], [before_done])
