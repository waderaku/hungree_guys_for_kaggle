from collections import deque
from typing import Any, Deque

import numpy as np
from geese.controller.ppo_helper import (
    add_delta,
    add_delta_list,
    calc_gae,
    calc_gae_list,
    create_padding_data,
)
from geese.structure.parameter.ppo_parameter import PPOParameter
from geese.structure.train_data import TrainData
from geese.constants import ACTIONLIST
from copy import deepcopy
from geese.util.converter import action2int


def test_add_delta():
    d_q = deque()
    reward = 1.0
    v_new = 3.0
    v_old = 1.0
    gamma = 0.5
    add_delta(d_q, reward, v_old, v_new, gamma)
    delta = d_q.popleft()
    assert delta == 1.5


def test_add_delta_list():
    num_queue = 32
    delta_que = [deque() for _ in range(num_queue)]
    reward_list = [float(i + 1) for i in range(num_queue)]
    v_new_list = [float(i + 1) * 3.0 for i in range(num_queue)]
    v_old_list = [float(i + 1) * 1.0 for i in range(num_queue)]
    gamma = 0.5
    add_delta_list(delta_que, reward_list, v_new_list, v_old_list, gamma)
    deltas = [queue.pop() for queue in delta_que]
    for i, delta in enumerate(deltas):
        assert delta == (i + 1) * 1.5


def test_calc_gae():
    base = 1.0
    num_step = 4
    gamma = 0.5
    lmd = 0.5
    queue = deque([base * (i + 1) for i in range(num_step)])
    param = PPOParameter._create_gae_param(gamma, lmd, num_step)
    gae = calc_gae(queue, param)
    assert np.isclose(
        gae,
        (1.0 * 0.25 ** 0 + 2.0 * 0.25 ** 1 + 3.0 * 0.25 ** 2 + 4.0 * 0.25 ** 3) * base,
    )


def test_calc_gae_list():
    num_step = 4
    num_queue = 32
    gamma = 0.5
    lmd = 0.5
    list_queue = [
        deque([base * (i + 1) for i in range(num_step)])
        for base in range(1, num_queue + 1)
    ]
    param = PPOParameter._create_gae_param(gamma, lmd, num_step)
    gaes = calc_gae_list(list_queue, param)
    for base, gae in zip(range(1, num_queue + 1), gaes):
        assert np.isclose(
            gae,
            (1.0 * 0.25 ** 0 + 2.0 * 0.25 ** 1 + 3.0 * 0.25 ** 2 + 4.0 * 0.25 ** 3)
            * base,
        )


def cp(val: Any) -> Any:
    if isinstance(val, TrainData):
        return val
    else:
        return deepcopy(val)


def to_string(array: np.ndarray) -> str:
    array = array.flatten()
    array = array.tolist()
    return "".join(map(str, array))


def test_create_padding_data_notfull():
    ppo_param = PPOParameter(
        num_parallels=None,
        num_step=4,
        gamma=0.5,
        param_lambda=0.5,
        num_sample_size=None,
        save_freq=None,
        save_dir=None,
        ppo_trainer_parameter=None,
        env_parameter=None,
        agent_parameter=None,
        reward_log_freq=None,
    )
    # num_step未満
    test_step = 2
    train_data = TrainData()
    obs_shape = (17, 40, 21)
    obs_q = deque([np.ones(shape=obs_shape) * (i + 1) for i in range(test_step)])
    action_q = deque([ACTIONLIST[i] for i in range(test_step)])
    reward_q = deque([1.0 for _ in range(test_step)])
    delta_q = deque([1.0])
    value_q = deque([1.0 * (i + 1) for i in range(test_step)])
    prob_shape = (len(ACTIONLIST),)
    prob_q = deque([np.ones(shape=prob_shape) for _ in range(test_step)])

    create_padding_data(
        *map(
            cp,
            [
                ppo_param,
                train_data,
                obs_q,
                action_q,
                reward_q,
                delta_q,
                value_q,
                prob_q,
            ],
        )
    )

    assert set(map(to_string, obs_q)) == set(map(to_string, train_data.obs_list))
    assert set(map(action2int, action_q)) == set(train_data.action_list)
    assert set(value_q) == set(train_data.v_list)
    assert set(map(to_string, prob_q)) == set(map(to_string, train_data.pi_list))
    assert set([1 - 0.25 ** 1, -1]) == set(train_data.gae_list)


def test_create_padding_data_full():
    ppo_param = PPOParameter(
        num_parallels=None,
        num_step=2,
        gamma=0.5,
        param_lambda=0.5,
        num_sample_size=None,
        save_freq=None,
        save_dir=None,
        ppo_trainer_parameter=None,
        env_parameter=None,
        agent_parameter=None,
        reward_log_freq=None,
    )
    # num_step未満
    test_step = 2
    train_data = TrainData()
    obs_shape = (17, 40, 21)
    obs_q = deque([np.ones(shape=obs_shape) * (i + 1) for i in range(test_step)])
    action_q = deque([ACTIONLIST[i] for i in range(test_step)])
    reward_q = deque([1.0 for _ in range(test_step)])
    delta_q = deque([1.0])
    value_q = deque([1.0 * (i + 1) for i in range(test_step)])
    prob_shape = (len(ACTIONLIST),)
    prob_q = deque([np.ones(shape=prob_shape) for _ in range(test_step)])

    create_padding_data(
        *map(
            cp,
            [
                ppo_param,
                train_data,
                obs_q,
                action_q,
                reward_q,
                delta_q,
                value_q,
                prob_q,
            ],
        )
    )

    assert set(map(to_string, obs_q)) == set(map(to_string, train_data.obs_list))
    assert set(map(action2int, action_q)) == set(train_data.action_list)
    assert set(value_q) == set(train_data.v_list)
    assert set(map(to_string, prob_q)) == set(map(to_string, train_data.pi_list))
    assert set([1 - 0.25 ** 1, -1]) == set(train_data.gae_list)
