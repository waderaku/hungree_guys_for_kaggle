from __future__ import annotations
import datetime
from collections import deque
from dataclasses import dataclass, fields
from geese.structure.sample.ppo_sample import PPOSample
from typing import Any, Deque, Iterable, List, Tuple

import numpy as np
from geese.agent.cartpole_agent import CartPoleAgent
from geese.constants import LOG_BASE_DIR
from geese.controller.controller import Controller
from geese.env.cartpole_env import VecCPEnv
from geese.structure.parameter import PPOParameter
from geese.structure.train_data import TrainData
from geese.trainer.ppo_trainer import PPOTrainer
from geese.util.tensor_boad_logger import TensorBoardLogger


@dataclass
class Memory:
    obs: List[Deque[np.ndarray]]
    action: List[Deque[int]]
    v: List[Deque[float]]
    delta: List[Deque[float]]
    pi: List[Deque[np.ndarray]]

    def add(
        self,
        obs: List[np.ndarray],
        action: List[int],
        v: List[float],
        delta: List[float],
        pi: List[float],
        pre_done: List[bool],
    ) -> None:
        for i, pd in enumerate(pre_done):
            if not pd:
                env_memory = self.indexof(i)
                env_memory.add(
                    obs[i],
                    action[i],
                    v[i],
                    delta[i],
                    pi[i],
                )

    def indexof(self, index: int) -> EnvWiseMemory:
        return EnvWiseMemory(
            self.obs[index],
            self.action[index],
            self.v[index],
            self.delta[index],
            self.pi[index],
        )


@dataclass
class EnvWiseMemory:
    obs: Deque[np.ndarray]
    action: Deque[int]
    v: Deque[float]
    delta: Deque[float]
    pi: Deque[np.ndarray]

    def add(
        self,
        obs: np.ndarray,
        action: int,
        v: float,
        delta: float,
        pi: float,
    ):
        self.obs.append(obs)
        self.action.append(action)
        self.v.append(v)
        self.delta.append(delta)
        self.pi.append(pi)

    def clear(self):
        self.obs.clear()
        self.action.clear()
        self.v.clear()
        self.delta.clear()
        self.pi.clear()

    def __len__(self):
        return len(self.obs)

    def __iter__(self) -> Iterable[Deque[Any]]:
        return iter([self.obs, self.action, self.v, self.delta, self.pi])

    @property
    def maxlen(self):
        return self.obs.maxlen


def fill_with_dummy(env_memory: EnvWiseMemory) -> None:
    num_fill = env_memory.maxlen - len(env_memory)
    for target in env_memory:
        target: Deque[Any]
        fill = [0.0 for _ in range(num_fill)]
        target.extend(fill)


def memory_factory(max_length: int, num_parallels: int) -> List[Deque[Any]]:
    return [deque(maxlen=max_length) for _ in range(num_parallels)]


def get_training_objects(
    parameter: PPOParameter,
) -> Tuple[TrainData, Memory, PPOTrainer, VecCPEnv, CartPoleAgent, TensorBoardLogger]:
    today = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    logger = TensorBoardLogger(f"{LOG_BASE_DIR}/{today}")
    train_data = TrainData()
    memory = Memory(
        *[
            memory_factory(parameter.num_step, parameter.num_parallels)
            for _ in range(len(fields(Memory)))
        ]
    )
    ppo_trainer = PPOTrainer(parameter.ppo_trainer_parameter, logger)
    vec_env = VecCPEnv(parameter.num_parallels, parameter.env_parameter)
    agent = CartPoleAgent(parameter.agent_parameter)
    return train_data, memory, ppo_trainer, vec_env, agent, logger


def _compute_delta(reward: float, v: float, v_next: float, gamma: float) -> float:
    return reward + gamma * v_next - v


def compute_delta(
    reward: List[float], v: List[float], v_next: List[float], gamma: float
) -> List[float]:
    return [_compute_delta(r, _v, vn, gamma) for r, _v, vn in zip(reward, v, v_next)]


def add_train_data_envwise(
    train_data: TrainData,
    env_memory: EnvWiseMemory,
    gae_param: np.ndarray,
) -> None:
    if len(env_memory.obs) != env_memory.obs.maxlen:
        return
    gae = np.sum(np.array(env_memory.delta) * gae_param)
    target: List[Deque] = [
        train_data.obs_list,
        train_data.action_list,
        train_data.gae_list,
        train_data.v_list,
        train_data.pi_list,
    ]
    new_comer = [
        env_memory.obs[0],
        env_memory.action[0],
        gae,
        env_memory.v[0],
        env_memory.pi[0],
    ]
    for t, n in zip(target, new_comer):
        t.append(n)


def add_train_data(
    train_data: TrainData, memory: Memory, parameter: PPOParameter
) -> None:
    for i in range(parameter.num_parallels):
        env_memory = memory.indexof(i)
        add_train_data_envwise(train_data, env_memory, parameter.gae_param)


def on_episode_end(
    train_data: TrainData,
    memory: Memory,
    obs: List[np.ndarray],
    action: List[np.ndarray],
    reward: List[np.ndarray],
    value: List[float],
    pi: List[float],
    value_next: List[float],
    done: List[bool],
    parameter: PPOParameter,
) -> None:
    for i, o, a, r, v, p, v_next, d in zip(
        range(parameter.num_step), obs, action, reward, value, pi, value_next, done
    ):
        if d:
            delta = _compute_delta(r, v, v_next, parameter.gamma)
            env_memory = memory.indexof(i)
            print(len(env_memory))
            env_memory.add(o, a, v, delta, p)
            add_train_data_envwise(train_data, env_memory, parameter.gae_param)
            padding(train_data, env_memory, parameter.gae_param)
            env_memory.clear()


def padding(train_data: TrainData, env_memory: EnvWiseMemory, gae_param: np.ndarray):
    num_padding = len(env_memory) - 1
    fill_with_dummy(env_memory)
    for _ in range(num_padding):
        pad(env_memory)
        add_train_data_envwise(train_data, env_memory, gae_param)


def pad(env_memory: EnvWiseMemory) -> None:
    for target in env_memory:
        target: Deque[Any]
        target.append(0.0)


class CartPoleController(Controller):
    def __init__(self, parameter: PPOParameter):
        self._parameter = parameter

    def train(self):
        # Initialize
        train_data, memory, ppo_trainer, vec_env, agent, logger = get_training_objects(
            self._parameter
        )
        obs = vec_env.reset()
        action, pi, v = agent.step(obs)
        v_next = [0.0 for _ in range(self._parameter.num_parallels)]
        while True:
            obs_next, reward, done = vec_env.step(action)
            on_episode_end(
                train_data,
                memory,
                obs,
                action,
                reward,
                v,
                pi,
                v_next,
                done,
                self._parameter,
            )
            action, pi_next, v_next = agent.step(obs)
            delta = compute_delta(reward, v, v_next, self._parameter.gamma)
            memory.add(obs, action, v, delta, pi, done)
            add_train_data(train_data, memory, self._parameter)
            if len(train_data) >= self._parameter.num_sample_size:
                sample = PPOSample(
                    np.array(train_data.obs_list),
                    np.array(train_data.action_list),
                    np.array(train_data.gae_list),
                    np.array(train_data.v_list),
                    np.array(train_data.pi_list),
                )
                ppo_trainer.train(agent.model, sample)
                train_data = TrainData()
            obs, v, pi = obs_next, v_next, pi_next
