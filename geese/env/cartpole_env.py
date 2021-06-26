from geese.structure.parameter.cpenv_parameter import CPEnvParameter
from typing import Tuple, List
import numpy as np
import gym


class CPEnv:
    def __init__(self, _: CPEnvParameter):
        self._gym_env = gym.make("CartPole-v0")

    def reset(self) -> np.ndarray:
        return self._gym_env.reset()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        obs, reward, done, _ = self._gym_env.step(action)
        if done:
            obs = self.reset()
        return obs, reward, done


class VecCPEnv:
    def __init__(self, num_parallels, parameter: CPEnvParameter):
        self._envs = [CPEnv(parameter) for _ in range(num_parallels)]

    def reset(self) -> List[np.ndarray]:
        return [env.reset() for env in self._envs]

    def step(
        self, action: List[int]
    ) -> Tuple[List[np.ndarray], List[float], List[bool]]:
        ret = tuple(zip(*[env.step(act) for env, act in zip(self._envs, action)]))
        obs, reward, done = tuple(map(list, ret))
        return obs, reward, done
