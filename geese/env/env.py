from geese.env.dena_env import Environment as DenaEnv
from geese.structure import Observation, Reward
from typing import List, Tuple
from geese.constants import ACTIONLIST


class Env:
    N_ACTION = len(ACTIONLIST)

    def __init__(self):
        self._env = DenaEnv()

    def reset(self) -> List[Observation]:
        self._env.reset()
        return [self._env.observation(p) for p in range(Env.N_ACTION)]

    def step(self, actions: List[int]) -> Tuple[List[Observation], List[Reward], List[bool]]:
        actions = {p: actions[p] for p in range(Env.N_ACTION)}
        self._env.step(actions)
        reward = [self._env.env.state[p]["reward"]
                  for p in range(len(actions))]
        done = [True if self._env.env.state[p]["status"] != "ACTIVE"
                else False for p in range(Env.N_ACTION)]
        return [self._env.observation(p) for p in range(Env.N_ACTION)], reward, done

    def __str__(self) -> str:
        return str(self._env)
