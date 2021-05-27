from geese.structure.parameter.env_parameter import EnvParameter
from geese.env.env import Env
from geese.structure import Observation, Reward
from typing import List, Tuple


class VecEnv:
    def __init__(self, num_parallels: int, parameter: EnvParameter):
        self._num_parallels = num_parallels
        self._envs = [Env(parameter) for _ in range(num_parallels)]

    def reset(self) -> List[List[Observation]]:
        return [env.reset() for env in self._envs]

    def step(self, action_list: List[List[Observation]]) -> Tuple[List[List[Observation]], List[List[Reward]], List[List[bool]]]:
        return [env.step(action) for env, action in zip(self._envs, action_list)]
