from typing import List, Tuple

from geese.env import SoloEnv
from geese.structure import Observation, Reward
from geese.structure.parameter.env_parameter import EnvParameter
from kaggle_environments.envs.hungry_geese.hungry_geese import Action


class VecSoloEnv:
    def __init__(self, num_parallels: int, parameter: EnvParameter):
        self._num_parallels = num_parallels
        self._envs = [SoloEnv(parameter) for _ in range(num_parallels)]

    def reset(self) -> List[Observation]:
        return [env.reset() for env in self._envs]

    def step(
        self, action_list: List[Action]
    ) -> Tuple[List[Observation], List[Reward], List[bool]]:
        ret = tuple(
            zip(*[env.step(action) for env, action in zip(self._envs, action_list)])
        )
        return tuple(map(list, ret))
