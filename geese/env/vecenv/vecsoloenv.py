from typing import List, Tuple

from geese.env.env import Env
from geese.structure import Observation, Reward
from geese.structure.parameter.env_parameter import EnvParameter
from kaggle_environments.envs.hungry_geese.hungry_geese import Action


class VecSoloEnv:
    def __init__(self, num_parallels: int, parameter: EnvParameter):
        raise NotImplementedError

    def reset(self) -> List[Observation]:
        raise NotImplementedError

    def step(
        self, action_list: List[Action]
    ) -> Tuple[List[Observation], List[Reward], List[bool]]:
        raise NotImplementedError
