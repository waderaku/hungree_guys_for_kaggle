from geese.structure.parameter import Parameter
from typing import List


class EnvParameter(Parameter):
    def __init__(self, reward_list: List[float]):
        self._reward_list = reward_list

    @property
    def reward_list(self) -> float:
        return self._reward_list
