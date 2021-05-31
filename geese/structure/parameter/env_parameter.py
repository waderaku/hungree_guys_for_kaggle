from dataclasses import dataclass
from typing import List

from geese.structure.parameter import Parameter


@dataclass(frozen=True)
class EnvParameter(Parameter):
    reward_list: List[float]
