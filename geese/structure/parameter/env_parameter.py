from dataclasses import dataclass
from typing import List, Optional

from geese.constants import RewardFunc
from geese.structure.parameter import Parameter


@dataclass(frozen=True)
class EnvParameter(Parameter):
    reward_func: RewardFunc
    max_reward_value: float
    reward_list: Optional[List[float]] = None
    scale_flg: bool = False
    press_flg: bool = False

    def __post_init__(self):
        if self.reward_func == RewardFunc.RANK:
            assert isinstance(self.reward_list, list)
