from dataclasses import dataclass
from typing import List


@dataclass
class TrainData():
    obs_list: List
    action_list: List
    n_step_return_list: List
    v_list: List
    v_n_list: List
    pi_list: List
