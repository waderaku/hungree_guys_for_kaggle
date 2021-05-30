from dataclasses import dataclass
from typing import List


@dataclass
class TrainData():
    obs_list: List
    action_list: List
    gae_list: List
    v_list: List
    pi_list: List
