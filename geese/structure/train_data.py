from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class TrainData:
    obs_list: List[np.ndarray] = field(default_factory=list)
    action_list: List[int] = field(default_factory=list)
    gae_list: List[float] = field(default_factory=list)
    v_list: List[float] = field(default_factory=list)
    pi_list: List[np.ndarray] = field(default_factory=list)
