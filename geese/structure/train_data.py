from dataclasses import dataclass, field
from geese.structure.sample.ppo_sample import PPOSample
from typing import List

import numpy as np


@dataclass
class TrainData:
    obs_list: List[np.ndarray] = field(default_factory=list)
    action_list: List[int] = field(default_factory=list)
    gae_list: List[float] = field(default_factory=list)
    v_list: List[float] = field(default_factory=list)
    pi_list: List[np.ndarray] = field(default_factory=list)

    def as_ppo_sample(self) -> PPOSample:
        return PPOSample(
            np.ndarray(self.obs_list),
            np.ndarray(self.action_list),
            np.ndarray(self.gae_list),
            np.ndarray(self.v_list),
            np.ndarray(self.pi_list),
        )

    def __len__(self):
        return len(self.obs_list)
