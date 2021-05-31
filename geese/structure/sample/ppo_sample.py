from dataclasses import dataclass

import numpy as np
from geese.structure.sample.sample import Sample


@dataclass(frozen=True)
class PPOSample(Sample):
    observation: np.ndarray
    action: np.ndarray
    gae: np.ndarray
    v: np.ndarray
    pi: np.ndarray

    def __post_init__(self):
        self._check_size()

    def _check_size(self) -> None:
        size = self.observation.shape[0]
        assert self.action.shape[0] == size
        assert self.gae.shape[0] == size
        assert self.v.shape[0] == size
        assert self.pi.shape[0] == size

    def __len__(self):
        return self.observation.shape[0]
