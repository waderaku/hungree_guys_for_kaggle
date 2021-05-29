import numpy as np

from geese.structure.sample.sample import Sample


class PPOSample(Sample):
    def __init__(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        n_step_return: np.ndarray,
        v: np.ndarray,
        v_n: np.ndarray,
        pi: np.ndarray
    ):
        self._observation = observation
        self._action = action
        self._n_step_return = n_step_return
        self._v = v
        self._v_n = v_n
        self._pi = pi
        self._size = self._observation.shape[0]
        self._check_size()

    @property
    def observation(self) -> np.ndarray:
        return self._observation

    @property
    def action(self) -> np.ndarray:
        return self._action

    @property
    def n_step_return(self) -> np.ndarray:
        return self._n_step_return

    @property
    def v(self) -> np.ndarray:
        return self._v

    @property
    def v_n(self) -> np.ndarray:
        return self._v_n

    @property
    def pi(self) -> np.ndarray:
        return self._pi

    def __len__(self) -> int:
        return self._size

    def _check_size(self) -> None:
        size = self._size
        assert self._action.shape[0] == size
        assert self._n_step_return.shape[0] == size
        assert self._v.shape[0] == size
        assert self._v_n.shape[0] == size
        assert self._pi.shape[0] == size
