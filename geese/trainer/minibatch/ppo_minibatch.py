import numpy as np

from geese.trainer.minibatch.minibatch import MiniBatch


class PPOMiniBatch(MiniBatch):
    def __init__(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        n_step_return: np.ndarray,
        v_0: np.ndarray,
        v_n: np.ndarray,
        pi: np.ndarray
    ):
        self._observation = observation
        self._action = action
        self._n_step_return = n_step_return
        self._v_0 = v_0
        self._v_n = v_n
        self._pi = pi

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
    def v_0(self) -> np.ndarray:
        return self._v_0

    @property
    def v_n(self) -> np.ndarray:
        return self._v_n

    @property
    def pi(self) -> np.ndarray:
        return self._pi
