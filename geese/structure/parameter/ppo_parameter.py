from geese.structure.parameter import PPOTrainerParameter
import numpy as np
from typing import List
from geese.structure.parameter import Parameter
from geese.structure.parameter import EnvParameter
from geese.structure.parameter import AgentParameter


class PPOParameter(Parameter):
    def __init__(
        self,
        num_parallels: int,
        num_step: int,
        gamma: float,
        num_sample_size: int,
        ppo_trainer_parameter: PPOTrainerParameter,
        env_parameter: EnvParameter,
        agent_parameter: AgentParameter
    ):
        self._num_parallels = num_parallels
        self._env_parameter = env_parameter
        self._num_step = num_step
        self._gamma = self._create_gamma(gamma, num_step)
        self._num_sample_size = num_sample_size
        self._ppo_trainer_parameter = ppo_trainer_parameter
        self._agent_parameter = agent_parameter

        self._obs_list = []
        self._action_list = []
        self._n_step_return_list = []
        self._v_list = []
        self._v_n_list = []
        self._pi_list = []

    def _create_gamma(self, gamma: float, num_step: int) -> np.ndarray:
        return np.geomspace(1, gamma**(num_step-1), num_step)

    @ property
    def num_parallels(self) -> int:
        return self._num_parallels

    @ property
    def env_parameter(self) -> EnvParameter:
        return self._env_parameter

    @ property
    def num_step(self) -> int:
        return self._num_step

    @ property
    def gamma(self) -> np.ndarray:
        return self._gamma

    @ property
    def num_sample_size(self) -> int:
        return self._num_sample_size

    @property
    def ppo_trainer_parameter(self) -> PPOTrainerParameter:
        return self._ppo_trainer_parameter

    @property
    def agent_parameter(self) -> AgentParameter:
        return self._agent_parameter

    @ property
    def obs_list(self) -> List[np.ndarray]:
        return self._obs_list

    @ property
    def n_step_return_list(self) -> List[np.ndarray]:
        return self._n_step_return_list

    @ property
    def action_list(self) -> List[np.ndarray]:
        return self._action_list

    @ property
    def v_list(self) -> List[np.ndarray]:
        return self._v_list

    @ property
    def v_n_list(self) -> List[np.ndarray]:
        return self._v_n_list

    @ property
    def pi_list(self) -> List[np.ndarray]:
        return self._pi_list
