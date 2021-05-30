from geese.structure.parameter import PPOTrainerParameter
import numpy as np
from geese.structure.parameter import Parameter
from geese.structure.parameter import EnvParameter
from geese.structure.parameter import AgentParameter


class PPOParameter(Parameter):
    def __init__(
        self,
        num_parallels: int,
        num_step: int,
        gamma: float,
        param_lambda: float,
        num_sample_size: int,
        ppo_trainer_parameter: PPOTrainerParameter,
        env_parameter: EnvParameter,
        agent_parameter: AgentParameter
    ):
        self._num_parallels = num_parallels
        self._env_parameter = env_parameter
        self._num_step = num_step
        self._gamma = gamma
        self._gae_param = self._create_gae_param(gamma, param_lambda, num_step)
        self._num_sample_size = num_sample_size
        self._ppo_trainer_parameter = ppo_trainer_parameter
        self._agent_parameter = agent_parameter

    def _create_gae_param(self, gamma: float, param_lambda: float, num_step: int) -> np.ndarray:
        return np.geomspace(1, (gamma*param_lambda)**(num_step-1), num_step)

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
