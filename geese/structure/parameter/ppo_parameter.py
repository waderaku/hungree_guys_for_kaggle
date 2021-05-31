from dataclasses import InitVar, dataclass, field
from re import A

import numpy as np
from geese.structure.parameter import (AgentParameter, EnvParameter, Parameter,
                                       PPOTrainerParameter)


@dataclass
class PPOParameter(Parameter):
    num_parallels: int
    num_step: int
    gamma: float
    param_lambda: InitVar[float]
    num_sample_size: int
    save_freq: int
    save_dir: str
    ppo_trainer_parameter: PPOTrainerParameter
    env_parameter: EnvParameter
    agent_parameter: AgentParameter
    gae_param: np.ndarray = field(init=False)

    def __post_init__(
        self,
        param_lambda
    ):
        self.gae_param = self._create_gae_param(
            self.gamma, param_lambda, self.num_step)

    def _create_gae_param(self, gamma: float, param_lambda: float, num_step: int) -> np.ndarray:
        return np.geomspace(1, (gamma*param_lambda)**(num_step-1), num_step)
