from geese.structure.parameter.parameter import Parameter
from dataclasses import dataclass


class TrainerParameter(Parameter):
    pass


@dataclass(frozen=True)
class PPOTrainerParameter(TrainerParameter):
    learning_rate: float
    batch_size: int
    num_epoch: int
    clip_eps: float
    entropy_coefficient: float
