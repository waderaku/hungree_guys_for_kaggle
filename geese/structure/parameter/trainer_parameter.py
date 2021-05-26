from geese.structure.parameter.parameter import Parameter


class TrainerParameter(Parameter):
    pass


class PPOTrainerParameter(TrainerParameter):
    def __init__(self, learning_rate: float, clip_eps: float):
        self._learning_rate = learning_rate
        self._clip_eps = clip_eps

    @property
    def learning_rate(self) -> float:
        return self._learning_rate

    @property
    def clip_eps(self) -> float:
        return self._clip_eps
