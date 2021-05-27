from geese.structure.parameter.parameter import Parameter


class TrainerParameter(Parameter):
    pass


class PPOTrainerParameter(TrainerParameter):
    def __init__(self, learning_rate: float, batch_size: int, num_epoch: int, clip_eps: float):
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._num_epoch = num_epoch
        self._clip_eps = clip_eps

    @property
    def learning_rate(self) -> float:
        return self._learning_rate

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def num_epoch(self) -> int:
        return self._num_epoch

    @property
    def clip_eps(self) -> float:
        return self._clip_eps
