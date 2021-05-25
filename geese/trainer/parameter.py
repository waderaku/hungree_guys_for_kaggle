from abc import ABC


class TrainerParameter(ABC):
    pass


class PPOTrainerParameter(TrainerParameter):
    def __init__(self, clip_eps: float):
        self._clip_eps = clip_eps

    @property
    def clip_eps(self) -> float:
        return self._clip_eps
