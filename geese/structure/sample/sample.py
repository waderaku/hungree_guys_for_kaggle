from abc import ABC, abstractmethod


class Sample(ABC):
    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError
