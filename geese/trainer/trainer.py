from abc import ABC, abstractmethod

import tensorflow as tf

from geese.structure.sample import Sample


class Trainer(ABC):
    @abstractmethod
    def train(self, model: tf.keras.models.Model, sample: Sample) -> None:
        raise NotImplementedError
