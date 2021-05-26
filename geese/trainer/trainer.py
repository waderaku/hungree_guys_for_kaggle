from abc import ABC, abstractmethod

import tensorflow as tf

from geese.trainer.minibatch.minibatch import MiniBatch


class Trainer(ABC):
    @abstractmethod
    def train(self, model: tf.keras.models.Model, minibatch: MiniBatch) -> None:
        raise NotImplementedError
