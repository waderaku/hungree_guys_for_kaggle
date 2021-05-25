from abc import ABC, abstractmethod

import tensorflow as tf


class MiniBatch(ABC):
    pass


class Trainer(ABC):
    @abstractmethod
    def train(self, model: tf.keras.models.Model, minibatch: MiniBatch) -> None:
        raise NotImplementedError
