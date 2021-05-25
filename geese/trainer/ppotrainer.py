import tensorflow as tf

from geese.trainer.trainer import MiniBatch, Trainer
from geese.trainer.parameter import PPOTrainerParameter


class PPOMiniBatch(MiniBatch):
    def __init__(self):
        raise NotImplementedError


class PPOTrainer(Trainer):
    def __init__(self, parameter: PPOTrainerParameter):
        raise NotImplementedError

    def train(self, model: tf.keras.models.Model, minibatch: PPOMiniBatch):
        raise NotImplementedError
