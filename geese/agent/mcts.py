import tensorflow as tf
from geese.structure import Observation


class MCTS:
    def __init__(self, model: tf.keras.models.Model):
        self.model = model

    def get_prob(self) -> tf.Tensor:
        raise NotImplementedError

    def search(self, obs: Observation, last_obs: Observation) -> tf.Tensor:
        raise NotImplementedError
