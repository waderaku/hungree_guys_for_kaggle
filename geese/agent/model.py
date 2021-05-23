import tensorflow as tf


class BaseModel(tf.keras.models.Model):
    def __init__(self, network_parameter: dict):
        raise NotImplementedError

    def call(self, x: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError
