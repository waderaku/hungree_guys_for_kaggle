import tensorflow as tf


class Agent():
    def __init__(
        self,
        model: tf.keras.models.Model,
    ):
        self.model = model

    def get_action(self, tf_obs: tf.Tensor) -> tf.Tensor:
        pass

    def save(self, path: str) -> None:
        raise NotImplementedError

    def load(self, path: str) -> None:
        raise NotImplementedError
