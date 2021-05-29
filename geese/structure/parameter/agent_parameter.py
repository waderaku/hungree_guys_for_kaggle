from geese.structure.parameter.parameter import Parameter
import tensorflow as tf


class AgentParameter(Parameter):
    def __init__(self, model: tf.keras.models.Model):
        self._model = model

    @property
    def model(self) -> tf.keras.models.Model:
        return self._model
