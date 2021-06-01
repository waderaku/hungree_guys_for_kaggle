from dataclasses import dataclass

import tensorflow as tf
from geese.structure.parameter.parameter import Parameter


@dataclass
class AgentParameter(Parameter):
    model: tf.keras.models.Model
