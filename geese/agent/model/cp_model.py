from __future__ import annotations

from dataclasses import asdict
from typing import Tuple

import tensorflow as tf
from geese.structure.parameter import CPModelParameter

NUM_ACTION = 2
MAX_REWARD = 200


@tf.keras.utils.register_keras_serializable()
class CPModel(tf.keras.models.Model):
    def __init__(self, parameter: CPModelParameter):
        super().__init__()
        self._parameter = parameter
        self._dense_layers = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(units=parameter.dim_hidden, activation="relu")
                for _ in range(parameter.num_layers)
            ]
        )
        self._head_p = tf.keras.layers.Dense(units=NUM_ACTION, use_bias=False)
        self._head_v = tf.keras.layers.Dense(units=1, use_bias=False)
        self._flatten = tf.keras.layers.Flatten()

    def call(self, x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        out = self._dense_layers(x)
        flatten = self._flatten(out)
        p = self._head_p(flatten)
        v = self._head_v(flatten)
        p = tf.keras.activations.softmax(p)
        v = tf.keras.activations.tanh(v) * MAX_REWARD
        v = tf.squeeze(v, axis=-1)
        return p, v

    def get_config(self) -> dict:
        return asdict(self._parameter)

    @classmethod
    def from_config(cls, config: dict) -> CPModel:
        parameter = CPModelParameter(**config)
        return CPModel(parameter)
