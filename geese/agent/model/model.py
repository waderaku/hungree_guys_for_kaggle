import tensorflow as tf
from geese.agent.model.parameter import BaseModelParameter, TorusConv2dParameter
from geese.constants import ACTIONLIST


class BaseModel(tf.keras.models.Model):
    def __init__(self, parameter: BaseModelParameter):
        super().__init__()
        self._init_block = TorusConv2d(parameter.torusconv2d_parameter)
        self._blocks = [TorusConv2d(parameter.torusconv2d_parameter)
                        for _ in range(parameter.num_layers)]
        self._head_p = tf.keras.layers.Dense(
            units=len(ACTIONLIST), use_bias=False)
        self._head_v = tf.keras.layers.Dense(units=1, use_bias=False)
        self._flatten = tf.keras.layers.Flatten()

    def call(self, x: tf.Tensor) -> tf.Tensor:
        out = self._init_block(x)
        for block in self._blocks:
            out = tf.keras.activations.relu(block(out) + out)
        flatten = self._flatten(out)
        p = self._head_p(flatten)
        v = self._head_v(flatten)
        p = tf.keras.activations.softmax(p)
        v = tf.keras.activations.tanh(v)
        return p, v


class TorusConv2d(tf.keras.layers.Layer):
    def __init__(self, parameter: TorusConv2dParameter):
        super().__init__()
        self._edge_size = (
            parameter.kernel_size[0] // 2, parameter.kernel_size[1] // 2)
        self._conv = tf.keras.layers.Conv2D(
            filters=parameter.num_filters,
            kernel_size=parameter.kernel_size,
            data_format="channels_first",
        )
        self._bn = tf.keras.layers.BatchNormalization() if parameter.bn else None

    def call(self, x: tf.Tensor) -> tf.Tensor:
        out = tf.concat([x[:, :, :, -self._edge_size[1]:],
                        x, x[:, :, :, :self._edge_size[1]]], axis=3)
        out = tf.concat([out[:, :, -self._edge_size[0]:], out,
                        out[:, :, :self._edge_size[0]]], axis=2)
        out = self._conv(out)
        if self._bn is not None:
            out = self._bn(out)
        return out
