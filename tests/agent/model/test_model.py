import tensorflow as tf
from geese.agent.model.model import BaseModel
from geese.agent.model.parameter import BaseModelParameter
from geese.constants import ACTIONLIST


def test_base_model_ioshape():
    B, C, H, W = 32, 17, 7, 11
    num_layers = 12
    num_filters = 32
    kernel_size = (3, 3)
    bn = True
    parameter = BaseModelParameter(
        num_layers=num_layers,
        num_filters=num_filters,
        kernel_size=kernel_size,
        bn=bn,
    )
    model = BaseModel(parameter)
    inputs = tf.random.normal((B, C, H, W))
    p, v = model(inputs)
    assert p.shape == (B, len(ACTIONLIST))
    assert v.shape == (B, 1)
