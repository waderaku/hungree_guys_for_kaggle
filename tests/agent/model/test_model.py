import shutil
import tensorflow as tf
from geese.agent.model.model import BaseModel
from geese.constants import ACTIONLIST
from geese.structure.parameter import BaseModelParameter


def test_base_model_ioshape():
    B, C, H, W = 32, 17, 7, 11
    num_layers = 12
    num_filters = 32
    kernel_size = (3, 3)
    bn = True
    use_gpu = False
    parameter = BaseModelParameter(
        num_layers=num_layers,
        num_filters=num_filters,
        kernel_size=kernel_size,
        bn=bn,
        use_gpu=use_gpu,
    )
    model = BaseModel(parameter)
    inputs = tf.random.normal((B, C, H, W))
    p, v = model(inputs)
    assert p.shape == (B, len(ACTIONLIST))
    assert v.shape == (B)


def test_base_model_save_load():
    B, C, H, W = 32, 17, 7, 11
    num_layers = 12
    num_filters = 32
    kernel_size = (3, 3)
    bn = True
    use_gpu = False
    path = "./temp"
    parameter = BaseModelParameter(
        num_layers=num_layers,
        num_filters=num_filters,
        kernel_size=kernel_size,
        bn=bn,
        use_gpu=use_gpu,
    )
    model = BaseModel(parameter)
    inputs = tf.random.normal((B, C, H, W))
    p, v = model(inputs)
    model.save(path)
    new_model = tf.keras.models.load_model(path)
    p2, v2 = new_model(inputs)
    shutil.rmtree(path)
    assert tf.reduce_all(p == p2)
    assert tf.reduce_all(v == v2)
