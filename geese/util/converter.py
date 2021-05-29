import tensorflow as tf
from kaggle_environments.envs.hungry_geese.hungry_geese import Action


def action2int(action: Action) -> int:
    if action == Action.NORTH:
        return 0
    elif action == Action.SOUTH:
        return 1
    elif action == Action.WEST:
        return 2
    elif action == Action.EAST:
        return 3
    else:
        raise ValueError("Unexpected Action Input")


def type32(tensor: tf.Tensor):
    if tensor.dtype in (tf.float16, tf.float32, tf.float64):
        dtype = tf.float32
    elif tensor.dtype in (tf.int16, tf.int32, tf.int64):
        dtype = tf.int32
    elif tensor.dtype == tf.bool:
        dtype = tf.bool
    else:
        raise TypeError("Unexpected Type")
    return tf.cast(tensor, dtype)
