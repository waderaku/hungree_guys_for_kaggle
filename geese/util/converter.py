from geese.constants import FIELD_HEIGHT, FIELD_WIDTH, NUM_CHANNELS
import numpy as np
import tensorflow as tf
from kaggle_environments.envs.hungry_geese.hungry_geese import Action
from kaggle_environments.envs.hungry_geese.hungry_geese import Observation as KaggleObservation


def to_np_obs(obs: KaggleObservation, last_obs: KaggleObservation) -> tf.Tensor:
    index = obs.index
    b = np.zeros((NUM_CHANNELS, FIELD_HEIGHT * FIELD_WIDTH), dtype=np.float32)

    for p, pos_list in enumerate(obs.geese):
        # head position
        for pos in pos_list[:1]:
            b[0 + (p - index) % 4, pos] = 1
        # tip position
        for pos in pos_list[-1:]:
            b[4 + (p - index) % 4, pos] = 1
        # whole position
        for pos in pos_list:
            b[8 + (p - index) % 4, pos] = 1

    # previous head position
    if last_obs is not None:
        for p, pos_list in enumerate(last_obs.geese):
            for pos in pos_list[:1]:
                b[12 + (p - index) % 4, pos] = 1

    # food
    for pos in obs.food:
        b[16, pos] = 1

    return b.reshape(-1, FIELD_HEIGHT, FIELD_WIDTH)


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
