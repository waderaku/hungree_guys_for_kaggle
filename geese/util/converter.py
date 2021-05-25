import tensorflow as tf
import numpy as np

from geese.structure import Observation
from geese.constants import FIELD_HEIGHT, FIELD_WIDTH, NUM_CHANNELS


def to_tf_tensor(obs: Observation) -> tf.Tensor:
    last_obs = obs.last_obs
    index = obs.index
    obs = obs.now_obs
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

    return tf.convert_to_tensor(b.reshape(-1, FIELD_HEIGHT, FIELD_WIDTH))
