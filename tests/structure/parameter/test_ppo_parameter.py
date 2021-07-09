from geese.structure.parameter.ppo_parameter import PPOParameter
import numpy as np


def test_create_gae_param():
    gamma = 0.5
    lmd = 0.5
    param = PPOParameter._create_gae_param(gamma, lmd, 4)
    assert np.all(
        np.isclose(param, np.array([0.25 ** 0, 0.25 ** 1, 0.25 ** 2, 0.25 ** 3]))
    )
