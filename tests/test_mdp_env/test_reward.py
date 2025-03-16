import numpy as np
from mdp_env.reward import reward_eval


def test_cara_reward():
    assert reward_eval(1, 1) == (-np.exp(-1))
