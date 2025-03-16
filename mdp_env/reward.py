import numpy as np


def reward_eval(wealth, cara_coef) -> float:
    "The CARA Utility Function."
    return (-np.exp(-cara_coef * wealth)) / cara_coef
