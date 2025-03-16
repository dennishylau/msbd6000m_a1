from dataclasses import dataclass
import numpy as np
from mdp_agent.action import Action
import random


@dataclass
class ActionSpace:
    """
    The action space indicates all different possible allocation amount of \
    risky asset.

    Args:
        min_alloc (float): define the minimum allocation.
        max_alloc (float): define the maximum allocation.
        increment (float): as action is discrete, this is the increment \
            between `min_alloc` and `max_alloc`.
    """
    min_alloc: float
    max_alloc: float
    increment: float

    @property
    def choices(self) -> list[Action]:
        "All possible allocation actions."
        actions = np.arange(
            self.min_alloc,
            self.max_alloc + self.increment,
            self.increment).tolist()
        return [round(Action(i), 2) for i in actions]

    def random(self) -> Action:
        "Get random action with uniform distribution."
        return random.choices(self.choices)[0]

    def __len__(self):
        return len(self.choices)
