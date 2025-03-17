from dataclasses import dataclass
from mdp_agent.action_space import ActionSpace
from mdp_agent.action import Action
import numpy as np
import pandas as pd


@dataclass
class Policy:
    """
    The policy that provides action based on state, with Q-Learning update \
        rule.

    Args:
        state_space (list): Used to generate Q-Table.
        action_space (ActionSpace): Used to generate Q-Table.
        training (bool): If `True`, decay epsilon greedy; else always greedy.
        epochs (int): Used for alpha/epsilone linear decay.
        init_epsilon (float): Initial probability of exploration when \
            `training`. Linearly decays to 0.
        init_alpha (float): Initial learning rate on TD Error. \
            Linearly decays to 0.
        gamma (float): Reward discount rate. Constant over training.
    """
    state_space: list
    action_space: ActionSpace
    training: bool
    epochs: int
    init_epsilon: float = 0.3
    init_alpha: float = 0.01
    gamma: float = 0.9

    def __post_init__(self):
        "Init the Q-Table."
        state_action_space = np.zeros(
            (len(self.state_space), len(self.action_space)))
        self.q_table = pd.DataFrame(
            state_action_space, columns=self.action_space.choices)
        self.alpha = self.init_alpha
        self.epsilon = self.init_epsilon

    def __exploit(self, state) -> Action:
        "Return the action with max value based on a state."
        return self.q_table.loc[[state]].idxmax(axis=1).tolist()[0]

    def get_action(self, state) -> Action:
        """
        When `training == True`, use epsilon-greedy algorthm for actions.
        When `training == False`, use greedy algorthm for actions.
        """
        # epsilon-greedy
        if self.training and np.random.rand() < self.init_epsilon:
            # explore with random policy
            return self.action_space.random()
        else:
            # exploitation
            return self.__exploit(state)

    def get_q_value(self, state, action) -> float:
        return self.q_table.loc[state, action]

    def get_max_q_value(self, state) -> float:
        return self.q_table.loc[state].max()

    def update(self, state, action, reward, next_state):
        "Q-Learning update rule."
        current_value = self.get_q_value(state, action)
        td_target = (
            reward
            + self.gamma * self.get_max_q_value(next_state))
        td_error = td_target - current_value
        new_value = current_value + (
            self.init_alpha * td_error)
        self.q_table.loc[state, action] = new_value
        self.decay_alpha()
        self.decay_epsilon(1.2)

    def decay_epsilon(self, factor: float = 1.0):
        """
        Linear decay of epsilon to 0.

        Args:
            factor (float): default `1.0`, meaning decay to zero when training\
            ends at `epochs / 1.0`. Increase to decay faster.
        """
        update_count = self.epochs * (len(self.state_space) - 1)
        self.epsilon -= self.init_epsilon / update_count * factor
        if self.epsilon < 0:
            self.epsilon = 0

    def decay_alpha(self, factor: float = 1.0):
        """
        Linear decay of alpha to 0.

        Args:
            factor (float): default `1.0`, meaning decay to zero when training\
            ends at `epochs / 1.0`. Increase to decay faster.
        """
        update_count = self.epochs * (len(self.state_space) - 1)
        self.alpha -= self.init_alpha / update_count * factor
        if self.alpha < 0:
            self.alpha = 0

    @property
    def optimal(self):
        "Return the optimal policy per the Q-Table."
        return self.q_table.idxmax(axis=1)

    def print(self):
        "Print the optimal policy per the Q-Table."
        optimal = self.optimal
        print(optimal)
