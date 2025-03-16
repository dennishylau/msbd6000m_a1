# %%
from mdp_agent.action import Action
from mdp_agent.policy import Policy
from mdp_agent.action_space import ActionSpace
# from mdp_agent.policy import Policy
from mdp_env.risky_asset import RiskyAsset
from mdp_env.risk_free_asset import RiskFreeAsset
import numpy as np
from tqdm import tqdm
from IPython.core.display_functions import display


EPOCHS = 10000
# Total periods
T = 10
# epsilon
INIT_EPSILON = 0.3
INIT_ALPHA = 0.01
# action space
action_space = ActionSpace(0, 3, 0.1)
# policy
# policy = Policy(0.2, action_space, training=True)
risky_asset = RiskyAsset(0.15, 0.1, 0.5)
risk_free_asset = RiskFreeAsset(0.07)


states = np.arange(0, T + 1, 1)
actions = ActionSpace(0, 1.0, 0.2)

policy = Policy(
    epochs=EPOCHS,
    init_epsilon=INIT_EPSILON,
    init_alpha=INIT_ALPHA,
    state_space=states,
    action_space=actions,
    training=True
)

cara_coef = 1


def reward_eval(wealth) -> float:
    return (-np.exp(-cara_coef*wealth)) / cara_coef


print(
    """
------------------------------------------------
|                                              |
|                   New Epoch                  |
|                                              |
------------------------------------------------
"""
)

for epoch in tqdm(range(EPOCHS)):

    # wealth trajectory
    wealth: list[float] = [1.0]
    # action trajectory
    actions: list[Action] = []
    # risky trajectory
    risky_returns: list[float] = []
    state = 0

    for t in range(T):
        # action is the risky asset allocation in percentage
        action = policy.get_action(state=state)
        actions.append(action)
        wealth_current = wealth[-1]
        risky_return = risky_asset.sample()
        risky_returns.append(risky_return)
        wealth_next = wealth_current * (
            float(action) * (1 + risky_return)
            + (1.0 - float(action)) * (1 + risk_free_asset.sample())
        )
        wealth.append(wealth_next)

        next_state = state + 1
        # testing
        reward = reward_eval(wealth_next)

        # works
        # reward = wealth_next

        # doesn't work
        # if next_state == T:
        #     reward = terminal_reward(wealth_next)
        # else:
        #     reward = 0
        policy.update(state, action, reward, next_state)
        state = next_state

        # if next_state == T:
        # print(f"Epoch: {epoch}")
        # print(f"Wealth Trajectory: {[f"{i:.2f}" for i in wealth]}")
        # print(f"Action Trajectory: {[f"{i:.2f}" for i in actions]}")
        # print(f"Risky  Trajectory: {[f"{i:.2f}" for i in risky_returns]}")
        # print(f"Reward: {reward}")

    if epoch % 1000 == 0:
        print(f"Epoch: {epoch}, Alpha: {policy.alpha}, Epsilon: {policy.epsilon}")
        display(policy.q_table.idxmax(axis=1))

display(policy.q_table)

display(policy.q_table.idxmax(axis=1))

# %%
