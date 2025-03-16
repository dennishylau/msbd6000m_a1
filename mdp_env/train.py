from typing import Callable
from tqdm import tqdm
from mdp_agent.action import Action
from mdp_agent.policy import Policy
from mdp_env.risky_asset import RiskyAsset
from mdp_env.risk_free_asset import RiskFreeAsset
from IPython.core.display_functions import display


def train(
    *,
    epochs: int,
    T: int,
    risky_asset: RiskyAsset,
    risk_free_asset: RiskFreeAsset,
    policy: Policy,
    reward_eval: Callable[[float, float], float],
    cara_coef: float
):

    for epoch in tqdm(range(epochs)):

        # init state
        state = 0
        # wealth trajectory
        wealth: list[float] = [1.0]
        # action trajectory
        actions: list[Action] = []
        # risky trajectory
        risky_returns: list[float] = []

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
            reward = reward_eval(wealth_next, cara_coef)

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
            #     print(f"Epoch: {epoch}")
            #     print(f"Wealth Trajectory: {[f"{i:.2f}" for i in wealth]}")
            #     print(f"Action Trajectory: {[f"{i:.2f}" for i in actions]}")
            #     print(f"Risky  Trajectory: {[f"{i:.2f}" for i in risky_returns]}")
            #     print(f"Reward: {reward}")

        if epoch % 1000 == 0:
            print(f"Epoch: {epoch}, Alpha: {policy.alpha}, Epsilon: {policy.epsilon}")
            display(policy.q_table.idxmax(axis=1))
