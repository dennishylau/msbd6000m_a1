from typing import Callable
from tqdm import tqdm
from mdp_agent.policy import Policy
from mdp_env.risky_asset import RiskyAsset
from mdp_env.risk_free_asset import RiskFreeAsset
import pandas as pd
import numpy as np


def train(
    *,
    epochs: int,
    T: int,
    risky_asset: RiskyAsset,
    risk_free_asset: RiskFreeAsset,
    policy: Policy,
    reward_eval: Callable[[float, float], float],
    cara_coef: float,
    early_stopping: float = 2e-4
):

    prev_q_table: pd.DataFrame | None = None
    max_delta_hist: list[float] = []
    max_delta_mean: float | None = None

    for epoch in tqdm(range(epochs)):

        # init state
        state = 0
        # wealth trajectory
        wealth: list[float] = [1.0]

        for t in range(T):
            # action is the risky asset allocation in percentage
            action = policy.get_action(state=state)
            # get latest wealth
            wealth_current = wealth[-1]
            # sample the risky asset return distribution
            risky_return = risky_asset.sample()
            # calculate wealth after action
            wealth_next = wealth_current * (
                float(action) * (1 + risky_return)
                + (1.0 - float(action)) * (1 + risk_free_asset.sample())
            )
            wealth.append(wealth_next)
            # prepare transition to next state
            next_state = state + 1
            # calculate reward
            reward = reward_eval(wealth_next, cara_coef)
            # state-action value update
            policy.update(state, action, reward, next_state)
            # transition to next state
            state = next_state

        # check if q_table has converged
        if prev_q_table is not None:
            # calculate maximum delta in one epoch across all state-action
            # values
            max_delta = (policy.q_table - prev_q_table).abs().max().max()
            max_delta_hist.append(max_delta)
            # smoothen delta to avoid noise and incorrect early stopping
            max_delta_mean = np.mean(max_delta_hist[-10:])
            # early stopping when delta is small
            if max_delta_mean < early_stopping:
                print(f"Converged. Total Epochs: {epoch}.")
                print(f"Last 10 max delta mean: {max_delta_mean:.6f}.")
                break
        # update the previous q-table
        prev_q_table = policy.q_table.copy()

        # print for progress monitoring
        if epoch % 1000 == 0:
            print(f"Epoch: {epoch}")
            print(f"Alpha: {policy.alpha:.6f}, Epsilon: {policy.epsilon:.6f}.")
            if max_delta_mean:
                print(f"Last 10 max delta mean: {max_delta_mean:.6f}.")
