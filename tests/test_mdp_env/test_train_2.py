from mdp_agent.policy import Policy
from mdp_agent.action_space import ActionSpace
from mdp_env.risky_asset import RiskyAsset
from mdp_env.risk_free_asset import RiskFreeAsset
from mdp_env.train import train
from mdp_env.reward import reward_eval
import numpy as np
from decimal import Decimal
import pandas as pd


def test_train_2():
    "Testing main.py scenario 2."
    # Parameters Configuration
    EPOCHS = 20000
    # Total time periods
    T = 10
    # init for epsilon-greedy
    INIT_EPSILON = 0.3
    # init for learning rate
    INIT_ALPHA = 0.01
    # state space: this is dependent on time, per section 8.4 of Rao and Jelvis
    states = np.arange(0, T + 1, 1)
    # action space
    actions = ActionSpace(
        Decimal("0"),
        Decimal("1.0"),
        Decimal("0.25"))
    # policy
    policy = Policy(
        epochs=EPOCHS,
        init_epsilon=INIT_EPSILON,
        init_alpha=INIT_ALPHA,
        state_space=states,
        action_space=actions,
        training=True
    )
    # assets
    risky_asset = RiskyAsset(-0.04, -0.10, 0.5)
    risk_free_asset = RiskFreeAsset(0.04)
    # CARA Coefficient
    CARA_COEF = 1

    train(
        epochs=EPOCHS,
        T=T,
        risky_asset=risky_asset,
        risk_free_asset=risk_free_asset,
        policy=policy,
        reward_eval=reward_eval,
        cara_coef=CARA_COEF,
        early_stopping=1e-4
    )

    expected = pd.Series(
        [
            Decimal("0.00")
            for _
            in range(10)
        ] + [Decimal("0.00")]
    )

    assert all(
        policy.optimal == expected
    ), f"""
    Actual:
    {policy.optimal}

    Expected:
    {expected}
    """
