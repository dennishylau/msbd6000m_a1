from mdp_env.risk_free_asset import RiskFreeAsset


def test_risk_free_yield():
    "Ensure that yield is deterministic for risk-free asset."
    # declare an risk-free asset with constant yield
    r = 0.1
    rfa = RiskFreeAsset(r)
    # assert constant
    assert rfa.sample() == r
