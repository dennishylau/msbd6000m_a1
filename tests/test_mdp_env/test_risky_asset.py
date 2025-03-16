from mdp_env.risky_asset import RiskyAsset


def test_sample_scenarios():
    "Ensure that only y_a and y_b will appear in sample."
    # declare an asset with yield scenarios and probability
    y_a = 1
    y_b = 0
    p = 0.5
    ra = RiskyAsset(y_a, y_b, p)
    # declare list to contain sampling results
    res = []
    # sample N times
    for _ in range(10):
        res.append(ra.sample())
    # assert both scenarios appear in result
    assert y_a in res
    assert y_b in res


def test_sample_scenarios_deterministic():
    "Ensure that only y_a will appear in sample when p = 1."
    # declare an asset with yield scenarios and probability
    y_a = 1
    y_b = 0
    p = 1
    ra = RiskyAsset(y_a, y_b, p)
    # declare list to contain sampling results
    res = []
    # sample N times
    for _ in range(10):
        res.append(ra.sample())
    # assert both scenarios appear in result
    assert y_a in res
    assert y_b not in res
