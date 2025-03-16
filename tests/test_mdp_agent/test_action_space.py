import pytest
from mdp_agent.action_space import ActionSpace
from decimal import Decimal


@pytest.fixture
def as_fixture():
    return ActionSpace(0, 0.2, 0.1)


def test_action_space_choices(as_fixture):
    choices = ["0.0", "0.1", "0.2"]
    choices = [Decimal(i) for i in choices]
    assert as_fixture.choices == choices


def test_action_space_random(as_fixture):
    choices = ["0.0", "0.1", "0.2"]
    choices = [Decimal(i) for i in choices]

    random_result = []
    for i in range(20):
        random_result.append(as_fixture.random())

    for choice in choices:
        assert choice in random_result


def test_action_space_len(as_fixture):
    assert len(as_fixture) == 3
