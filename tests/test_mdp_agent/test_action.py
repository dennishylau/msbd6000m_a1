from mdp_agent.action import Action
from decimal import Decimal


def test_action():
    assert Action("0.12345") == Decimal("0.12")
