from dataclasses import dataclass
from random import choices


@dataclass
class RiskyAsset:
    """
    Class to express the payout of the risky asset, which follows a
    bernoulli distribution as key assumption.

    Args:
        y_a (float): possible yield for scenario A.
        y_b (float): possible yield for scenario B.
        p (float): probability of scenario A happening; thus (1-p) is the \
            chance of scenario B happening.
    """
    y_a: float
    y_b: float
    p: float

    def sample(self) -> float:
        "Sample the distribution to obtain a yield."
        population = [float(self.y_a), float(self.y_b)]
        weights = [self.p, 1-self.p]
        return choices(population, weights)[0]
