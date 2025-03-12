from dataclasses import dataclass


@dataclass
class RiskFreeAsset:
    """
    Class to express the payout of risk-free asset, which has deterministic\
        yield `r`.

    Args:
        r (float): deterministic yield.
    """
    r: float

    def sample(self) -> float:
        "Returns a deterministic yield `r`."
        return self.r
