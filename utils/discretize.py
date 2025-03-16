from decimal import Decimal


class Discretize(Decimal):
    """
    Discretize continuous value.
    """
    def __new__(cls, value):
        "Automatically round init value to N decimal points."
        obj = super().__new__(cls, value)
        return round(obj, 2)
