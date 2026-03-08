import math
from dataclasses import dataclass

from config import SimulationConfig


@dataclass
class Quote:
    reservation_price: float
    bid: float
    ask: float
    spread: float


class BaseStrategy:
    """
    Abstract base class for quoting strategies.
    """

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config

    def compute_quotes(
        self,
        midprice: float,
        inventory: int,
        time_remaining: float,
    ) -> Quote:
        raise NotImplementedError("Subclasses must implement compute_quotes().")


class FixedSpreadStrategy(BaseStrategy):
    """
    Baseline strategy that quotes symmetrically around the midprice
    using a constant spread.
    """

    def compute_quotes(
        self,
        midprice: float,
        inventory: int,
        time_remaining: float,
    ) -> Quote:
        spread = self.config.base_spread
        reservation_price = midprice
        bid = reservation_price - spread / 2.0
        ask = reservation_price + spread / 2.0

        return Quote(
            reservation_price=reservation_price,
            bid=bid,
            ask=ask,
            spread=spread,
        )


class AvellanedaStoikovStrategy(BaseStrategy):
    """
    Inventory-aware market making strategy inspired by Avellaneda-Stoikov.
    """

    def compute_quotes(
        self,
        midprice: float,
        inventory: int,
        time_remaining: float,
    ) -> Quote:
        gamma = self.config.gamma
        sigma = self.config.sigma
        k = self.config.k

        reservation_price = midprice - inventory * gamma * (sigma ** 2) * time_remaining

        # Numerical safeguard in case gamma is extremely small.
        if gamma <= 0:
            spread = self.config.base_spread
        else:
            spread = gamma * (sigma ** 2) * time_remaining + (2.0 / gamma) * math.log(1.0 + gamma / k)

        bid = reservation_price - spread / 2.0
        ask = reservation_price + spread / 2.0

        return Quote(
            reservation_price=reservation_price,
            bid=bid,
            ask=ask,
            spread=spread,
        )