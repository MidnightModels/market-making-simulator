import math
from dataclasses import dataclass

import numpy as np

from config import SimulationConfig
from market_maker.strategy import Quote


@dataclass
class FillResult:
    bid_fill: bool
    ask_fill: bool
    bid_probability: float
    ask_probability: float
    bid_intensity: float
    ask_intensity: float


class ExponentialOrderFlow:
    """
    Models market order arrivals using exponential intensity functions:

        lambda_b = A * exp(-k * delta_b)
        lambda_a = A * exp(-k * delta_a)

    where:
        delta_b = midprice - bid
        delta_a = ask - midprice

    Fill probabilities over one time step dt are:

        p = 1 - exp(-lambda * dt)
    """

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed + 1)

    def _arrival_intensity(self, distance: float) -> float:
        """
        Compute order arrival intensity as a function of quote distance
        from the midprice.

        Distances below zero are clipped to zero for numerical stability,
        though well-formed quotes should usually not produce them.
        """
        distance = max(distance, 0.0)
        return self.config.A * math.exp(-self.config.k * distance)

    def _fill_probability(self, intensity: float) -> float:
        """
        Convert a Poisson arrival intensity into the probability
        of at least one arrival over dt.
        """
        return 1.0 - math.exp(-intensity * self.config.dt)

    def simulate_fills(self, midprice: float, quote: Quote) -> FillResult:
        """
        Simulate whether the bid and/or ask quotes are hit during this time step.
        """
        delta_b = midprice - quote.bid
        delta_a = quote.ask - midprice

        lambda_b = self._arrival_intensity(delta_b)
        lambda_a = self._arrival_intensity(delta_a)

        p_b = self._fill_probability(lambda_b)
        p_a = self._fill_probability(lambda_a)

        bid_fill = self.rng.uniform() < p_b
        ask_fill = self.rng.uniform() < p_a

        return FillResult(
            bid_fill=bid_fill,
            ask_fill=ask_fill,
            bid_probability=p_b,
            ask_probability=p_a,
            bid_intensity=lambda_b,
            ask_intensity=lambda_a,
        )