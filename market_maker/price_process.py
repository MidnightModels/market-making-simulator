import numpy as np

from config import SimulationConfig


class MidpriceProcess:
    """
    Simulates the midprice using an arithmetic Brownian motion model:

        S_{t+dt} = S_t + mu * dt + sigma * sqrt(dt) * Z_t

    where Z_t ~ N(0, 1).
    """

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)

    def step(self, current_price: float) -> float:
        """
        Simulate one time step of the midprice process.
        """
        shock = self.rng.normal()
        next_price = (
            current_price
            + self.config.mu * self.config.dt
            + self.config.sigma * np.sqrt(self.config.dt) * shock
        )
        return next_price

    def simulate_path(self) -> np.ndarray:
        """
        Simulate a full midprice path of length n_steps + 1,
        including the initial price S0.
        """
        prices = np.empty(self.config.n_steps + 1)
        prices[0] = self.config.S0

        for t in range(1, self.config.n_steps + 1):
            prices[t] = self.step(prices[t - 1])

        return prices