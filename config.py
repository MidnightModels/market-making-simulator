from dataclasses import dataclass


@dataclass
class SimulationConfig:
    # Time/grid parameters
    T: float = 1.0                 # total simulation horizon
    dt: float = 0.01               # time step
    n_steps: int = 100             # number of simulation steps

    # Midprice process parameters
    S0: float = 100.0              # initial midprice
    mu: float = 0.0                # drift
    sigma: float = 2.0             # volatility

    # Market order arrival model
    A: float = 5.0                 # baseline arrival intensity
    k: float = 1.5                 # sensitivity of fills to quote distance

    # Market making / risk parameters
    gamma: float = 0.37             # inventory risk aversion
    inventory_limit: int = 50      # optional hard cap for inventory
    base_spread: float = 1.0       # fallback / baseline spread

    # Trade sizing
    trade_size: int = 1            # number of shares per fill

    # Randomness / reproducibility
    seed: int = 42