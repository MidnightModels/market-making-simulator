from dataclasses import dataclass

import numpy as np

from market_maker.simulator import SimulationResult


@dataclass
class SimulationMetrics:
    terminal_pnl: float
    mean_pnl: float
    pnl_std: float
    max_abs_inventory: int
    mean_inventory: float
    inventory_std: float
    total_bid_fills: int
    total_ask_fills: int
    total_fills: int
    average_spread: float
    sharpe_like_ratio: float


def compute_metrics(result: SimulationResult) -> SimulationMetrics:
    """
    Compute summary performance and risk metrics from a simulation result.
    """
    terminal_pnl = float(result.pnl[-1])
    mean_pnl = float(np.mean(result.pnl))
    pnl_std = float(np.std(result.pnl))

    max_abs_inventory = int(np.max(np.abs(result.inventories)))
    mean_inventory = float(np.mean(result.inventories))
    inventory_std = float(np.std(result.inventories))

    total_bid_fills = int(np.sum(result.bid_fills))
    total_ask_fills = int(np.sum(result.ask_fills))
    total_fills = total_bid_fills + total_ask_fills

    average_spread = float(np.mean(result.spreads))

    if pnl_std > 0:
        sharpe_like_ratio = terminal_pnl / pnl_std
    else:
        sharpe_like_ratio = 0.0

    return SimulationMetrics(
        terminal_pnl=terminal_pnl,
        mean_pnl=mean_pnl,
        pnl_std=pnl_std,
        max_abs_inventory=max_abs_inventory,
        mean_inventory=mean_inventory,
        inventory_std=inventory_std,
        total_bid_fills=total_bid_fills,
        total_ask_fills=total_ask_fills,
        total_fills=total_fills,
        average_spread=average_spread,
        sharpe_like_ratio=sharpe_like_ratio,
    )