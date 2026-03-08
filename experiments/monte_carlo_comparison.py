import os
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np

from config import SimulationConfig
from market_maker.metrics import compute_metrics
from market_maker.order_flow import ExponentialOrderFlow
from market_maker.price_process import MidpriceProcess
from market_maker.simulator import MarketMakingSimulator
from market_maker.strategy import AvellanedaStoikovStrategy, FixedSpreadStrategy


def run_single_simulation(strategy, config: SimulationConfig):
    price_process = MidpriceProcess(config)
    order_flow = ExponentialOrderFlow(config)

    simulator = MarketMakingSimulator(
        config=config,
        strategy=strategy,
        price_process=price_process,
        order_flow=order_flow,
    )

    result = simulator.run()
    metrics = compute_metrics(result)
    return result, metrics


def run_monte_carlo(strategy_class, base_config: SimulationConfig, n_simulations: int):
    terminal_pnls = []
    max_abs_inventories = []
    sharpe_like_ratios = []
    total_fills = []

    for i in range(n_simulations):
        sim_config = replace(base_config, seed=base_config.seed + i)

        strategy = strategy_class(sim_config)
        _, metrics = run_single_simulation(strategy, sim_config)

        terminal_pnls.append(metrics.terminal_pnl)
        max_abs_inventories.append(metrics.max_abs_inventory)
        sharpe_like_ratios.append(metrics.sharpe_like_ratio)
        total_fills.append(metrics.total_fills)

    return {
        "terminal_pnls": np.array(terminal_pnls),
        "max_abs_inventories": np.array(max_abs_inventories),
        "sharpe_like_ratios": np.array(sharpe_like_ratios),
        "total_fills": np.array(total_fills),
    }


def summarize_results(name: str, results: dict) -> None:
    print(f"\n{name}")
    print("-" * len(name))
    print(f"Mean Terminal PnL: {results['terminal_pnls'].mean():.4f}")
    print(f"Std Terminal PnL: {results['terminal_pnls'].std():.4f}")
    print(f"Mean Max Abs Inventory: {results['max_abs_inventories'].mean():.4f}")
    print(f"Mean Sharpe-like Ratio: {results['sharpe_like_ratios'].mean():.4f}")
    print(f"Mean Total Fills: {results['total_fills'].mean():.4f}")


def main() -> None:
    base_config = SimulationConfig()
    n_simulations = 200

    fixed_results = run_monte_carlo(FixedSpreadStrategy, base_config, n_simulations)
    avellaneda_results = run_monte_carlo(AvellanedaStoikovStrategy, base_config, n_simulations)

    print(f"\n===== MONTE CARLO COMPARISON ({n_simulations} simulations) =====")
    summarize_results("Fixed Spread Strategy", fixed_results)
    summarize_results("Avellaneda-Stoikov Strategy", avellaneda_results)

    os.makedirs("outputs/figures", exist_ok=True)

    # Histogram of terminal PnL
    plt.figure(figsize=(12, 6))
    plt.hist(fixed_results["terminal_pnls"], bins=30, alpha=0.6, label="Fixed Spread")
    plt.hist(avellaneda_results["terminal_pnls"], bins=30, alpha=0.6, label="Avellaneda-Stoikov")
    plt.xlabel("Terminal PnL")
    plt.ylabel("Frequency")
    plt.title("Terminal PnL Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/figures/monte_carlo_terminal_pnl_distribution.png")
    plt.close()

    # Histogram of max absolute inventory
    plt.figure(figsize=(12, 6))
    plt.hist(fixed_results["max_abs_inventories"], bins=20, alpha=0.6, label="Fixed Spread")
    plt.hist(avellaneda_results["max_abs_inventories"], bins=20, alpha=0.6, label="Avellaneda-Stoikov")
    plt.xlabel("Max Absolute Inventory")
    plt.ylabel("Frequency")
    plt.title("Max Absolute Inventory Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/figures/monte_carlo_inventory_distribution.png")
    plt.close()

    # Boxplot of terminal PnL
    plt.figure(figsize=(10, 6))
    plt.boxplot(
        [fixed_results["terminal_pnls"], avellaneda_results["terminal_pnls"]],
        tick_labels=["Fixed Spread", "Avellaneda-Stoikov"],
    )
    plt.ylabel("Terminal PnL")
    plt.title("Terminal PnL Boxplot Comparison")
    plt.tight_layout()
    plt.savefig("outputs/figures/monte_carlo_terminal_pnl_boxplot.png")
    plt.close()


if __name__ == "__main__":
    main()