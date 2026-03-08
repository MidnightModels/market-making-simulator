import os

import matplotlib.pyplot as plt

from config import SimulationConfig
from market_maker.metrics import compute_metrics
from market_maker.order_flow import ExponentialOrderFlow
from market_maker.price_process import MidpriceProcess
from market_maker.simulator import MarketMakingSimulator
from market_maker.strategy import AvellanedaStoikovStrategy, FixedSpreadStrategy


def run_strategy(strategy, config: SimulationConfig):
    """
    Runs a single simulation using the provided strategy.
    """
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


def main() -> None:
    config = SimulationConfig()

    fixed_strategy = FixedSpreadStrategy(config)
    avellaneda_strategy = AvellanedaStoikovStrategy(config)

    fixed_result, fixed_metrics = run_strategy(fixed_strategy, config)
    avellaneda_result, avellaneda_metrics = run_strategy(avellaneda_strategy, config)

    print("\n===== STRATEGY COMPARISON =====\n")

    print("Fixed Spread Strategy")
    print(f"Terminal PnL: {fixed_metrics.terminal_pnl:.4f}")
    print(f"Mean PnL: {fixed_metrics.mean_pnl:.4f}")
    print(f"PnL Std Dev: {fixed_metrics.pnl_std:.4f}")
    print(f"Max Abs Inventory: {fixed_metrics.max_abs_inventory}")
    print(f"Inventory Std Dev: {fixed_metrics.inventory_std:.4f}")
    print(f"Total Fills: {fixed_metrics.total_fills}")
    print(f"Average Spread: {fixed_metrics.average_spread:.4f}")
    print(f"Sharpe-like Ratio: {fixed_metrics.sharpe_like_ratio:.4f}")

    print("\nAvellaneda–Stoikov Strategy")
    print(f"Terminal PnL: {avellaneda_metrics.terminal_pnl:.4f}")
    print(f"Mean PnL: {avellaneda_metrics.mean_pnl:.4f}")
    print(f"PnL Std Dev: {avellaneda_metrics.pnl_std:.4f}")
    print(f"Max Abs Inventory: {avellaneda_metrics.max_abs_inventory}")
    print(f"Inventory Std Dev: {avellaneda_metrics.inventory_std:.4f}")
    print(f"Total Fills: {avellaneda_metrics.total_fills}")
    print(f"Average Spread: {avellaneda_metrics.average_spread:.4f}")
    print(f"Sharpe-like Ratio: {avellaneda_metrics.sharpe_like_ratio:.4f}")

    os.makedirs("outputs/figures", exist_ok=True)

    # PnL comparison plot
    plt.figure(figsize=(12, 6))
    plt.plot(fixed_result.times, fixed_result.pnl, label="Fixed Spread")
    plt.plot(avellaneda_result.times, avellaneda_result.pnl, label="Avellaneda-Stoikov")
    plt.xlabel("Time")
    plt.ylabel("PnL")
    plt.title("PnL Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/figures/strategy_pnl_comparison.png")
    plt.close()

    # Inventory comparison plot
    plt.figure(figsize=(12, 6))
    plt.plot(fixed_result.times, fixed_result.inventories, label="Fixed Spread")
    plt.plot(avellaneda_result.times, avellaneda_result.inventories, label="Avellaneda-Stoikov")
    plt.xlabel("Time")
    plt.ylabel("Inventory")
    plt.title("Inventory Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/figures/strategy_inventory_comparison.png")
    plt.close()

    # Spread comparison plot
    plt.figure(figsize=(12, 6))
    plt.plot(fixed_result.times, fixed_result.spreads, label="Fixed Spread")
    plt.plot(avellaneda_result.times, avellaneda_result.spreads, label="Avellaneda-Stoikov")
    plt.xlabel("Time")
    plt.ylabel("Spread")
    plt.title("Spread Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/figures/strategy_spread_comparison.png")
    plt.close()


if __name__ == "__main__":
    main()