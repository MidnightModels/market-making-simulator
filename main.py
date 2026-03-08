import os

import matplotlib.pyplot as plt

from config import SimulationConfig
from market_maker.metrics import compute_metrics
from market_maker.order_flow import ExponentialOrderFlow
from market_maker.price_process import MidpriceProcess
from market_maker.simulator import MarketMakingSimulator
from market_maker.strategy import AvellanedaStoikovStrategy


def main() -> None:
    config = SimulationConfig()

    strategy = AvellanedaStoikovStrategy(config)
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

    print("Simulation completed.\n")

    print("Final state:")
    print(f"  Final midprice: {result.midprices[-1]:.4f}")
    print(f"  Final inventory: {result.inventories[-1]}")
    print(f"  Final cash: {result.cash[-1]:.4f}")
    print(f"  Final PnL: {result.pnl[-1]:.4f}")

    print("\nSummary metrics:")
    print(f"  Terminal PnL: {metrics.terminal_pnl:.4f}")
    print(f"  Mean PnL: {metrics.mean_pnl:.4f}")
    print(f"  PnL Std Dev: {metrics.pnl_std:.4f}")
    print(f"  Max Absolute Inventory: {metrics.max_abs_inventory}")
    print(f"  Mean Inventory: {metrics.mean_inventory:.4f}")
    print(f"  Inventory Std Dev: {metrics.inventory_std:.4f}")
    print(f"  Total Bid Fills: {metrics.total_bid_fills}")
    print(f"  Total Ask Fills: {metrics.total_ask_fills}")
    print(f"  Total Fills: {metrics.total_fills}")
    print(f"  Average Spread: {metrics.average_spread:.4f}")
    print(f"  Sharpe-like Ratio: {metrics.sharpe_like_ratio:.4f}")

    os.makedirs("outputs/figures", exist_ok=True)

    # Plot 1: Midprice, reservation price, bid, ask
    plt.figure(figsize=(12, 6))
    plt.plot(result.times, result.midprices, label="Midprice")
    plt.plot(result.times, result.reservation_prices, label="Reservation Price")
    plt.plot(result.times, result.bids, label="Bid")
    plt.plot(result.times, result.asks, label="Ask")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.title("Market Making Quotes Over Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/figures/quotes_over_time.png")
    plt.show()

    # Plot 2: Inventory path
    plt.figure(figsize=(12, 5))
    plt.plot(result.times, result.inventories)
    plt.xlabel("Time")
    plt.ylabel("Inventory")
    plt.title("Inventory Over Time")
    plt.tight_layout()
    plt.savefig("outputs/figures/inventory_over_time.png")
    plt.show()

    # Plot 3: PnL path
    plt.figure(figsize=(12, 5))
    plt.plot(result.times, result.pnl)
    plt.xlabel("Time")
    plt.ylabel("PnL")
    plt.title("Mark-to-Market PnL Over Time")
    plt.tight_layout()
    plt.savefig("outputs/figures/pnl_over_time.png")
    plt.show()


if __name__ == "__main__":
    main()