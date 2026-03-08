import os
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import SimulationConfig
from market_maker.metrics import compute_metrics
from market_maker.order_flow import ExponentialOrderFlow
from market_maker.price_process import MidpriceProcess
from market_maker.simulator import MarketMakingSimulator
from market_maker.strategy import AvellanedaStoikovStrategy


def run_single_simulation(config: SimulationConfig):
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
    return metrics


def run_monte_carlo_for_gamma(
    base_config: SimulationConfig,
    gamma: float,
    n_simulations: int,
):
    terminal_pnls = []
    max_abs_inventories = []
    sharpe_like_ratios = []
    total_fills = []

    for i in range(n_simulations):
        sim_config = replace(
            base_config,
            gamma=gamma,
            seed=base_config.seed + i,
        )

        metrics = run_single_simulation(sim_config)

        terminal_pnls.append(metrics.terminal_pnl)
        max_abs_inventories.append(metrics.max_abs_inventory)
        sharpe_like_ratios.append(metrics.sharpe_like_ratio)
        total_fills.append(metrics.total_fills)

    return {
        "gamma": gamma,
        "mean_terminal_pnl": float(np.mean(terminal_pnls)),
        "std_terminal_pnl": float(np.std(terminal_pnls)),
        "mean_max_abs_inventory": float(np.mean(max_abs_inventories)),
        "mean_sharpe_like_ratio": float(np.mean(sharpe_like_ratios)),
        "mean_total_fills": float(np.mean(total_fills)),
    }


def main() -> None:
    os.makedirs("outputs/results", exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)

    base_config = SimulationConfig()

    gamma_grid = np.linspace(0.01, 0.50, 20)
    n_simulations = 200

    rows = []

    print("\nRunning optimal gamma search...\n")

    for gamma in gamma_grid:
        gamma = float(round(gamma, 4))
        summary = run_monte_carlo_for_gamma(
            base_config=base_config,
            gamma=gamma,
            n_simulations=n_simulations,
        )
        rows.append(summary)
        print(
            f"gamma={summary['gamma']:.4f} | "
            f"mean_pnl={summary['mean_terminal_pnl']:.4f} | "
            f"std_pnl={summary['std_terminal_pnl']:.4f} | "
            f"mean_max_inv={summary['mean_max_abs_inventory']:.4f} | "
            f"mean_sharpe={summary['mean_sharpe_like_ratio']:.4f}"
        )

    df = pd.DataFrame(rows)
    df.to_csv("outputs/results/optimal_gamma_search.csv", index=False)

    best_row = df.loc[df["mean_sharpe_like_ratio"].idxmax()]
    best_gamma = best_row["gamma"]
    best_sharpe = best_row["mean_sharpe_like_ratio"]
    best_pnl = best_row["mean_terminal_pnl"]
    best_inventory = best_row["mean_max_abs_inventory"]

    top_5 = df.sort_values("mean_sharpe_like_ratio", ascending=False).head(5)
    top_5.to_csv("outputs/results/optimal_gamma_top5.csv", index=False)

    print("\n===== OPTIMAL GAMMA SEARCH RESULT =====")
    print(f"Optimal gamma: {best_gamma:.4f}")
    print(f"Best mean Sharpe-like ratio: {best_sharpe:.4f}")
    print(f"Mean terminal PnL at optimum: {best_pnl:.4f}")
    print(f"Std terminal PnL at optimum: {best_row['std_terminal_pnl']:.4f}")
    print(f"Mean max absolute inventory at optimum: {best_inventory:.4f}")
    print(f"Mean total fills at optimum: {best_row['mean_total_fills']:.4f}")

    print("\nTop 5 gamma values by mean Sharpe-like ratio:")
    print(top_5.to_string(index=False))

    # Plot 1: Sharpe-like ratio vs gamma
    plt.figure(figsize=(10, 6))
    plt.plot(df["gamma"], df["mean_sharpe_like_ratio"], marker="o")
    plt.axvline(best_gamma, linestyle="--", label=f"Optimal gamma = {best_gamma:.4f}")
    plt.scatter([best_gamma], [best_sharpe], s=80, label="Best Sharpe-like point")
    plt.xlabel("Gamma")
    plt.ylabel("Mean Sharpe-like Ratio")
    plt.title("Risk-Adjusted Performance vs Gamma")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/figures/optimal_gamma_sharpe.png")
    plt.close()

    # Plot 2: Mean terminal PnL vs gamma
    plt.figure(figsize=(10, 6))
    plt.plot(df["gamma"], df["mean_terminal_pnl"], marker="o")
    plt.axvline(best_gamma, linestyle="--", label=f"Optimal gamma = {best_gamma:.4f}")
    plt.scatter([best_gamma], [best_pnl], s=80, label="PnL at optimal gamma")
    plt.xlabel("Gamma")
    plt.ylabel("Mean Terminal PnL")
    plt.title("Mean Terminal PnL vs Gamma")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/figures/optimal_gamma_pnl.png")
    plt.close()

    # Plot 3: Mean max inventory vs gamma
    plt.figure(figsize=(10, 6))
    plt.plot(df["gamma"], df["mean_max_abs_inventory"], marker="o")
    plt.axvline(best_gamma, linestyle="--", label=f"Optimal gamma = {best_gamma:.4f}")
    plt.scatter([best_gamma], [best_inventory], s=80, label="Inventory at optimal gamma")
    plt.xlabel("Gamma")
    plt.ylabel("Mean Max Absolute Inventory")
    plt.title("Inventory Risk vs Gamma")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/figures/optimal_gamma_inventory.png")
    plt.close()

    # Plot 4: Combined summary figure
    fig, axes = plt.subplots(3, 1, figsize=(10, 14), sharex=True)

    axes[0].plot(df["gamma"], df["mean_sharpe_like_ratio"], marker="o")
    axes[0].axvline(best_gamma, linestyle="--")
    axes[0].scatter([best_gamma], [best_sharpe], s=80)
    axes[0].set_ylabel("Mean Sharpe-like Ratio")
    axes[0].set_title("Optimal Gamma Calibration Summary")

    axes[1].plot(df["gamma"], df["mean_terminal_pnl"], marker="o")
    axes[1].axvline(best_gamma, linestyle="--")
    axes[1].scatter([best_gamma], [best_pnl], s=80)
    axes[1].set_ylabel("Mean Terminal PnL")

    axes[2].plot(df["gamma"], df["mean_max_abs_inventory"], marker="o")
    axes[2].axvline(best_gamma, linestyle="--")
    axes[2].scatter([best_gamma], [best_inventory], s=80)
    axes[2].set_xlabel("Gamma")
    axes[2].set_ylabel("Mean Max Abs Inventory")

    plt.tight_layout()
    plt.savefig("outputs/figures/optimal_gamma_summary.png")
    plt.close()


if __name__ == "__main__":
    main()