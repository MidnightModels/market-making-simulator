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
    return metrics


def run_monte_carlo(strategy_class, base_config: SimulationConfig, n_simulations: int):
    terminal_pnls = []
    pnl_stds = []
    max_abs_inventories = []
    sharpe_like_ratios = []
    total_fills = []

    for i in range(n_simulations):
        sim_config = replace(base_config, seed=base_config.seed + i)
        strategy = strategy_class(sim_config)
        metrics = run_single_simulation(strategy, sim_config)

        terminal_pnls.append(metrics.terminal_pnl)
        pnl_stds.append(metrics.pnl_std)
        max_abs_inventories.append(metrics.max_abs_inventory)
        sharpe_like_ratios.append(metrics.sharpe_like_ratio)
        total_fills.append(metrics.total_fills)

    return {
        "mean_terminal_pnl": float(np.mean(terminal_pnls)),
        "std_terminal_pnl": float(np.std(terminal_pnls)),
        "mean_path_pnl_std": float(np.mean(pnl_stds)),
        "mean_max_abs_inventory": float(np.mean(max_abs_inventories)),
        "mean_sharpe_like_ratio": float(np.mean(sharpe_like_ratios)),
        "mean_total_fills": float(np.mean(total_fills)),
    }


def run_parameter_sweep(
    parameter_name: str,
    parameter_values: list,
    base_config: SimulationConfig,
    n_simulations: int = 100,
) -> pd.DataFrame:
    rows = []

    for value in parameter_values:
        config = replace(base_config, **{parameter_name: value})

        fixed_results = run_monte_carlo(FixedSpreadStrategy, config, n_simulations)
        avellaneda_results = run_monte_carlo(AvellanedaStoikovStrategy, config, n_simulations)

        rows.append(
            {
                "parameter": parameter_name,
                "value": value,
                "strategy": "FixedSpread",
                **fixed_results,
            }
        )
        rows.append(
            {
                "parameter": parameter_name,
                "value": value,
                "strategy": "AvellanedaStoikov",
                **avellaneda_results,
            }
        )

        print(f"Completed {parameter_name} = {value}")

    return pd.DataFrame(rows)


def plot_sensitivity(df: pd.DataFrame, parameter_name: str, metric_name: str, output_filename: str) -> None:
    plt.figure(figsize=(10, 6))

    for strategy_name in df["strategy"].unique():
        subset = df[df["strategy"] == strategy_name].sort_values("value")
        plt.plot(subset["value"], subset[metric_name], marker="o", label=strategy_name)

    plt.xlabel(parameter_name)
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} vs {parameter_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()


def main() -> None:
    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/results", exist_ok=True)

    base_config = SimulationConfig()
    n_simulations = 100

    parameter_grids = {
        "gamma": [0.01, 0.05, 0.1, 0.2, 0.5],
        "sigma": [1.0, 1.5, 2.0, 3.0, 4.0],
        "A": [2.0, 5.0, 8.0, 12.0],
        "k": [0.5, 1.0, 1.5, 2.0, 3.0],
        "base_spread": [0.5, 1.0, 1.5, 2.0],
    }

    all_results = []

    for parameter_name, values in parameter_grids.items():
        print(f"\nRunning sensitivity analysis for {parameter_name}...")
        df = run_parameter_sweep(
            parameter_name=parameter_name,
            parameter_values=values,
            base_config=base_config,
            n_simulations=n_simulations,
        )

        all_results.append(df)

        csv_path = f"outputs/results/sensitivity_{parameter_name}.csv"
        df.to_csv(csv_path, index=False)

        plot_sensitivity(
            df,
            parameter_name,
            "mean_terminal_pnl",
            f"outputs/figures/{parameter_name}_mean_terminal_pnl.png",
        )
        plot_sensitivity(
            df,
            parameter_name,
            "mean_max_abs_inventory",
            f"outputs/figures/{parameter_name}_mean_max_abs_inventory.png",
        )
        plot_sensitivity(
            df,
            parameter_name,
            "mean_sharpe_like_ratio",
            f"outputs/figures/{parameter_name}_mean_sharpe_like_ratio.png",
        )

    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_csv("outputs/results/sensitivity_analysis_all.csv", index=False)

    print("\nSensitivity analysis completed.")
    print("Results saved to outputs/results/")
    print("Plots saved to outputs/figures/")


if __name__ == "__main__":
    main()