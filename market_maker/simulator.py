from dataclasses import dataclass

import numpy as np

from config import SimulationConfig
from market_maker.order_flow import ExponentialOrderFlow
from market_maker.price_process import MidpriceProcess
from market_maker.strategy import BaseStrategy


@dataclass
class SimulationResult:
    times: np.ndarray
    midprices: np.ndarray
    reservation_prices: np.ndarray
    bids: np.ndarray
    asks: np.ndarray
    spreads: np.ndarray
    inventories: np.ndarray
    cash: np.ndarray
    pnl: np.ndarray
    bid_fills: np.ndarray
    ask_fills: np.ndarray
    bid_fill_probabilities: np.ndarray
    ask_fill_probabilities: np.ndarray


class MarketMakingSimulator:
    """
    Runs an end-to-end market making simulation by combining:
    - a midprice process
    - a quoting strategy
    - an order flow model
    """

    def __init__(
        self,
        config: SimulationConfig,
        strategy: BaseStrategy,
        price_process: MidpriceProcess,
        order_flow: ExponentialOrderFlow,
    ) -> None:
        self.config = config
        self.strategy = strategy
        self.price_process = price_process
        self.order_flow = order_flow

    def run(self) -> SimulationResult:
        n = self.config.n_steps

        times = np.arange(n + 1) * self.config.dt
        midprices = np.empty(n + 1)
        reservation_prices = np.empty(n + 1)
        bids = np.empty(n + 1)
        asks = np.empty(n + 1)
        spreads = np.empty(n + 1)
        inventories = np.empty(n + 1, dtype=int)
        cash = np.empty(n + 1)
        pnl = np.empty(n + 1)
        bid_fills = np.zeros(n + 1, dtype=bool)
        ask_fills = np.zeros(n + 1, dtype=bool)
        bid_fill_probabilities = np.zeros(n + 1)
        ask_fill_probabilities = np.zeros(n + 1)

        current_midprice = self.config.S0
        current_inventory = 0
        current_cash = 0.0

        midprices[0] = current_midprice
        inventories[0] = current_inventory
        cash[0] = current_cash
        pnl[0] = current_cash + current_inventory * current_midprice

        initial_time_remaining = self.config.T
        initial_quote = self.strategy.compute_quotes(
            midprice=current_midprice,
            inventory=current_inventory,
            time_remaining=initial_time_remaining,
        )

        reservation_prices[0] = initial_quote.reservation_price
        bids[0] = initial_quote.bid
        asks[0] = initial_quote.ask
        spreads[0] = initial_quote.spread

        initial_fill_result = self.order_flow.simulate_fills(
            midprice=current_midprice,
            quote=initial_quote,
        )
        bid_fill_probabilities[0] = initial_fill_result.bid_probability
        ask_fill_probabilities[0] = initial_fill_result.ask_probability

        for t in range(1, n + 1):
            current_midprice = self.price_process.step(current_midprice)

            time_elapsed = times[t]
            time_remaining = max(self.config.T - time_elapsed, 0.0)

            quote = self.strategy.compute_quotes(
                midprice=current_midprice,
                inventory=current_inventory,
                time_remaining=time_remaining,
            )

            fill_result = self.order_flow.simulate_fills(
                midprice=current_midprice,
                quote=quote,
            )

            if fill_result.bid_fill and current_inventory < self.config.inventory_limit:
                current_inventory += self.config.trade_size
                current_cash -= quote.bid * self.config.trade_size
                bid_fills[t] = True

            if fill_result.ask_fill and current_inventory > -self.config.inventory_limit:
                current_inventory -= self.config.trade_size
                current_cash += quote.ask * self.config.trade_size
                ask_fills[t] = True

            midprices[t] = current_midprice
            reservation_prices[t] = quote.reservation_price
            bids[t] = quote.bid
            asks[t] = quote.ask
            spreads[t] = quote.spread
            inventories[t] = current_inventory
            cash[t] = current_cash
            pnl[t] = current_cash + current_inventory * current_midprice
            bid_fill_probabilities[t] = fill_result.bid_probability
            ask_fill_probabilities[t] = fill_result.ask_probability

        return SimulationResult(
            times=times,
            midprices=midprices,
            reservation_prices=reservation_prices,
            bids=bids,
            asks=asks,
            spreads=spreads,
            inventories=inventories,
            cash=cash,
            pnl=pnl,
            bid_fills=bid_fills,
            ask_fills=ask_fills,
            bid_fill_probabilities=bid_fill_probabilities,
            ask_fill_probabilities=ask_fill_probabilities,
        )