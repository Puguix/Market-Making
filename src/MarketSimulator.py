import polars as pl
from polars import col as c
import copy

from OrderBook import OrderBook, PriceLevel
from MarketMaker import MarketMaker
from EURUSDPriceSimulator import EURUSDPriceSimulator

class MarketSimulator:
    """
    A market simulator for a single asset, with three different markets (A, B, and C),
    and a unique market maker on exchange A.
    """

    def __init__(self, order_book_A: OrderBook, order_book_B: OrderBook, order_book_C: OrderBook, market_maker: MarketMaker, price_simulator: EURUSDPriceSimulator):
        """
        order_book_A: The order book for exchange A
        order_book_B: The order book for exchange B
        order_book_C: The order book for exchange C
        market_maker: The market maker for exchange A
        price_simulator: The price simulator for the base currency
        """
        
        self.order_book_A = order_book_A
        self.order_books_B = [order_book_B] + [None] * 20 # 200ms
        self.order_books_C = [order_book_C] + [None] * 17 # 170ms
        self.current_idx_B = 0
        self.current_idx_C = 0
        self.market_maker = market_maker
        self.price_simulator = price_simulator
        self.pending_orders_B = [[] for _ in range(21)]
        self.pending_orders_C = [[] for _ in range(18)]

        # Backtesting report data
        self.data = pl.DataFrame(schema={
            "best_bid_A": pl.Struct({"value": pl.Float64, "diff": pl.Float64}),
            "best_ask_A": pl.Struct({"value": pl.Float64, "diff": pl.Float64}),
            "best_bid_B": pl.Struct({"value": pl.Float64, "diff": pl.Float64}),
            "best_ask_B": pl.Struct({"value": pl.Float64, "diff": pl.Float64}),
            "best_bid_C": pl.Struct({"value": pl.Float64, "diff": pl.Float64}),
            "best_ask_C": pl.Struct({"value": pl.Float64, "diff": pl.Float64}),
            "midpoint_A": pl.Struct({"value": pl.Float64, "diff": pl.Float64}),
            "midpoint_B": pl.Struct({"value": pl.Float64, "diff": pl.Float64}),
            "midpoint_C": pl.Struct({"value": pl.Float64, "diff": pl.Float64}),
            "fill_rate": pl.Struct({"bid": pl.Float64, "ask": pl.Float64}),
            "top_trades": pl.List(pl.Struct({"price": pl.Float64, "quantity": pl.Float64, "side": pl.String}))
        })

    def get_A_midpoint(self) -> float:
        return self.order_book_A.get_midpoint()

    def get_A_spread(self) -> float:
        return self.order_book_A.get_spread()

    def get_A_best_bid(self) -> PriceLevel:
        return self.order_book_A.get_best_bid()

    def get_A_best_ask(self) -> PriceLevel:
        return self.order_book_A.get_best_ask()

    def get_B_midpoint(self) -> float:
        return self.order_book_B.get_midpoint()

    def get_B_spread(self) -> float:
        return self.order_book_B.get_spread()

    def get_B_best_bid(self) -> PriceLevel:
        return self.order_book_B.get_best_bid()

    def get_B_best_ask(self) -> PriceLevel:
        return self.order_book_B.get_best_ask()

    def get_C_midpoint(self) -> float:
        return self.order_book_C.get_midpoint()

    def get_C_spread(self) -> float:
        return self.order_book_C.get_spread()

    def get_C_best_bid(self) -> PriceLevel:
        return self.order_book_C.get_best_bid()

    def get_C_best_ask(self) -> PriceLevel:
        return self.order_book_C.get_best_ask()

    def save_data(self, fill_rate: dict[str, float], top_trades: list[dict]):
        # Update the backtesting report data
        A_bid_diff = self.get_A_best_bid().price - self.data["best_bid_A"]["value"]
        A_ask_diff = self.get_A_best_ask().price - self.data["best_ask_A"]["value"]
        B_bid_diff = self.get_B_best_bid().price - self.data["best_bid_B"]["value"]
        B_ask_diff = self.get_B_best_ask().price - self.data["best_ask_B"]["value"]
        C_bid_diff = self.get_C_best_bid().price - self.data["best_bid_C"]["value"]
        C_ask_diff = self.get_C_best_ask().price - self.data["best_ask_C"]["value"]
        A_midpoint_diff = self.get_A_midpoint() - self.data["midpoint_A"]["value"]
        B_midpoint_diff = self.get_B_midpoint() - self.data["midpoint_B"]["value"]
        C_midpoint_diff = self.get_C_midpoint() - self.data["midpoint_C"]["value"]

        self.data = self.data.extend({
                "best_bid_A": {"value": self.get_A_best_bid().price, "diff": A_bid_diff},
                "best_ask_A": {"value": self.get_A_best_ask().price, "diff": A_ask_diff},
                "best_bid_B": {"value": self.get_B_best_bid().price, "diff": B_bid_diff},
                "best_ask_B": {"value": self.get_B_best_ask().price, "diff": B_ask_diff},
                "best_bid_C": {"value": self.get_C_best_bid().price, "diff": C_bid_diff},
                "best_ask_C": {"value": self.get_C_best_ask().price, "diff": C_ask_diff},
                "midpoint_A": {"value": self.get_A_midpoint(), "diff": A_midpoint_diff},
                "midpoint_B": {"value": self.get_B_midpoint(), "diff": B_midpoint_diff},
                "midpoint_C": {"value": self.get_C_midpoint(), "diff": C_midpoint_diff},
                "fill_rate": fill_rate,
                "top_trades": top_trades,
            })

    def simulate_order_book_evolution(self):
        mid, mid_B, mid_C = self.price_simulator.next_prices()
        new_B = copy.deepcopy(self.order_books_B[self.current_idx_B]).evolve_one_step(mid_B, 0.01)
        new_C = copy.deepcopy(self.order_books_C[self.current_idx_C]).evolve_one_step(mid_C, 0.01)
        self.order_books_B[(self.current_idx_B + 1) % 21] = new_B
        self.order_books_C[(self.current_idx_C + 1) % 18] = new_C

        # TODO return fill rate and top trades
        return None, None 

    def simulate_200ms_history(self):
        # Simulate the evolution of the order books for 200ms so the main simulation can start with the history necessary for the main flow to work correctly
        for _ in range(20): # 200ms
            self.simulate_order_book_evolution()
            self.current_idx_B = (self.current_idx_B + 1) % 21
            self.current_idx_C = (self.current_idx_C + 1) % 18

    def simulate_single_step(self):
        # Simulate the evolution of the midprice and then reconstruct the orders
        fill_rate, top_trades = self.simulate_order_book_evolution()
        self.current_idx_B = (self.current_idx_B + 1) % 21
        self.current_idx_C = (self.current_idx_C + 1) % 18

        # Match pending orders
        for order in self.pending_orders_B[self.current_idx_B]:
            self.order_books_B[self.current_idx_B].add_limit_order(order)
        for order in self.pending_orders_C[self.current_idx_C]:
            self.order_books_C[self.current_idx_C].add_limit_order(order)

        # HFT snipes orders on A given B and C 50ms ago
        orders_A, orders_B, orders_C = self.hft.snipe(
            self.order_book_A, 
            self.order_books_B[(self.current_idx_B - 5) % 21],
            self.order_books_C[(self.current_idx_C - 5) % 18]
        )
        for order_A in orders_A:
            self.order_book_A.add_limit_order(order_A)
        for order_B in orders_B:
            self.order_books_B[(self.current_idx_B + 5) % 21].add_limit_order(order_B)
        for order_C in orders_C:
            self.order_books_C[(self.current_idx_C + 5) % 18].add_limit_order(order_C)

        # Then let the market maker make the market on A given B and C 200ms and 170ms ago
        self.market_maker.make_market(
            self.order_book_A, 
            self.order_books_B[(self.current_idx_B - 20) % 21], 
            self.order_books_C[(self.current_idx_C - 17) % 18]
        )

        # He then hedges himself if his inventory is too skewed
        order_B, order_C = self.market_maker.check_and_hedge(
            self.order_books_B[(self.current_idx_B - 20) % 21], 
            self.order_books_C[(self.current_idx_C - 17) % 18]
        )
        if order_B is not None:
            self.pending_orders_B[(self.current_idx_B + 20) % 21].append(order_B)
        if order_C is not None:
            self.pending_orders_C[(self.current_idx_C + 17) % 18].append(order_C)

        # Finally, update the backtesting report data and market maker metrics
        self.save_data(fill_rate, top_trades)
        self.market_maker.save_metrics()

        pass

    def simulate_multiple_steps(self, steps: int, generate_200ms_history: bool = True):
        if generate_200ms_history:
            self.simulate_200ms_history()
        for _ in range(steps):
            self.simulate_single_step()