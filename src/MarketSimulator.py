import polars as pl
from polars import col as c
import copy

from OrderBook import OrderBook, PriceLevel
from MarketMaker import MarketMaker
from EURUSDPriceSimulator import EURUSDPriceSimulator
from HFT import HFT
class MarketSimulator:
    """
    A market simulator for a single asset, with three different markets (A, B, and C),
    and a unique market maker on exchange A.
    """

    def __init__(self, order_book_A: OrderBook, order_book_B: OrderBook, order_book_C: OrderBook, market_maker: MarketMaker, price_simulator: EURUSDPriceSimulator, hft: HFT):
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
        self.hft = hft

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
        return self.order_book_A.mid

    def get_A_spread(self) -> float:
        return self.order_book_A.spread

    def get_A_best_bid(self) -> tuple:
        return self.order_book_A.best_bid

    def get_A_best_ask(self) -> tuple:
        return self.order_book_A.best_ask

    def get_B_midpoint(self) -> float:
        return self.order_books_B[self.current_idx_B].mid

    def get_B_spread(self) -> float:
        return self.order_books_B[self.current_idx_B].spread

    def get_B_best_bid(self) -> tuple:
        return self.order_books_B[self.current_idx_B].best_bid

    def get_B_best_ask(self) -> tuple:
        return self.order_books_B[self.current_idx_B].best_ask

    def get_C_midpoint(self) -> float:
        return self.order_books_C[self.current_idx_C].mid

    def get_C_spread(self) -> float:
        return self.order_books_C[self.current_idx_C].spread

    def get_C_best_bid(self) -> tuple:
        return self.order_books_C[self.current_idx_C].best_bid

    def get_C_best_ask(self) -> tuple:
        return self.order_books_C[self.current_idx_C].best_ask

    def save_data(self, fill_rate: dict[str, float], top_trades: list[dict]):
        if len(self.data) == 0:
            A_bid_diff = 0.0
            A_ask_diff = 0.0
            B_bid_diff = 0.0
            B_ask_diff = 0.0
            C_bid_diff = 0.0
            C_ask_diff = 0.0
            A_midpoint_diff = 0.0
            B_midpoint_diff = 0.0
            C_midpoint_diff = 0.0
        else:
            A_bid_diff = self.get_A_best_bid()[0] - self.data["best_bid_A"][-1]["value"]
            A_ask_diff = self.get_A_best_ask()[0] - self.data["best_ask_A"][-1]["value"]
            B_bid_diff = self.get_B_best_bid()[0] - self.data["best_bid_B"][-1]["value"]
            B_ask_diff = self.get_B_best_ask()[0] - self.data["best_ask_B"][-1]["value"]
            C_bid_diff = self.get_C_best_bid()[0] - self.data["best_bid_C"][-1]["value"]
            C_ask_diff = self.get_C_best_ask()[0] - self.data["best_ask_C"][-1]["value"]
            A_midpoint_diff = self.get_A_midpoint() - self.data["midpoint_A"][-1]["value"]
            B_midpoint_diff = self.get_B_midpoint() - self.data["midpoint_B"][-1]["value"]
            C_midpoint_diff = self.get_C_midpoint() - self.data["midpoint_C"][-1]["value"]

        self.data = self.data.vstack(
            pl.DataFrame([{
                "best_bid_A": {"value": self.get_A_best_bid()[0], "diff": A_bid_diff},
                "best_ask_A": {"value": self.get_A_best_ask()[0], "diff": A_ask_diff},
                "best_bid_B": {"value": self.get_B_best_bid()[0], "diff": B_bid_diff},
                "best_ask_B": {"value": self.get_B_best_ask()[0], "diff": B_ask_diff},
                "best_bid_C": {"value": self.get_C_best_bid()[0], "diff": C_bid_diff},
                "best_ask_C": {"value": self.get_C_best_ask()[0], "diff": C_ask_diff},
                "midpoint_A": {"value": self.get_A_midpoint(), "diff": A_midpoint_diff},
                "midpoint_B": {"value": self.get_B_midpoint(), "diff": B_midpoint_diff},
                "midpoint_C": {"value": self.get_C_midpoint(), "diff": C_midpoint_diff},
                "fill_rate": fill_rate,
                "top_trades": top_trades,
            }], schema=self.data.schema)
        )

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
        fills = []
        # Simulate the evolution of the midprice and then reconstruct the orders
        fill_rate, top_trades = self.simulate_order_book_evolution()
        self.current_idx_B = (self.current_idx_B + 1) % 21
        self.current_idx_C = (self.current_idx_C + 1) % 18

        # Match and collecting fills from pending orders (hedge)
        fills_hedge = []
        for order in self.pending_orders_B[self.current_idx_B]:
            order_fills = self.order_books_B[self.current_idx_B].add_limit_order(order)
            fills_hedge.extend(order_fills)
        for order in self.pending_orders_C[self.current_idx_C]:
            order_fills = self.order_books_C[self.current_idx_C].add_limit_order(order)
            fills_hedge.extend(order_fills)

        fills.extend(fills_hedge)

        # HFT snipes orders on A given B and C 50ms ago
        orders_A, orders_B, orders_C = self.hft.snipe(
            self.order_book_A, 
            self.order_books_B[(self.current_idx_B - 5) % 21],
            self.order_books_C[(self.current_idx_C - 5) % 18]
        )

        # HFT fills on A
        for order_A in orders_A:
            order_fills = self.order_book_A.add_limit_order(order_A)
            fills.extend(order_fills)
        
        hft_snipe_count = len(orders_A)
        hft_snipe_qty = sum(o.quantity for o in orders_A)

        # HFT fills on B and C
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
            self.order_books_C[(self.current_idx_C - 17) % 18],
            self.price_simulator._t_seconds
        )
        if order_B is not None:
            self.pending_orders_B[(self.current_idx_B + 20) % 21].append(order_B)
        if order_C is not None:
            self.pending_orders_C[(self.current_idx_C + 17) % 18].append(order_C)

        # Finally, update the backtesting report data and market maker metrics
        self.save_data(fill_rate, top_trades)
        self.market_maker.save_metrics(
            order_book_A=self.order_book_A,
            order_book_B=self.order_books_B[(self.current_idx_B - 20) % 21],  # t-200ms
            order_book_C=self.order_books_C[(self.current_idx_C - 17) % 18],  # t-170ms
            timestamp=self.price_simulator._t_seconds,
            fills=fills,
            hft_snipe_count=hft_snipe_count,
            hft_snipe_qty=hft_snipe_qty,
        )

        pass

    def simulate_multiple_steps(self, steps: int, generate_200ms_history: bool = True):
        if generate_200ms_history:
            self.simulate_200ms_history()
        for _ in range(steps):
            self.simulate_single_step()