import polars as pl
from polars import col as c

from OrderBook import OrderBook, Level
from MarketMaker import MarketMaker

class MarketSimulator:
    """
    A market simulator for a single asset, with three different markets (A, B, and C),
    and a unique market maker on exchange A.
    """

    def __init__(self, order_book_A: OrderBook, order_book_B: OrderBook, order_book_C: OrderBook, market_maker: MarketMaker):
        self.order_book_A = order_book_A
        self.order_book_B = order_book_B
        self.order_book_C = order_book_C
        self.market_maker = market_maker
        
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

    def get_A_best_bid(self) -> Level:
        return self.order_book_A.get_best_bid()

    def get_A_best_ask(self) -> Level:
        return self.order_book_A.get_best_ask()

    def get_B_midpoint(self) -> float:
        return self.order_book_B.get_midpoint()

    def get_B_spread(self) -> float:
        return self.order_book_B.get_spread()

    def get_B_best_bid(self) -> Level:
        return self.order_book_B.get_best_bid()

    def get_B_best_ask(self) -> Level:
        return self.order_book_B.get_best_ask()

    def get_C_midpoint(self) -> float:
        return self.order_book_C.get_midpoint()

    def get_C_spread(self) -> float:
        return self.order_book_C.get_spread()

    def get_C_best_bid(self) -> Level:
        return self.order_book_C.get_best_bid()

    def get_C_best_ask(self) -> Level:
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

    def simulate_single_step(self):
        # Either simulate orders to be executed on A, B and C
        # Or simulate the evolution of the midprice and then reconstruct 
        # the orders
        fill_rate, top_trades = self.simulate_order_book_evolution()

        # Then let the market maker make the market
        self.market_maker.make_market(self.order_book_A, self.order_book_B, self.order_book_C)

        # Finally, update the backtesting report data and market maker metrics
        self.save_data(fill_rate, top_trades)
        self.market_maker.save_metrics()

        pass

    def simulate_multiple_steps(self, steps: int):
        for _ in range(steps):
            self.simulate_single_step()