import polars as pl
from polars import col as c
import copy
from random import random

from OrderBook import OrderBook, PriceLevel
from MarketMaker import MarketMaker
from EURUSDPriceSimulator import EURUSDPriceSimulator
from HFT import HFT
from PoissonSimulation import ArrivalIntensity, PoissonGenerator

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
        self.all_trades: list[dict] = []

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
        bid_A_tuple = self.get_A_best_bid()
        ask_A_tuple = self.get_A_best_ask()
        
        if bid_A_tuple[0] is None or ask_A_tuple[0] is None:
            return

        if len(self.data) == 0:
            A_bid_diff = A_ask_diff = B_bid_diff = B_ask_diff = C_bid_diff = C_ask_diff = 0.0
            A_midpoint_diff = B_midpoint_diff = C_midpoint_diff = 0.0
        else:
            A_bid_diff = float(self.get_A_best_bid()[0] - self.data["best_bid_A"][-1]["value"])
            A_ask_diff = float(self.get_A_best_ask()[0] - self.data["best_ask_A"][-1]["value"])
            B_bid_diff = float(self.get_B_best_bid()[0] - self.data["best_bid_B"][-1]["value"])
            B_ask_diff = float(self.get_B_best_ask()[0] - self.data["best_ask_B"][-1]["value"])
            C_bid_diff = float(self.get_C_best_bid()[0] - self.data["best_bid_C"][-1]["value"])
            C_ask_diff = float(self.get_C_best_ask()[0] - self.data["best_ask_C"][-1]["value"])
            A_midpoint_diff = float(self.get_A_midpoint() - self.data["midpoint_A"][-1]["value"])
            B_midpoint_diff = float(self.get_B_midpoint() - self.data["midpoint_B"][-1]["value"])
            C_midpoint_diff = float(self.get_C_midpoint() - self.data["midpoint_C"][-1]["value"])

        new_row = {
            "best_bid_A": {"value": float(bid_A_tuple[0]), "diff": float(A_bid_diff)},
            "best_ask_A": {"value": float(ask_A_tuple[0]), "diff": float(A_ask_diff)},
            "best_bid_B": {"value": float(self.get_B_best_bid()[0]), "diff": float(B_bid_diff)},
            "best_ask_B": {"value": float(self.get_B_best_ask()[0]), "diff": float(B_ask_diff)},
            "best_bid_C": {"value": float(self.get_C_best_bid()[0]), "diff": float(C_bid_diff)},
            "best_ask_C": {"value": float(self.get_C_best_ask()[0]), "diff": float(C_ask_diff)},
            "midpoint_A": {"value": float(self.get_A_midpoint()), "diff": float(A_midpoint_diff)},
            "midpoint_B": {"value": float(self.get_B_midpoint()), "diff": float(B_midpoint_diff)},
            "midpoint_C": {"value": float(self.get_C_midpoint()), "diff": float(C_midpoint_diff)},
            "fill_rate": fill_rate,
            "top_trades": top_trades,
        }

        self.data = self.data.vstack(pl.DataFrame([new_row], schema=self.data.schema))

    def simulate_order_book_evolution(self):
        mid, mid_B, mid_C = self.price_simulator.next_prices()
        new_B, fills_B = copy.deepcopy(self.order_books_B[self.current_idx_B]).evolve_one_step(mid_B, 0.01)
        new_C, fills_C = copy.deepcopy(self.order_books_C[self.current_idx_C]).evolve_one_step(mid_C, 0.01)

        self.order_books_B[(self.current_idx_B + 1) % 21] = new_B
        self.order_books_C[(self.current_idx_C + 1) % 18] = new_C

        all_fills = [("B", o, qty) for o, qty in fills_B] + [("C", o, qty) for o, qty in fills_C]

        # Top trades
        self.all_trades.extend(
            {"price": o.price, "quantity": qty, "side": o.side, "exchange": exch}
            for exch, o, qty in all_fills
        )


    def simulate_200ms_history(self):
        # Warm up B/C ring buffers. B has 21 slots and writes to (idx+1)%21, so slot 0 is only
        # updated after the 21st evolution; 21 steps (~210ms) ensure (current-20)%21 is never stale.
        for _ in range(21):
            self.simulate_order_book_evolution()
            self.current_idx_B = (self.current_idx_B + 1) % 21
            self.current_idx_C = (self.current_idx_C + 1) % 18

    def simulate_single_step(self):
        fills = []
        # Simulate the evolution of the midprice and then reconstruct the orders
        self.simulate_order_book_evolution()
        self.current_idx_B = (self.current_idx_B + 1) % 21
        self.current_idx_C = (self.current_idx_C + 1) % 18

        # Match and collecting fills from pending orders (hedge)
        fills_hedge = []
        for order in self.pending_orders_B[self.current_idx_B]:
            order_fills = self.order_books_B[self.current_idx_B].add_market_order(order.side, order.quantity)
            fills_hedge.extend(order_fills)
            # Mise à jour inventaire MM
            filled_qty = sum(qty for _, qty in order_fills)
            if order.side == "ask":  # MM vend EUR pour se déhedger
                self.market_maker.EUR_quantity -= filled_qty
                self.market_maker.USD_quantity += sum(o.price * qty for o, qty in order_fills)
            else:  # MM achète EUR
                self.market_maker.EUR_quantity += filled_qty
                self.market_maker.USD_quantity -= sum(o.price * qty for o, qty in order_fills)
        self.pending_orders_B[self.current_idx_B] = []

        for order in self.pending_orders_C[self.current_idx_C]:
            order_fills = self.order_books_C[self.current_idx_C].add_market_order(order.side, order.quantity)
            fills_hedge.extend(order_fills)
            filled_qty = sum(qty for _, qty in order_fills)
            if order.side == "ask":
                self.market_maker.EUR_quantity -= filled_qty
                self.market_maker.USD_quantity += sum(o.price * qty for o, qty in order_fills)
            else:
                self.market_maker.EUR_quantity += filled_qty
                self.market_maker.USD_quantity -= sum(o.price * qty for o, qty in order_fills)
        self.pending_orders_C[self.current_idx_C] = []

        fills.extend(fills_hedge)

        # HFT snipes orders on A given B and C 50ms ago
        orders_A, orders_B, orders_C = self.hft.snipe(
            self.order_book_A, 
            self.order_books_B[(self.current_idx_B - 5) % 21],
            self.order_books_C[(self.current_idx_C - 5) % 18]
        )

        # HFT fills on A
        hft_fills_A = []
        for order_A in orders_A:
            order_fills = self.order_book_A.add_limit_order(order_A)
            hft_fills_A.extend(order_fills)
        fills.extend(hft_fills_A)
        self.market_maker.update_inventory_from_fills(hft_fills_A)

        hft_snipe_count = len(orders_A)
        hft_snipe_qty = sum(o.quantity for o in orders_A)

        # HFT fills on B and C
        for order_B in orders_B:
            self.order_books_B[(self.current_idx_B + 5) % 21].add_limit_order(order_B)
        for order_C in orders_C:
            self.order_books_C[(self.current_idx_C + 5) % 18].add_limit_order(order_C)

        # Organic MOs on A : spot already shifted with _shift_prices
        # MOs generate fills on MM orders without moving the mid again.
        n_mo = PoissonGenerator(ArrivalIntensity(
            spread=0.0, alpha=0.0, 
            lambda_0=self.order_book_A.lambda_mo * 0.01
        )).generate()

        fills_A_organic = []
        for _ in range(n_mo):
            side = "bid" if random() < 0.5 else "ask"
            fills_A_organic.extend(self.order_book_A.add_market_order(side, self.order_book_A.v_unit))
        #print(f"[MOs] n_mo={n_mo}, fills={len(fills_A_organic)}")

        fills.extend(fills_A_organic)
        self.market_maker.update_inventory_from_fills(fills_A_organic)

        #print(f"[PRE-MAKE] bid_A={self.order_book_A.best_bid}, ask_A={self.order_book_A.best_ask}, fills={len(fills_A_organic)}, has_new_fills={self.market_maker._has_new_fills}, last_mid={self.market_maker._last_posted_mid}")

        # Then let the market maker make the market on A given B and C 200ms and 170ms ago
        self.market_maker.make_market(
            self.order_book_A, 
            self.order_books_B[(self.current_idx_B - 20) % 21], 
            self.order_books_C[(self.current_idx_C - 17) % 18]
        )
        #print(f"[POST-MAKE] bid_A={self.order_book_A.best_bid}, ask_A={self.order_book_A.best_ask}, active_orders={len(self.market_maker._active_orders)}")


        # Accumulate all fills on A for top trades reporting
        self.all_trades.extend(
            {"price": o.price, "quantity": qty, "side": o.side, "exchange": "A"}
            for o, qty in fills_A_organic
        )
        self.all_trades.extend(
            {"price": o.price, "quantity": qty, "side": o.side, "exchange": "A"}
            for o, qty in hft_fills_A
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

        #print(f"best_bid_A={self.order_book_A.best_bid}, best_ask_A={self.order_book_A.best_ask}")
        # Finally, update the backtesting report data and market maker metrics
        self.save_data(
            fill_rate={"bid": 0.0, "ask": 0.0},  # placeholder — fill rate MM calculé dans save_metrics
            top_trades=self.get_top_trades(10),
        )
        self.market_maker.save_metrics(
            order_book_A=self.order_book_A,
            order_book_B=self.order_books_B[(self.current_idx_B - 20) % 21],  # t-200ms
            order_book_C=self.order_books_C[(self.current_idx_C - 17) % 18],  # t-170ms
            timestamp=float(self.price_simulator._t_seconds),
            fills=fills,
            hft_snipe_count=hft_snipe_count,
            hft_snipe_qty=hft_snipe_qty,
        )

    def simulate_multiple_steps(self, steps: int, generate_200ms_history: bool = True):
        # Generate 190ms of history for B and C
        if generate_200ms_history:
            self.simulate_200ms_history()

        # Make the market on A given B and C 200ms and 170ms ago
        self.market_maker.make_market(
            self.order_book_A,
            self.order_books_B[(self.current_idx_B - 20) % 21],
            self.order_books_C[(self.current_idx_C - 17) % 18],
        )

        # Simulate the steps until the end
        for i in range(steps):
            self.simulate_single_step()
            if i % 500 == 0:
                print(f"Step {i}/{steps} ({ (i/steps)*100:.1f}%) | Inv: {self.market_maker.EUR_quantity:.0f} EUR")



    def get_top_trades(self, n: int = 10) -> list[dict]:
        raw_trades = sorted(self.all_trades, key=lambda x: x["quantity"], reverse=True)[:n]
        clean_trades = []
        for t in raw_trades:
            clean_trades.append({
                "price": float(t["price"]),
                "quantity": float(t["quantity"]),
                "side": str(t["side"])
            })
        return clean_trades


if __name__ == "__main__":
    OB_A = OrderBook(lambda_a0=50.0, alpha=0.05, theta=0.1, lambda_mo=20.0, v_unit=50000)
    OB_B = OrderBook(lambda_a0=5.0, alpha=0.05, theta=0.1, lambda_mo=5.0, v_unit=100_000)
    OB_C = OrderBook(lambda_a0=5.0, alpha=0.05, theta=0.1, lambda_mo=5.0, v_unit=100_000)
    MM = MarketMaker(EUR_quantity=500_000, USD_quantity=500_000, gamma=0.05, sigma=0.0005, kappa=100, T=1000, q_max=900_000.0, s0=1.08500)
    PS = EURUSDPriceSimulator(s0=1.15, dt_seconds=0.01)
    HFT = HFT()
    SIM = MarketSimulator(OB_A, OB_B, OB_C, MM, PS, HFT)
    SIM.simulate_multiple_steps(steps=1000, generate_200ms_history=True)