import heapq
import polars as pl
from random import random
from time import perf_counter

from OrderBook import OrderBook
from MarketMaker import MarketMaker
from EURUSDPriceSimulator import EURUSDPriceSimulator
from HFT import HFT
from PoissonSimulation import ArrivalIntensity, PoissonGenerator
from config import (
    FEES_TAKER_B, FEES_TAKER_C, SIMULATOR_BUFFER_B_SIZE,
    SIMULATOR_BUFFER_C_SIZE, SIMULATOR_STEP_DT, SIMULATOR_HFT_LOOKBACK_STEPS,
    SIMULATOR_HEDGE_LOOKBACK_B, SIMULATOR_HEDGE_LOOKBACK_C, 
    SIMULATOR_MM_B_LOOKBACK_STEPS, SIMULATOR_MM_C_LOOKBACK_STEPS,
    SIMULATOR_TOP_TRADES_COUNT,
    SIMULATOR_RANDOM_BUY_PROB, SIMULATOR_ORGANIC_LAMBDA_SCALE,
    SIMULATOR_DEFAULT_PHASE,
    SIMULATOR_PHASE1_ORGANIC_LAMBDA_MULTIPLIER,
    LAMBDA_A0_A, ALPHA_A, THETA_A, LAMBDA_MO_A, V_UNIT_A,
    LAMBDA_A0_B, ALPHA_B, THETA_B, LAMBDA_MO_B, V_UNIT_B,
    BACKTEST_MM_GAMMA, BACKTEST_MM_SIGMA, BACKTEST_MM_KAPPA,
    PRICE_SIM_DEFAULT_DT_SECONDS,
    HFT_PROB_OFF,
    HFT_PROB_ONE_SIDED,
    ORDERBOOK_DEFAULT_LEVELS,
)

_REPORT_DATA_SCHEMA = {
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
    "top_trades": pl.List(pl.Struct({"price": pl.Float64, "quantity": pl.Float64, "Is Ask": pl.Boolean})),
}

class MarketSimulator:
    """
    A market simulator for a single asset, with three different markets (A, B, and C),
    and a unique market maker on exchange A.

    ``phase`` controls which HFT layers run each step (see ``SIMULATOR_DEFAULT_PHASE`` in config).
    """

    def __init__(
        self,
        order_book_A: OrderBook,
        order_book_B: OrderBook,
        order_book_C: OrderBook,
        market_maker: MarketMaker,
        price_simulator: EURUSDPriceSimulator,
        hft: HFT,
        phase: int = SIMULATOR_DEFAULT_PHASE,
    ):
        """
        order_book_A: The order book for exchange A
        order_book_B: The order book for exchange B
        order_book_C: The order book for exchange C
        market_maker: The market maker for exchange A
        price_simulator: The price simulator for the base currency
        phase: Simulation regime — 1 = MM + organic only; 2 = + HFT snipe; 3 = + HFT make_market_on_A
            (tight touch quotes from delayed B/C, ``HFT_PROB_OFF`` / ``HFT_PROB_ONE_SIDED``).
        """
        if phase not in (1, 2, 3):
            raise ValueError(f"phase must be 1, 2, or 3, got {phase!r}")

        self.phase = int(phase)
        self.order_book_A = order_book_A
        self.order_books_B = [order_book_B] + [None] * (SIMULATOR_BUFFER_B_SIZE - 1) # 200ms
        self.order_books_C = [order_book_C] + [None] * (SIMULATOR_BUFFER_C_SIZE - 1) # 170ms
        self.current_idx_B = 0
        self.current_idx_C = 0
        self.market_maker = market_maker
        self.price_simulator = price_simulator
        self.hft = hft
        self.all_trades: list[dict] = []
        self._report_rows: list[dict] = []
        self._report_schema = _REPORT_DATA_SCHEMA
        self._top_trades_heap: list[tuple[float, int, dict]] = []
        self._top_trade_seq = 0
        self._single_step_count = 0
        # Counts `simulate_order_book_evolution` calls; first call logs a detailed timing breakdown.
        self._order_book_evolution_calls = 0

    def _organic_lambda_scale(self) -> float:
        """Scale A organic MO intensity vs ``order_book_A.lambda_mo`` (higher in phase 1 without HFT)."""
        if self.phase <= 1:
            return SIMULATOR_ORGANIC_LAMBDA_SCALE * SIMULATOR_PHASE1_ORGANIC_LAMBDA_MULTIPLIER
        return SIMULATOR_ORGANIC_LAMBDA_SCALE

    def _first_step_format_level(self, price, qty: float) -> str:
        if price is None:
            return "—"
        return f"{price:.6f} × {qty:.0f}"

    def _first_step_log_orderbooks(self, tag: str) -> None:
        ob_a = self.order_book_A
        ob_b = self.order_books_B[self.current_idx_B]
        ob_c = self.order_books_C[self.current_idx_C]
        ba, aa = ob_a.best_bid, ob_a.best_ask
        bb, ab = ob_b.best_bid, ob_b.best_ask
        bc, ac = ob_c.best_bid, ob_c.best_ask
        print(f"  [first step] books — {tag}")
        print(
            f"    A: bid {self._first_step_format_level(ba[0], ba[1])} | "
            f"ask {self._first_step_format_level(aa[0], aa[1])}"
        )
        print(
            f"    B: bid {self._first_step_format_level(bb[0], bb[1])} | "
            f"ask {self._first_step_format_level(ab[0], ab[1])}"
        )
        print(
            f"    C: bid {self._first_step_format_level(bc[0], bc[1])} | "
            f"ask {self._first_step_format_level(ac[0], ac[1])}"
        )

    @property
    def data(self) -> pl.DataFrame:
        if not self._report_rows:
            return pl.DataFrame(schema=self._report_schema)
        return pl.DataFrame(self._report_rows, schema=self._report_schema)

    def _register_trade(self, trade: dict) -> None:
        self.all_trades.append(trade)
        q = trade["quantity"]
        self._top_trade_seq += 1
        item = (q, self._top_trade_seq, trade)
        k = SIMULATOR_TOP_TRADES_COUNT
        h = self._top_trades_heap
        if len(h) < k:
            heapq.heappush(h, item)
        elif q > h[0][0]:
            heapq.heapreplace(h, item)

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
        bid_B_tuple = self.get_B_best_bid()
        ask_B_tuple = self.get_B_best_ask()
        bid_C_tuple = self.get_C_best_bid()
        ask_C_tuple = self.get_C_best_ask()
        
        # Skip if any exchange lacks liquidity
        if (bid_A_tuple[0] is None or ask_A_tuple[0] is None or
            bid_B_tuple[0] is None or ask_B_tuple[0] is None or
            bid_C_tuple[0] is None or ask_C_tuple[0] is None):
            return

        if not self._report_rows:
            A_bid_diff = A_ask_diff = B_bid_diff = B_ask_diff = C_bid_diff = C_ask_diff = 0.0
            A_midpoint_diff = B_midpoint_diff = C_midpoint_diff = 0.0
        else:
            prev = self._report_rows[-1]
            A_bid_diff = float(bid_A_tuple[0] - prev["best_bid_A"]["value"])
            A_ask_diff = float(ask_A_tuple[0] - prev["best_ask_A"]["value"])
            B_bid_diff = float(bid_B_tuple[0] - prev["best_bid_B"]["value"])
            B_ask_diff = float(ask_B_tuple[0] - prev["best_ask_B"]["value"])
            C_bid_diff = float(bid_C_tuple[0] - prev["best_bid_C"]["value"])
            C_ask_diff = float(ask_C_tuple[0] - prev["best_ask_C"]["value"])
            A_midpoint_diff = float(self.get_A_midpoint() - prev["midpoint_A"]["value"])
            B_midpoint_diff = float(self.get_B_midpoint() - prev["midpoint_B"]["value"])
            C_midpoint_diff = float(self.get_C_midpoint() - prev["midpoint_C"]["value"])

        new_row = {
            "best_bid_A": {"value": float(bid_A_tuple[0]), "diff": float(A_bid_diff)},
            "best_ask_A": {"value": float(ask_A_tuple[0]), "diff": float(A_ask_diff)},
            "best_bid_B": {"value": float(bid_B_tuple[0]), "diff": float(B_bid_diff)},
            "best_ask_B": {"value": float(ask_B_tuple[0]), "diff": float(B_ask_diff)},
            "best_bid_C": {"value": float(bid_C_tuple[0]), "diff": float(C_bid_diff)},
            "best_ask_C": {"value": float(ask_C_tuple[0]), "diff": float(C_ask_diff)},
            "midpoint_A": {"value": float(self.get_A_midpoint()), "diff": float(A_midpoint_diff)},
            "midpoint_B": {"value": float(self.get_B_midpoint()), "diff": float(B_midpoint_diff)},
            "midpoint_C": {"value": float(self.get_C_midpoint()), "diff": float(C_midpoint_diff)},
            "fill_rate": fill_rate,
            "top_trades": top_trades,
        }

        self._report_rows.append(new_row)

    def _print_evolve_timing_block(self, label: str, d: dict) -> None:
        ms_keys = [k for k in d if not str(k).startswith("count")]
        ct_keys = [k for k in d if str(k).startswith("count")]
        print(f"    {label} — internal breakdown:")
        for k in sorted(ms_keys):
            print(f"      {k}: {d[k]:.3f} ms")
        for k in sorted(ct_keys):
            print(f"      {k}: {d[k]}")

    def simulate_order_book_evolution(self):
        profile = self._order_book_evolution_calls == 0
        if profile:
            print(
                "======== [warmup: 1st simulate_order_book_evolution] timing breakdown ========"
            )
            print(
                f"  (each evolve: {2 * ORDERBOOK_DEFAULT_LEVELS} Poisson draws in the LO grid "
                f"— ORDERBOOK_DEFAULT_LEVELS={ORDERBOOK_DEFAULT_LEVELS})"
            )

        t0 = perf_counter()
        mid, mid_B, mid_C = self.price_simulator.next_prices()
        ms_next_prices = (perf_counter() - t0) * 1000.0

        src_B = self.order_books_B[self.current_idx_B]
        src_C = self.order_books_C[self.current_idx_C]
        n_orders_B = len(src_B._orders)
        n_orders_C = len(src_C._orders)

        t0 = perf_counter()
        new_B = src_B.copy()
        ms_copy_B = (perf_counter() - t0) * 1000.0

        timing_B: dict | None = {} if profile else None
        t0 = perf_counter()
        fills_B = new_B.evolve_one_step(mid_B, SIMULATOR_STEP_DT, _timing=timing_B)[1]
        ms_evolve_B_wall = (perf_counter() - t0) * 1000.0

        t0 = perf_counter()
        new_C = src_C.copy()
        ms_copy_C = (perf_counter() - t0) * 1000.0

        timing_C: dict | None = {} if profile else None
        t0 = perf_counter()
        fills_C = new_C.evolve_one_step(mid_C, SIMULATOR_STEP_DT, _timing=timing_C)[1]
        ms_evolve_C_wall = (perf_counter() - t0) * 1000.0

        t0 = perf_counter()
        self.order_books_B[(self.current_idx_B + 1) % SIMULATOR_BUFFER_B_SIZE] = new_B
        self.order_books_C[(self.current_idx_C + 1) % SIMULATOR_BUFFER_C_SIZE] = new_C

        all_fills = [("B", o, qty) for o, qty in fills_B] + [("C", o, qty) for o, qty in fills_C]

        for exch, o, qty in all_fills:
            self._register_trade(
                {"price": o.price, "quantity": qty, "is_ask": o.is_ask, "exchange": exch}
            )
        ms_buffer_and_trades = (perf_counter() - t0) * 1000.0

        if profile and timing_B is not None and timing_C is not None:
            ms_total = (
                ms_next_prices
                + ms_copy_B
                + ms_evolve_B_wall
                + ms_copy_C
                + ms_evolve_C_wall
                + ms_buffer_and_trades
            )
            print(f"  next_prices: {ms_next_prices:.3f} ms")
            print(f"  copy_B: {ms_copy_B:.3f} ms (source orders in _orders: {n_orders_B})")
            print(f"  evolve_B wall clock: {ms_evolve_B_wall:.3f} ms")
            self._print_evolve_timing_block("book B", timing_B)
            print(f"  copy_C: {ms_copy_C:.3f} ms (source orders in _orders: {n_orders_C})")
            print(f"  evolve_C wall clock: {ms_evolve_C_wall:.3f} ms")
            self._print_evolve_timing_block("book C", timing_C)
            print(
                f"  ring_buffer_write + register_trades ({len(all_fills)} fills): "
                f"{ms_buffer_and_trades:.3f} ms"
            )
            print(f"  TOTAL (one simulate_order_book_evolution): {ms_total:.3f} ms")
            print("======== end warmup 1st evolve profile ========\n")

        self._order_book_evolution_calls += 1

    def simulate_200ms_history(self):
        # Warm up B/C ring buffers. B has 21 slots and writes to (idx+1)%21, so slot 0 is only
        # updated after the 21st evolution; 21 steps (~210ms) ensure (current-20)%21 is never stale.
        for _ in range(SIMULATOR_BUFFER_B_SIZE):
            self.simulate_order_book_evolution()
            self.current_idx_B = (self.current_idx_B + 1) % SIMULATOR_BUFFER_B_SIZE
            self.current_idx_C = (self.current_idx_C + 1) % SIMULATOR_BUFFER_C_SIZE

    def simulate_single_step(self):
        fills = []
        is_first = self._single_step_count == 0
        if is_first:
            print("======== [first step] simulate_single_step — detailed log ========")
            print(f"  [first step] simulator phase={self.phase} "
                  f"(1=MM+organic, 2=+HFT snipe, 3=+HFT make_market_on_A)")

        def _lap_ms(label: str, t0: float) -> float:
            dt_ms = (perf_counter() - t0) * 1000.0
            if is_first:
                print(f"  [first step] timing — {label}: {dt_ms:.3f} ms")
            return perf_counter()

        # Simulate the evolution of the midprice and then reconstruct the orders
        t0 = perf_counter()
        self.simulate_order_book_evolution()
        self.current_idx_B = (self.current_idx_B + 1) % SIMULATOR_BUFFER_B_SIZE
        self.current_idx_C = (self.current_idx_C + 1) % SIMULATOR_BUFFER_C_SIZE
        # Calculer le mid instantané B/C (sans lag) — référence externe pour spread capture
        _ob_b_now = self.order_books_B[self.current_idx_B]
        _ob_c_now = self.order_books_C[self.current_idx_C]
        _mid_b_now = _ob_b_now.mid if _ob_b_now is not None and _ob_b_now.mid is not None else self.market_maker.s0
        _mid_c_now = _ob_c_now.mid if _ob_c_now is not None and _ob_c_now.mid is not None else self.market_maker.s0
        _true_mid = (
            self.market_maker.WEIGHT_B * _mid_b_now
            + self.market_maker.WEIGHT_C * _mid_c_now
        )
        t0 = _lap_ms("simulate_order_book_evolution + buffer advance", t0)
        if is_first:
            self._first_step_log_orderbooks("after order book evolution (B/C stepped)")

        hft_fills_A: list = []
        hft_snipe_count = 0
        hft_snipe_qty = 0.0
        orders_B: list = []
        orders_C: list = []

        # HFT snipes orders on A given B and C 50ms ago
        orders_A, orders_B, orders_C = self.hft.snipe(
            self.order_book_A,
            self.order_books_B[(self.current_idx_B - SIMULATOR_HFT_LOOKBACK_STEPS) % SIMULATOR_BUFFER_B_SIZE],
            self.order_books_C[(self.current_idx_C - SIMULATOR_HFT_LOOKBACK_STEPS) % SIMULATOR_BUFFER_C_SIZE]
        )

        for order_A in orders_A:
            order_fills = self.order_book_A.add_market_order_from_LO(order_A)
            hft_fills_A.extend(order_fills)
        fills.extend(hft_fills_A)
        self.market_maker.update_inventory_from_fills(hft_fills_A, mid_ref=_true_mid)

        hft_snipe_count = len(orders_A)
        hft_snipe_qty = sum(o.quantity for o in orders_A)

        for order_B in orders_B:
            self.order_books_B[self.current_idx_B].add_market_order_from_LO(order_B)
        for order_C in orders_C:
            self.order_books_C[self.current_idx_C].add_market_order_from_LO(order_C)

        t0 = _lap_ms("HFT snipe + apply orders on A/B/C", t0)
        if is_first:
            self._first_step_log_orderbooks("after HFT snipe" if self.phase >= 2 else "after HFT snipe (skipped)")
            print(
                f"  [first step] HFT snipe: {hft_snipe_count} orders on A "
                f"(total qty {hft_snipe_qty:.0f}), B/C child orders {len(orders_B)}/{len(orders_C)}"
            )

        hft_mm_fills: list = []
        if self.phase >= 3:
            # HFT make_market_on_A in phase 3 only
            ob_b_lag = self.order_books_B[
                (self.current_idx_B - SIMULATOR_HFT_LOOKBACK_STEPS) % SIMULATOR_BUFFER_B_SIZE
            ]
            ob_c_lag = self.order_books_C[
                (self.current_idx_C - SIMULATOR_HFT_LOOKBACK_STEPS) % SIMULATOR_BUFFER_C_SIZE
            ]
            hft_mm_fills = self.hft.make_market_on_A(
                self.order_book_A,
                ob_b_lag,
                ob_c_lag,
                prob_off=HFT_PROB_OFF,
                prob_one_sided=HFT_PROB_ONE_SIDED,
            )
            fills.extend(hft_mm_fills)

        t0 = _lap_ms("HFT make_market_on_A (phase 3)", t0)
        if is_first:
            self._first_step_log_orderbooks(
                "after HFT make_market_on_A" if self.phase >= 3 else "after HFT make_market_on_A (skipped)"
            )
            print(
                f"  [first step] HFT make_market_on_A: fill events={len(hft_mm_fills)} "
                f"(phase {self.phase})"
            )

        # Organic MOs on A : spot already shifted with _shift_prices
        current_spread_A = self.order_book_A.spread
        if current_spread_A is None:
            current_spread_A = 0.0001
        n_mo = PoissonGenerator(ArrivalIntensity(
            spread=current_spread_A, alpha=ALPHA_A,
            lambda_0=self.order_book_A.lambda_mo * self._organic_lambda_scale()
        )).generate()

        fills_A_organic = []
        for _ in range(n_mo):
            fills_A_organic.extend(self.order_book_A.add_market_order(random() > SIMULATOR_RANDOM_BUY_PROB, self.order_book_A.v_unit))

        fills.extend(fills_A_organic)
        self.market_maker.update_inventory_from_fills(fills_A_organic, mid_ref=_true_mid)

        t0 = _lap_ms("organic market orders on A", t0)
        if is_first:
            self._first_step_log_orderbooks("after organic MOs on A")
            print(
                f"  [first step] organic MOs: sampled count n_mo={n_mo}, "
                f"fill events on A={len(fills_A_organic)}"
            )

        # MM hedges after all arrivals (LO/MO/HFT), before quoting
        order_B, order_C = self.market_maker.check_and_hedge(
            self.order_books_B[self.current_idx_B],
            self.order_books_C[self.current_idx_C],
            self.price_simulator._t_seconds
        )
        if order_B is not None:
            order_fills_B = self.order_books_B[self.current_idx_B].add_market_order(order_B.is_ask, order_B.quantity)
            fills.extend(order_fills_B)
            self.market_maker.apply_hedge_fills(order_fills_B, order_B.is_ask, FEES_TAKER_B)
        if order_C is not None:
            order_fills_C = self.order_books_C[self.current_idx_C].add_market_order(order_C.is_ask, order_C.quantity)
            fills.extend(order_fills_C)
            self.market_maker.apply_hedge_fills(order_fills_C, order_C.is_ask, FEES_TAKER_C)

        t0 = _lap_ms("MM check_and_hedge + hedge MO execution on B/C", t0)
        if is_first:
            self._first_step_log_orderbooks("after MM hedge on B/C")
            side_b = f"{'ask' if order_B.is_ask else 'bid'} {order_B.quantity:.0f}" if order_B else "none"
            side_c = f"{'ask' if order_C.is_ask else 'bid'} {order_C.quantity:.0f}" if order_C else "none"
            print(f"  [first step] MM hedge choice: venue B → {side_b}, venue C → {side_c}")

        # Then let the market maker make the market on A
        self.market_maker.make_market(
            self.order_book_A,
            self.order_books_B[(self.current_idx_B - SIMULATOR_MM_B_LOOKBACK_STEPS) % SIMULATOR_BUFFER_B_SIZE],
            self.order_books_C[(self.current_idx_C - SIMULATOR_MM_C_LOOKBACK_STEPS) % SIMULATOR_BUFFER_C_SIZE]
        )

        t0 = _lap_ms("make_market on A", t0)
        if is_first:
            self._first_step_log_orderbooks("after make_market")
            mm = self.market_maker
            posted = getattr(mm, "_last_posted_mid", None)
            active = getattr(mm, "_active_orders", {})
            print(f"  [first step] MM make_market: last_posted_mid={posted}, active A levels={len(active)}")
            for level_key in sorted(active.keys()):
                rows = active[level_key]
                qsum = sum(q for _, q in rows)
                print(f"    {level_key}: qty_target≈{qsum:.0f} ({len(rows)} order slot(s))")

        # Accumulate all fills on A for top trades reporting
        t0 = perf_counter()
        for o, qty in fills_A_organic:
            self._register_trade(
                {"price": o.price, "quantity": qty, "is_ask": o.is_ask, "exchange": "A"}
            )
        for o, qty in hft_fills_A:
            self._register_trade(
                {"price": o.price, "quantity": qty, "is_ask": o.is_ask, "exchange": "A"}
            )
        for o, qty in hft_mm_fills:
            self._register_trade(
                {"price": o.price, "quantity": qty, "is_ask": o.is_ask, "exchange": "A"}
            )
        t0 = _lap_ms("register A-side trades for reporting", t0)

        self.save_data(
            fill_rate={"bid": 0.0, "ask": 0.0},  # placeholder — fill rate MM calculé dans save_metrics
            top_trades=self.get_top_trades(SIMULATOR_TOP_TRADES_COUNT),
        )
        self.market_maker.save_metrics(
            order_book_A=self.order_book_A,
            order_book_B=self.order_books_B[self.current_idx_B],
            order_book_C=self.order_books_C[self.current_idx_C],
            timestamp=float(self.price_simulator._t_seconds),
            fills=fills,
            hft_snipe_count=hft_snipe_count,
            hft_snipe_qty=hft_snipe_qty,
        )
        self.market_maker.advance_clock(self.price_simulator.dt_seconds)
        _lap_ms("save_data + save_metrics + advance_clock", t0)

        self._single_step_count += 1

    def simulate_n_steps(self, n_steps: int = 1):
        # The price simulator should have generated the prices for the next n_steps
        for _ in range(n_steps):
            self.simulate_single_step()
            if _ % 5000 == 0:
                print(f"======== Step {_} / {n_steps} completed =========")

    def simulate_days(self, n_days: int = 1):
        daily_steps = int(86_400 / self.price_simulator.dt_seconds)
        for i in range(n_days):
            self.price_simulator.generate_prices_for_simulation_day()
            self.simulate_n_steps(daily_steps)
            print(f"======== Day {i+1} completed =========")



    def get_top_trades(self, n: int = SIMULATOR_TOP_TRADES_COUNT) -> list[dict]:
        if n > SIMULATOR_TOP_TRADES_COUNT:
            raw_trades = heapq.nlargest(n, self.all_trades, key=lambda x: x["quantity"])
        else:
            raw_trades = [x[2] for x in sorted(self._top_trades_heap, key=lambda x: x[0], reverse=True)[:n]]
        clean_trades = []
        for t in raw_trades:
            clean_trades.append({
                "price": float(t["price"]),
                "quantity": float(t["quantity"]),
                "Is Ask": t["is_ask"]
            })
        return clean_trades


if __name__ == "__main__":
    OB_A = OrderBook(lambda_a0=LAMBDA_A0_A, alpha=ALPHA_A, theta=THETA_A, lambda_mo=LAMBDA_MO_A, v_unit=V_UNIT_A)
    OB_B = OrderBook(lambda_a0=LAMBDA_A0_B, alpha=ALPHA_B, theta=THETA_B, lambda_mo=LAMBDA_MO_B, v_unit=V_UNIT_B)
    OB_C = OrderBook(lambda_a0=LAMBDA_A0_B, alpha=ALPHA_B, theta=THETA_B, lambda_mo=LAMBDA_MO_B, v_unit=V_UNIT_B)
    MM = MarketMaker(EUR_quantity=500_000, USD_quantity=500_000, gamma=BACKTEST_MM_GAMMA, sigma=BACKTEST_MM_SIGMA, kappa=BACKTEST_MM_KAPPA, T=1, s0=1.08500)
    PS = EURUSDPriceSimulator(s0=1.15, dt_seconds=PRICE_SIM_DEFAULT_DT_SECONDS)
    HFT = HFT()
    SIM = MarketSimulator(OB_A, OB_B, OB_C, MM, PS, HFT)
    SIM.simulate_200ms_history()
    # SIM.simulate_one_day()