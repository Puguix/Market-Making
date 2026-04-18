import polars as pl
from polars import col as c
from OrderBook import OrderBook, Order, PriceLevel
import math
import random
from abc import ABC, abstractmethod
import time
import os
from typing import Optional
import warnings

from HFT import HFT, HFT_MM_ASK_ORDER_ID, HFT_MM_BID_ORDER_ID

from config import (
    FEES_TAKER_B, FEES_TAKER_C, PARQUET_PATH_REALTIME, PARQUET_PATH_AGGREGATED, DEFAULT_MAX_LEVELS,
    DEFAULT_TICK_SIZE, DEFAULT_GEO_INCREMENT, DEFAULT_GEO_QTY_ALPHA,
    MIN_REMAINING_TIME, PIPS_TO_PRICE_SCALE, DEFAULT_FEES_PIPS,
    DEFAULT_INITIAL_CAPITAL, MARKET_MAKER_WEIGHT_B, MARKET_MAKER_WEIGHT_C,
    MARKET_MAKER_LATENCY_B, MARKET_MAKER_LATENCY_C, MARKET_MAKER_DELTA_TAU,
    MARKET_MAKER_HEDGE_THRESHOLD, MARKET_MAKER_EPSILON,
    MARKET_MAKER_AGGREGATION_STEPS, INVENTORY_REGIME_NORMAL_THRESHOLD,
    INVENTORY_REGIME_ALERT_THRESHOLD,
    MARKET_MAKER_PHASE3_HEDGE_LEG_TRIGGER,
    MM_PHASE3_LATENCY_SAFETY_BUFFER_PIPS,
    MM_PHASE3_DEEPEN_BASE_TICKS,
    MM_PHASE3_TICK,
    MM_PHASE3_HFT_QTY_REFERENCE,
    MM_PHASE3_MAX_EXTRA_TICKS_FROM_HFT,
    MM_PHASE3_FALLBACK_SPREAD_MULTIPLIER,
    MM_PHASE3_HFT_MIN_TOTAL_QTY,
    MM_PHASE3_FALLBACK_DEPTH_PULL_TICKS,
    MM_PHASE3_FALLBACK_BUFFER_SCALE,
    MM_PHASE3_EMA_SPREAD_ALPHA,
    MM_PHASE3_INVENTORY_RAMP_START_LEG,
    MM_PHASE3_INVENTORY_RAMP_END_LEG,
    MM_PHASE3_INVENTORY_SKEW_STRENGTH,
    MM_PHASE3_SIDE_DEPTH_INV_SCALE,
)

_METRICS_RT_SCHEMA = {
    "timestamp": pl.Float64,
    "mid_A": pl.Float64,
    "mid_ref": pl.Float64,
    "EUR_quantity": pl.Float64,
    "USD_quantity": pl.Float64,
    "inventory_pct": pl.Float64,
    "hedge_regime": pl.String,
    "mtm_pnl": pl.Float64,
}

_METRICS_AGG_SCHEMA = {
    "timestamp": pl.Float64,
    "mid_A": pl.Float64,
    "mid_ref": pl.Float64,
    "reservation_price": pl.Float64,
    "optimal_spread": pl.Float64,
    "spread_quoted": pl.Float64,
    "best_bid_A": pl.Float64,
    "best_ask_A": pl.Float64,
    "EUR_quantity": pl.Float64,
    "USD_quantity": pl.Float64,
    "inventory_pct": pl.Float64,
    "hedge_regime": pl.String,
    "mtm_pnl": pl.Float64,
    "realized_pnl": pl.Float64,
    "hedge_cost": pl.Float64,
    "fill_rate_bid": pl.Float64,
    "fill_rate_ask": pl.Float64,
    "spread_capture": pl.Float64,
    "adverse_selection": pl.Float64,
    "hft_snipe_count": pl.Int32,
    "hft_snipe_qty": pl.Float64,
    "arb_opportunity_count": pl.Int32,
    "arb_opportunity_size": pl.Float64,
}

# %%%%%% Price Grid Methods %%%%%%

class PriceGridStrategy(ABC):

    @abstractmethod
    def generate(self, problem: "UtilityProblem") -> tuple[list[float], list[float]]:
        pass


class NaivePriceGridStrategy(PriceGridStrategy):

    def __init__(self, max_levels: int = DEFAULT_MAX_LEVELS, tick_size: float = DEFAULT_TICK_SIZE):
        self.max_levels = max_levels
        self.tick_size = tick_size

    def generate(self, problem: "UtilityProblem") -> tuple[list[float], list[float]]:

        ref_bid = problem.best_bid
        ref_ask = problem.best_ask
        bids = [ref_bid - i * self.tick_size for i in range(0, self.max_levels)]
        asks = [ref_ask + i * self.tick_size for i in range(0, self.max_levels)]
        return (bids, asks)
    

class GeometricPriceGridStrategy(PriceGridStrategy):

    def __init__(self, 
                 delta_grid: float = DEFAULT_TICK_SIZE,
                 geo_increment:float = DEFAULT_GEO_INCREMENT,
                 max_levels: int = DEFAULT_MAX_LEVELS,
                 tick_size: float = DEFAULT_TICK_SIZE
                 ):
        self.delta_grid = delta_grid
        self.geo_increment = geo_increment
        self.max_levels = max_levels
        self.tick_size = tick_size

    def generate(self, problem: "UtilityProblem") -> tuple[list[float], list[float]]:

        ref_bid = round(problem.best_bid, 4)
        ref_ask = round(problem.best_ask, 4)
        increment = 0 # starting at best bid / best ask
        bids, asks = [], []
        
        for i in range(0, self.max_levels):
            bids.append(round(ref_bid - increment,4)) # round up to the closest tick
            asks.append(round(ref_ask + increment, 4)) # round up to the closest tick
            increment += self.delta_grid * (self.geo_increment ** i)

        return (bids, asks)
    

# %%%%%% Qunatity Grid Methods %%%%%%

class QuantityGridStrategy(ABC):

    @abstractmethod
    def generate(self, problem: "UtilityProblem") -> tuple[list[float], list[float]]:
        pass


class NaiveQuantityGridStrategy(QuantityGridStrategy):

    def __init__(self, max_levels: int = DEFAULT_MAX_LEVELS):
        self.max_levels = max_levels

    def generate(self, problem: "UtilityProblem") -> tuple[list[float], list[float]]:

        qty_per_level = int(problem.inventory_max / self.max_levels)
        bids = [qty_per_level] * self.max_levels
        asks = [qty_per_level] * self.max_levels
        return (bids, asks)


class GeometricQuantityGridStrategy(QuantityGridStrategy):

    def __init__(self, alpha: float = DEFAULT_GEO_QTY_ALPHA, max_levels: int = DEFAULT_MAX_LEVELS):
        self.alpha = alpha
        self.max_levels = max_levels

    def generate(self, problem: "UtilityProblem") -> tuple[list[float], list[float]]:

        bids, asks = [], []

        # inventory_max is in USD notional; inventory is EUR (base). Compare in same units.
        inv_notional_usd = abs(problem.inventory) * problem.ref_price
        headroom = max(0.0, problem.inventory_max - inv_notional_usd)
        normalization = (1 - self.alpha) / (1 - (self.alpha ** self.max_levels))

        for k in range(0, int(self.max_levels)):
            qty = round(headroom * (self.alpha ** k) * normalization, 0) # round up to the closest integer
            bids.append(qty)
            asks.append(qty)

        return (bids, asks)
    

# %%%%%% Utility Problem %%%%%%

class UtilityProblem:

    def __init__(self,
                 gamma: float,        
                 sigma: float,
                 remaining_time: float,
                 inventory: float,
                 ref_price: float,
                 kappa: float,
                 latency: float,
                 inventory_max: float,
                 kapital: float = DEFAULT_INITIAL_CAPITAL,
                 fees_pips: float = DEFAULT_FEES_PIPS,
                 price_grid_strategy: PriceGridStrategy = NaivePriceGridStrategy(),
                 quantity_grid_strategy: QuantityGridStrategy = NaiveQuantityGridStrategy(),
                 spread_safety_addon: float = 0.0,
                 quote_depth_bid_delta: float = 0.0,
                 quote_depth_ask_delta: float = 0.0,
                 ):
        
        self.gamma = gamma
        self.sigma = sigma
        self.T_minus_t = self.T_minus_t = max(remaining_time, MIN_REMAINING_TIME) # clamp
        self.inventory = inventory
        self.ref_price = ref_price
        self.kappa = kappa
        self.latency = latency
        self.fees_pips = fees_pips
        self.kapital = kapital
        self.inventory_max = inventory_max
        self.spread_safety_addon = float(spread_safety_addon)
        self.quote_depth_bid_delta = float(quote_depth_bid_delta)
        self.quote_depth_ask_delta = float(quote_depth_ask_delta)

        # price and quantity grids
        self.price_grid_strategy = price_grid_strategy
        self.quantity_grid_strategy = quantity_grid_strategy

    # === Properties ===
    @property
    def reservation_price(self)->float:
        return self.ref_price - self.inventory * self.gamma * (self.sigma**2) * self.T_minus_t

    @property
    def psi_Avellaneda_Stoikov(self) -> float:
        return self.gamma * (self.sigma**2) * self.T_minus_t + (2/self.gamma) * math.log(1 + self.gamma/self.kappa)

    @property
    def psi_snipe(self) -> float:
        return 2 * self.sigma * math.sqrt(self.latency)

    @property
    def psi_fees(self) -> float:
        return self.fees_pips / PIPS_TO_PRICE_SCALE

    @property
    def optimal_spread(self) -> float:
        return (
            self.psi_Avellaneda_Stoikov
            + self.psi_snipe
            + self.psi_fees
            + self.spread_safety_addon
        )

    @property
    def best_ask(self) -> float:
        return self.reservation_price + self.optimal_spread / 2 + self.quote_depth_ask_delta

    @property
    def best_bid(self) -> float:
        return self.reservation_price - self.optimal_spread / 2 - self.quote_depth_bid_delta

    # === Methods ===
    def get_price_grid(self) -> tuple[list[float], list[float]]:

        return self.price_grid_strategy.generate(
            self
        )
    
    def get_qty_grid(self) -> tuple[list[float], list[float]]:
        
        return self.quantity_grid_strategy.generate(
            self
        )


# %%%%%% Market Maker Class %%%%%%

class MarketMaker:
    """
    The market maker for exchange A.
    Keep trak of inventory and PnL.
    """

    WEIGHT_B = MARKET_MAKER_WEIGHT_B
    WEIGHT_C = MARKET_MAKER_WEIGHT_C
    LATENCY_B = MARKET_MAKER_LATENCY_B
    LATENCY_C = MARKET_MAKER_LATENCY_C
    DELTA_TAU = MARKET_MAKER_DELTA_TAU

    def __init__(self, 
                 EUR_quantity: float, 
                 USD_quantity: float,
                 gamma: float,
                 sigma: float,
                 kappa: float,
                 T: float,
                 s0: float
                 ):
        self.s0 = s0
        self.EUR_quantity = EUR_quantity
        self.USD_quantity = USD_quantity
        self.EUR_quantity_t0 = EUR_quantity
        self.USD_quantity_t0 = USD_quantity
        self.gamma = gamma
        self.sigma = sigma
        self.kappa = kappa
        self.T = T
        self.hedge_threshold = MARKET_MAKER_HEDGE_THRESHOLD  # legacy cap (metrics / regimes)
        self.hedge_leg_trigger = min(
            MARKET_MAKER_HEDGE_THRESHOLD, MARKET_MAKER_PHASE3_HEDGE_LEG_TRIGGER
        )
        self._step_count = 0
        self.id_cpt = 0
        self._initial_capital = self.EUR_quantity_t0 * self.s0 + self.USD_quantity_t0

        # Accumulators
        self._realized_pnl = 0.0
        self._hedge_cost = 0.0

        # Orders
        self._active_orders: dict[str, list[tuple[str, float]]] = {}  # "bid_1.0850" -> [("MM_42", 300_000), ("MM_87", 150_000)]
        self._last_posted_mid: Optional[float] = None
        self._has_new_fills: bool = False
        self._ema_spread_A: Optional[float] = None
        self.epsilon: float = MARKET_MAKER_EPSILON  # 0.5 pip
        # Elapsed strategy time (seconds); drives (T - t) in Avellaneda–Stoikov.
        self._t = 0.0

        self._metrics_rt_rows: list[dict] = []
        self._metrics_agg_rows: list[dict] = []

    def advance_clock(self, dt: float) -> None:
        """Advance internal clock by one simulation step (must match price simulator dt)."""
        self._t += float(dt)

    def apply_hedge_fills(self, fills: list, aggressor_is_ask: bool, taker_fee: float) -> None:
        """
        Apply executed hedge market orders on B or C (taker).
        aggressor_is_ask=True: MM sells EUR (hits bids). False: MM buys EUR (hits asks).
        """
        if not fills:
            return
        filled_qty = sum(qty for _, qty in fills)
        if filled_qty <= 0:
            return
        if aggressor_is_ask:
            self.EUR_quantity -= filled_qty
            self.USD_quantity += sum(o.price * qty * (1.0 - taker_fee) for o, qty in fills)
        else:
            self.EUR_quantity += filled_qty
            self.USD_quantity -= sum(o.price * qty * (1.0 + taker_fee) for o, qty in fills)

    # === protected methods ===
    
    def _ref_price(self, mid_B: float, mid_C: float) -> float:
        if mid_B is None and mid_C is not None:
            return mid_C
        elif mid_B is not None and mid_C is None:
            return mid_B
        if mid_B is None and mid_C is None:
            raise ValueError("Both mid_B and mid_C cannot be None for reference price calculation.")
        return self.WEIGHT_B * mid_B + self.WEIGHT_C * mid_C
    
    def _build_utility_problem(
        self,
        mid_B: float,
        mid_C: float,
        *,
        spread_safety_addon: float = 0.0,
        quote_depth_bid_delta: float = 0.0,
        quote_depth_ask_delta: float = 0.0,
        inventory_skew_multiplier: float = 1.0,
    ) -> UtilityProblem:

        # compute inventory max
        _ref_price = self._ref_price(mid_B, mid_C)
        _inventory_max = (self.EUR_quantity * _ref_price + self.USD_quantity) * self.hedge_threshold
        # A–S inventory must be excess vs target allocation, not raw EUR balance (else a 50/50
        # book reads as ~400k long and drags reservation ~0.27 away from ref at typical T).
        v_tot = self.EUR_quantity * _ref_price + self.USD_quantity
        target_eur = (0.5 * v_tot) / _ref_price
        inventory_skew = (self.EUR_quantity - target_eur) * inventory_skew_multiplier

        return UtilityProblem(
            gamma=self.gamma,
            sigma=self.sigma,
            remaining_time=max(self.T - self._t, MIN_REMAINING_TIME),
            inventory=inventory_skew,
            ref_price=_ref_price,
            kappa=self.kappa,
            latency=self.DELTA_TAU,
            inventory_max=_inventory_max,
            price_grid_strategy=GeometricPriceGridStrategy(),
            quantity_grid_strategy=GeometricQuantityGridStrategy(),
            spread_safety_addon=spread_safety_addon,
            quote_depth_bid_delta=quote_depth_bid_delta,
            quote_depth_ask_delta=quote_depth_ask_delta,
        )

    def _flatten_mm_order_ids(self) -> list[str]:
        ids: list[str] = []
        for _key, rows in self._active_orders.items():
            for oid, _ in rows:
                ids.append(oid)
        return ids

    def _phase3_inventory_pressure(self, leg_share: float) -> float:
        lo = MM_PHASE3_INVENTORY_RAMP_START_LEG
        hi = MM_PHASE3_INVENTORY_RAMP_END_LEG
        if leg_share <= lo:
            return 0.0
        return max(0.0, min(1.0, (leg_share - lo) / max(hi - lo, 1e-12)))

    def plan_phase3_quote_actions(
        self,
        order_book_A: OrderBook,
        order_book_B: OrderBook,
        order_book_C: OrderBook,
    ) -> tuple[list[str], list[Order]]:
        """
        Phase 3 quoting: deeper peg behind HFT size, latency safety on spread, inventory skew,
        and fallback tightening when A widens or HFT size drops.

        Returns
        -------
        order_ids_to_cancel : list[str]
            All MM resting order ids that should be removed before posting ``orders_to_submit``.
        orders_to_submit : list[Order]
            New MM limit orders (caller assigns execution on ``order_book_A``).
        """
        if order_book_B.mid is None or order_book_C.mid is None:
            return self._flatten_mm_order_ids(), []

        mid_ref = self._ref_price(order_book_B.mid, order_book_C.mid)
        v_tot = self.EUR_quantity * mid_ref + self.USD_quantity
        eur_value = self.EUR_quantity * mid_ref
        usd_value = self.USD_quantity
        leg_share = max(eur_value, usd_value) / v_tot if v_tot > 0 else 0.0
        inv_pressure = self._phase3_inventory_pressure(leg_share)
        inv_skew_mult = 1.0 + MM_PHASE3_INVENTORY_SKEW_STRENGTH * inv_pressure

        target_eur = (0.5 * v_tot) / mid_ref if mid_ref else 0.0
        long_eur = self.EUR_quantity > target_eur

        spread_A = order_book_A.spread
        if spread_A is not None and spread_A > 0:
            a = MM_PHASE3_EMA_SPREAD_ALPHA
            if self._ema_spread_A is None:
                self._ema_spread_A = float(spread_A)
            else:
                self._ema_spread_A = (1.0 - a) * self._ema_spread_A + a * float(spread_A)

        hft_q = (
            order_book_A.resting_quantity(HFT_MM_BID_ORDER_ID)
            + order_book_A.resting_quantity(HFT_MM_ASK_ORDER_ID)
        )
        fallback = hft_q < MM_PHASE3_HFT_MIN_TOTAL_QTY
        if (
            spread_A is not None
            and self._ema_spread_A is not None
            and spread_A > self._ema_spread_A * MM_PHASE3_FALLBACK_SPREAD_MULTIPLIER
        ):
            fallback = True

        spread_safety_addon = MM_PHASE3_LATENCY_SAFETY_BUFFER_PIPS / PIPS_TO_PRICE_SCALE
        if fallback:
            spread_safety_addon *= MM_PHASE3_FALLBACK_BUFFER_SCALE

        extra_ticks = min(
            MM_PHASE3_MAX_EXTRA_TICKS_FROM_HFT,
            math.log1p(max(hft_q, 0.0) / max(MM_PHASE3_HFT_QTY_REFERENCE, 1.0)),
        )
        base_ticks = MM_PHASE3_DEEPEN_BASE_TICKS + extra_ticks
        if fallback:
            base_ticks = max(0.0, base_ticks - MM_PHASE3_FALLBACK_DEPTH_PULL_TICKS)
            base_ticks *= 0.25

        depth_px = base_ticks * MM_PHASE3_TICK
        s = MM_PHASE3_SIDE_DEPTH_INV_SCALE * inv_pressure
        if long_eur:
            bid_worse = 1.0 + s
            ask_worse = max(0.1, 1.0 - s)
        else:
            bid_worse = max(0.1, 1.0 - s)
            ask_worse = 1.0 + s

        utility_problem = self._build_utility_problem(
            order_book_B.mid,
            order_book_C.mid,
            spread_safety_addon=spread_safety_addon,
            quote_depth_bid_delta=depth_px * bid_worse,
            quote_depth_ask_delta=depth_px * ask_worse,
            inventory_skew_multiplier=inv_skew_mult,
        )

        bids_prices, ask_prices = utility_problem.get_price_grid()
        bids_qty, ask_qty = utility_problem.get_qty_grid()

        new_prices_bid = {f"bid_{p}": (p, bids_qty[i]) for i, p in enumerate(bids_prices)}
        new_prices_ask = {f"ask_{p}": (p, ask_qty[i]) for i, p in enumerate(ask_prices)}
        new_prices = {**new_prices_bid, **new_prices_ask}

        cancel_ids: list[str] = []
        for key in self._active_orders:
            if key not in new_prices:
                for oid, _ in self._active_orders[key]:
                    cancel_ids.append(oid)
        for key in new_prices:
            if key in self._active_orders:
                for oid, _ in self._active_orders[key]:
                    cancel_ids.append(oid)

        orders_to_submit: list[Order] = []
        for key, (price, qty_target) in new_prices.items():
            if qty_target <= 0:
                continue
            is_ask = key.startswith("ask")
            orders_to_submit.append(
                Order(
                    order_id=f"MM_{self.id_cpt}",
                    is_ask=is_ask,
                    price=price,
                    quantity=qty_target,
                )
            )
            self.id_cpt += 1

        return cancel_ids, orders_to_submit

    def apply_quote_plan(
        self,
        order_book_A: OrderBook,
        order_ids_to_cancel: list[str],
        orders_to_submit: list[Order],
    ) -> list[tuple[Order, float]]:
        """Cancel then post; refresh ``_active_orders`` from ``orders_to_submit``."""
        for oid in order_ids_to_cancel:
            order_book_A.cancel(oid)
        self._active_orders.clear()
        fills: list[tuple[Order, float]] = []
        for order in orders_to_submit:
            xf = order_book_A.add_limit_order(order)
            self.update_inventory_from_fills(xf)
            fills.extend(xf)
            lvl_key = f"{'ask' if order.is_ask else 'bid'}_{round(order.price, 4)}"
            self._active_orders[lvl_key] = [(order.order_id, order.quantity)]
        return fills

    # ===== Update inventory =====
    def update_inventory_from_fills(self, fills: list) -> None:
        for order, qty in fills:
            if not order.order_id.startswith("MM_"):
                continue

            # Mise à jour inventaire
            if order.is_ask:
                self.EUR_quantity -= qty
                self.USD_quantity += qty * order.price
            else:
                self.EUR_quantity += qty
                self.USD_quantity -= qty * order.price

            # Mise à jour tracking FIFO
            side_prefix = "ask" if order.is_ask else "bid"
            key = f"{side_prefix}_{round(order.price, 4)}"
            if key in self._active_orders:
                remaining = qty
                while remaining > 0 and self._active_orders[key]:
                    order_id, qty_active = self._active_orders[key][0]
                    if qty_active <= remaining:
                        remaining -= qty_active
                        self._active_orders[key].pop(0)
                    else:
                        self._active_orders[key][0] = (order_id, qty_active - remaining)
                        remaining = 0
                if not self._active_orders[key]:
                    self._active_orders.pop(key)

            self._has_new_fills = True
    
    
    # ===== Metrics Methods =====
    def _flush_to_parquet(self):
        """Write buffered metrics to parquet (single pass; no read-merge per flush)."""
        if self._metrics_rt_rows:
            pl.DataFrame(self._metrics_rt_rows, schema=_METRICS_RT_SCHEMA).write_parquet(PARQUET_PATH_REALTIME)
            self._metrics_rt_rows.clear()
        if self._metrics_agg_rows:
            pl.DataFrame(self._metrics_agg_rows, schema=_METRICS_AGG_SCHEMA).write_parquet(PARQUET_PATH_AGGREGATED)
            self._metrics_agg_rows.clear()

    def save_metrics(
        self,
        order_book_A: OrderBook,
        order_book_B: OrderBook,
        order_book_C: OrderBook,
        timestamp: float,
        fills: list,
        hft_snipe_count: int,
        hft_snipe_qty: float,
    ):
        # 1. CALCUL DU PRIX DE RÉFÉRENCE (VWAP B/C) - La "Fair Value"
        m_B = getattr(order_book_B, 'mid', None)
        m_C = getattr(order_book_C, 'mid', None)

        if m_B is not None and m_C is not None:
            mid_ref = self.WEIGHT_B * m_B + self.WEIGHT_C * m_C
        elif m_B is not None:
            mid_ref = m_B
        elif m_C is not None:
            mid_ref = m_C
        else:
            mid_ref = self.s0
        
        mid_ref = float(mid_ref)

        # 2. RÉCUPÉRATION DU MID LOCAL DE A (Pour le display)
        m_A = getattr(order_book_A, 'mid', None)
        mid_A_display = float(m_A) if m_A is not None else mid_ref

        # 3. CALCUL DU PNL MtM (Basé sur mid_ref pour éviter les sauts)
        self._step_count += 1
        mtm_pnl = float(self.EUR_quantity * mid_ref + self.USD_quantity - self._initial_capital)

        # 4. RÉGIME D'INVENTAIRE
        v_tot_metrics = self.EUR_quantity * mid_ref + self.USD_quantity
        target_eur_metrics = (0.5 * v_tot_metrics) / mid_ref
        dev = abs(self.EUR_quantity - target_eur_metrics)
        max_dev = (self.hedge_threshold - 0.5) * v_tot_metrics / mid_ref
        inv_pct = float(dev / max_dev) if max_dev > 0 else 0.0
        if inv_pct < INVENTORY_REGIME_NORMAL_THRESHOLD:
            regime = "Normal"
        elif inv_pct < INVENTORY_REGIME_ALERT_THRESHOLD:
            regime = "Alert"
        else:
            regime = "Hedge"

        # 5. ENREGISTREMENT REALTIME (Chaque step)
        row_rt = {
            "timestamp": float(timestamp),
            "mid_A": float(mid_A_display),
            "mid_ref": float(mid_ref),
            "EUR_quantity": float(self.EUR_quantity),
            "USD_quantity": float(self.USD_quantity),
            "inventory_pct": float(inv_pct),
            "hedge_regime": str(regime),
            "mtm_pnl": float(mtm_pnl),
        }
        self._metrics_rt_rows.append(row_rt)

        # 6. ENREGISTREMENT AGGREGATED (Tous les 100 steps)
        if self._step_count % MARKET_MAKER_AGGREGATION_STEPS == 0:
            # On recrée le problème d'utilité pour avoir reservation_price et optimal_spread
            utility = self._build_utility_problem(mid_B=mid_ref, mid_C=mid_ref) # On simplifie ici

            # --- SÉCURISATION DES DONNÉES DU CARNET A ---
            s_quoted = getattr(order_book_A, 'spread', None)
            s_quoted = float(s_quoted) if s_quoted is not None else 0.0
            
            b_price, _ = getattr(order_book_A, 'best_bid', (None, 0.0))
            b_price = float(b_price) if b_price is not None else 0.0
            
            a_price, _ = getattr(order_book_A, 'best_ask', (None, 0.0))
            a_price = float(a_price) if a_price is not None else 0.0
            # --------------------------------------------

            mm_fills = [f for f in fills if f[0].order_id.startswith("MM_")]
            total_qty = sum(f[1] for f in mm_fills)
            spread_cap = sum(abs(f[0].price - mid_ref) * f[1] for f in mm_fills) / total_qty if total_qty > 0 else 0.0

            row_agg = {
                "timestamp": float(timestamp),
                "mid_A": float(mid_A_display),
                "mid_ref": float(mid_ref),
                "reservation_price": float(utility.reservation_price),
                "optimal_spread": float(utility.optimal_spread),
                "spread_quoted": s_quoted, # Utilise la version sécurisée
                "best_bid_A": b_price,     # Utilise la version sécurisée
                "best_ask_A": a_price,     # Utilise la version sécurisée
                "EUR_quantity": float(self.EUR_quantity),
                "USD_quantity": float(self.USD_quantity),
                "inventory_pct": float(inv_pct),
                "hedge_regime": str(regime),
                "mtm_pnl": float(mtm_pnl),
                "realized_pnl": float(self._realized_pnl),
                "hedge_cost": float(self._hedge_cost),
                "fill_rate_bid": float(sum(f[1] for f in mm_fills if not f[0].is_ask) / utility.inventory_max),
                "fill_rate_ask": float(sum(f[1] for f in mm_fills if f[0].is_ask) / utility.inventory_max),
                "spread_capture": float(spread_cap),
                "adverse_selection": float(spread_cap),
                "hft_snipe_count": int(hft_snipe_count),
                "hft_snipe_qty": float(hft_snipe_qty),
                "arb_opportunity_count": int(hft_snipe_count),
                "arb_opportunity_size": float(hft_snipe_qty),
            }
            self._metrics_agg_rows.append(row_agg)

    def compute_summary_stats(self) -> pl.DataFrame:
        """
        Flush final + aggregated stats on all the backtest.
        Call one time at the end of the simulation.
        """
        self._flush_to_parquet()

        if not os.path.exists(PARQUET_PATH_AGGREGATED):
            return pl.DataFrame() 

        df = pl.read_parquet(PARQUET_PATH_AGGREGATED)
        if len(df) == 0:
            return pl.DataFrame()

        return df.select([
            pl.col("mtm_pnl").mean().alias("avg_mtm_pnl"),
            pl.col("mtm_pnl").median().alias("median_mtm_pnl"),
            pl.col("mtm_pnl").quantile(0.05).alias("pct5_mtm_pnl"),
            pl.col("mtm_pnl").quantile(0.95).alias("pct95_mtm_pnl"),
            pl.col("realized_pnl").last().alias("total_realized_pnl"),
            pl.col("hedge_cost").last().alias("total_hedge_cost"),
            pl.col("fill_rate_bid").mean().alias("avg_fill_rate_bid"),
            pl.col("fill_rate_ask").mean().alias("avg_fill_rate_ask"),
            pl.col("spread_capture").mean().alias("avg_spread_capture"),
            pl.col("hft_snipe_count").sum().alias("total_hft_snipes"),
            pl.col("hft_snipe_qty").sum().alias("total_hft_qty_sniped"),
            pl.col("inventory_pct").mean().alias("avg_inventory_pct"),
            pl.col("hedge_regime").value_counts().alias("regime_distribution"),
        ])

    def get_EUR_quantity(self) -> float:
        return self.EUR_quantity

    def get_USD_quantity(self) -> float:
        return self.USD_quantity

    def make_market(self, order_book_A: OrderBook, order_book_B: OrderBook, order_book_C: OrderBook) -> None:

        # Skip market making if reference prices are unavailable
        if order_book_B.mid is None or order_book_C.mid is None:
            warnings.warn("Reference price unavailable, skipping market making this step. No mid B and C.")
            for oid in self._flatten_mm_order_ids():
                order_book_A.cancel(oid)
            self._active_orders.clear()
            return

        utility_problem = self._build_utility_problem(
            mid_B=order_book_B.mid,
            mid_C=order_book_C.mid,
        )
        new_mid = utility_problem.ref_price

        if self._last_posted_mid is not None:
            delta = abs(new_mid - self._last_posted_mid)
            if delta < self.epsilon and not self._has_new_fills:
                return

        cancel_ids, orders_to_submit = self.plan_phase3_quote_actions(
            order_book_A, order_book_B, order_book_C
        )
        self.apply_quote_plan(order_book_A, cancel_ids, orders_to_submit)

        self._last_posted_mid = new_mid
        self._has_new_fills = False


    def check_and_hedge(self, order_book_B: OrderBook, order_book_C: OrderBook, current_time: float):
        """
        Check if inventory is too skewed and hedge if necessary.
        Hedge threshold: 90% of total capital value.
        Picks the cheapest venue (net of taker fees), with partial fill split if needed.
        Returns (order_B, order_C), either can be None if not needed.
        """

        price_B_ask, _ = order_book_B.best_ask
        price_B_bid, _ = order_book_B.best_bid
        price_C_ask, _ = order_book_C.best_ask
        price_C_bid, _ = order_book_C.best_bid

        if price_B_ask is None or price_B_bid is None or price_C_ask is None or price_C_bid is None:
            return None, None

        mid_B = (price_B_ask + price_B_bid) / 2
        mid_C = (price_C_ask + price_C_bid) / 2
        mid_ref = self.WEIGHT_B * mid_B + self.WEIGHT_C * mid_C
        eur_value = self.EUR_quantity * mid_ref
        usd_value = self.USD_quantity
        v_tot = eur_value + usd_value

        if max(eur_value, usd_value) < self.hedge_leg_trigger * v_tot:
            return None, None

        # Rebalance target: 50% of total capital in EUR (same ref as quoting)
        target_eur_qty = (0.5 * v_tot) / mid_ref

        is_ask = self.EUR_quantity > target_eur_qty
        hedge_qty_eur = abs(self.EUR_quantity - target_eur_qty)

        # Best prices and available quantities on each venue
        if is_ask:
            # We're selling EUR : we hit the bid
            price_B, qty_B = order_book_B.best_bid
            price_C, qty_C = order_book_C.best_bid
            net_B = price_B * (1 - FEES_TAKER_B)
            net_C = price_C * (1 - FEES_TAKER_C)
            prefer_B = net_B >= net_C
        else:
            # We're buying EUR : we lift the ask
            price_B, qty_B = order_book_B.best_ask
            price_C, qty_C = order_book_C.best_ask
            net_B = price_B * (1 + FEES_TAKER_B)
            net_C = price_C * (1 + FEES_TAKER_C)
            prefer_B = net_B <= net_C

        # Allocate quantities: preferred venue first, other venue for remainder
        if prefer_B:
            qty_primary = min(hedge_qty_eur, qty_B)
            qty_secondary = min(hedge_qty_eur - qty_primary, qty_C)
            primary = "B"
        else:
            qty_primary = min(hedge_qty_eur, qty_C)
            qty_secondary = min(hedge_qty_eur - qty_primary, qty_B)
            primary = "C"

        import time
        order_B = None
        order_C = None

        if primary == "B":
            if qty_primary > 0:
                order_B = Order(
                    order_id=f"hedge_B_{time.time_ns()}",
                    is_ask=is_ask,
                    price=float('inf') if not is_ask else 0.0,
                    quantity=qty_primary,
                    # timestamp=current_time + 0.200,
                )
            if qty_secondary > 0:
                order_C = Order(
                    order_id=f"hedge_C_{time.time_ns()}",
                    is_ask=is_ask,
                    price=float('inf') if not is_ask else 0.0,
                    quantity=qty_secondary,
                    # timestamp=current_time + 0.170,
                )
        else:
            if qty_primary > 0:
                order_C = Order(
                    order_id=f"hedge_C_{time.time_ns()}",
                    is_ask=is_ask,
                    price=float('inf') if not is_ask else 0.0,
                    quantity=qty_primary,
                    # timestamp=current_time + 0.170,
                )
            if qty_secondary > 0:
                order_B = Order(
                    order_id=f"hedge_B_{time.time_ns()}",
                    is_ask=is_ask,
                    price=float('inf') if not is_ask else 0.0,
                    quantity=qty_secondary,
                    # timestamp=current_time + 0.200,
                )

        return order_B, order_C


# %%%%%% Market Making Test %%%%%%
def test_making():
    print("=== test_making ===")
    order_book_A = OrderBook(lambda_a0=5.0, alpha=0.05, theta=0.1, lambda_mo=2.0, v_unit=100_000)
    order_book_B = OrderBook(lambda_a0=5.0, alpha=0.05, theta=0.1, lambda_mo=2.0, v_unit=100_000)
    order_book_C = OrderBook(lambda_a0=5.0, alpha=0.05, theta=0.1, lambda_mo=2.0, v_unit=100_000)

    # Simple books like in HFT.py
    order_book_B.add_limit_order(Order("B_BID_1", False, 1.1, 500_000))
    order_book_B.add_limit_order(Order("B_ASK_1", True, 1.2, 500_000))
    order_book_B.add_limit_order(Order("B_BID_2", False, 1.0, 1_000_000))
    order_book_B.add_limit_order(Order("B_ASK_2", True, 1.3, 1_000_000))
    order_book_C.add_limit_order(Order("C_BID_1", False, 1.1, 500_000))
    order_book_C.add_limit_order(Order("C_ASK_1", True, 1.2, 500_000))
    order_book_C.add_limit_order(Order("C_BID_2", False, 1.0, 1_000_000))
    order_book_C.add_limit_order(Order("C_ASK_2", True, 1.3, 1_000_000))

    # HERE TO CHECK MANUALLY FOR PARAMS    
    mm = MarketMaker(
        EUR_quantity=500_000.0,
        USD_quantity=600_000.0,
        gamma=0.005,
        sigma=0.005,
        kappa=50,
        T=1,
        s0=1.15,  # initial mid price (used for initial NAV only)
    )

    before_orders = len(order_book_A._orders)
    mm.make_market(order_book_A, order_book_B, order_book_C)
    after_orders = len(order_book_A._orders)

    print(f"Best bid/ask on A         : {order_book_A.best_bid} / {order_book_A.best_ask}")
    print(f"Best bid/ask on B         : {order_book_B.best_bid} / {order_book_B.best_ask}")
    print(f"Best bid/ask on C         : {order_book_C.best_bid} / {order_book_C.best_ask}")
    print(f"Orders on A before making: {before_orders}")
    print(f"Orders on A after making : {after_orders}")

    order_book_A.print_book(depth=5)


def test_making_phase_3() -> None:
    """
    Phase 3 MM quoting on A after HFT.make_market_on_A under three HFT regimes:
    both sides on, HFT off (pulled), one-sided only.
    """
    print("\n=== test_making_phase_3 ===")

    def _books():
        ob_a = OrderBook(lambda_a0=5.0, alpha=0.05, theta=0.1, lambda_mo=2.0, v_unit=100_000)
        ob_b = OrderBook(lambda_a0=5.0, alpha=0.05, theta=0.1, lambda_mo=2.0, v_unit=100_000)
        ob_c = OrderBook(lambda_a0=5.0, alpha=0.05, theta=0.1, lambda_mo=2.0, v_unit=100_000)
        ob_b.add_limit_order(Order("B_BID_1", False, 1.0985, 250_000))
        ob_b.add_limit_order(Order("B_ASK_1", True, 1.0990, 150_000))
        ob_c.add_limit_order(Order("C_BID_1", False, 1.1012, 120_000))
        ob_c.add_limit_order(Order("C_ASK_1", True, 1.1016, 250_000))
        return ob_a, ob_b, ob_c

    def _mm():
        return MarketMaker(
            EUR_quantity=500_000.0,
            USD_quantity=600_000.0,
            gamma=0.005,
            sigma=0.005,
            kappa=50,
            T=1,
            s0=1.15,
        )

    hft = HFT()

    def _fmt_px(p: Optional[float]) -> str:
        return f"{p:.4f}" if p is not None else "—"

    def _mm_best_own_quotes(mm: MarketMaker) -> tuple[Optional[float], Optional[float]]:
        """Best bid (highest) and best ask (lowest) among MM resting levels on A."""
        best_bid: Optional[float] = None
        best_ask: Optional[float] = None
        for key in mm._active_orders:
            if not mm._active_orders[key]:
                continue
            side, _, rest = key.partition("_")
            try:
                px = float(rest)
            except ValueError:
                continue
            if side == "bid":
                best_bid = px if best_bid is None else max(best_bid, px)
            elif side == "ask":
                best_ask = px if best_ask is None else min(best_ask, px)
        return best_bid, best_ask

    # --- prob_off=0, prob_one_sided=0: HFT at touch, MM posts deep ladder behind ---
    print("  [case 1] prob_off=0, prob_one_sided=0")
    ob_a, ob_b, ob_c = _books()
    hft.make_market_on_A(
        ob_a,
        ob_b,
        ob_c,
        prob_off=0.0,
        prob_one_sided=0.0,
        half_spread=0.00005,
        rng=random.Random(42),
    )
    p_hft_bid = ob_a.resting_price(HFT_MM_BID_ORDER_ID)
    p_hft_ask = ob_a.resting_price(HFT_MM_ASK_ORDER_ID)
    print(f"    HFT quote prices (bid / ask): {_fmt_px(p_hft_bid)} / {_fmt_px(p_hft_ask)}")
    q_hft_bid = ob_a.resting_quantity(HFT_MM_BID_ORDER_ID)
    q_hft_ask = ob_a.resting_quantity(HFT_MM_ASK_ORDER_ID)
    assert q_hft_bid > 0 and q_hft_ask > 0, "HFT should post bid and ask"
    mm = _mm()
    mm.make_market(ob_a, ob_b, ob_c)
    assert len(mm._active_orders) == DEFAULT_MAX_LEVELS * 2
    assert ob_a.resting_quantity(HFT_MM_BID_ORDER_ID) > 0
    assert ob_a.resting_quantity(HFT_MM_ASK_ORDER_ID) > 0
    print(f"    HFT bid/ask qty on A: {q_hft_bid:.0f} / {q_hft_ask:.0f} → after MM: "
          f"{ob_a.resting_quantity(HFT_MM_BID_ORDER_ID):.0f} / "
          f"{ob_a.resting_quantity(HFT_MM_ASK_ORDER_ID):.0f}")
    print(
        "    HFT quote prices after MM (bid / ask): "
        f"{_fmt_px(ob_a.resting_price(HFT_MM_BID_ORDER_ID))} / "
        f"{_fmt_px(ob_a.resting_price(HFT_MM_ASK_ORDER_ID))}"
    )
    mb, ma = _mm_best_own_quotes(mm)
    print(f"    MM best own quotes (bid / ask): {_fmt_px(mb)} / {_fmt_px(ma)}")
    print(f"    MM active levels: {len(mm._active_orders)}, A best (book): {ob_a.best_bid[0]} / {ob_a.best_ask[0]}")

    # --- prob_off=1: no HFT liquidity; MM should still quote (fallback-style tightness) ---
    print("  [case 2] prob_off=1, prob_one_sided=0")
    ob_a, ob_b, ob_c = _books()
    hft.make_market_on_A(
        ob_a,
        ob_b,
        ob_c,
        prob_off=1.0,
        prob_one_sided=0.0,
        rng=random.Random(0),
    )
    print(
        "    HFT quote prices (bid / ask): "
        f"{_fmt_px(ob_a.resting_price(HFT_MM_BID_ORDER_ID))} / "
        f"{_fmt_px(ob_a.resting_price(HFT_MM_ASK_ORDER_ID))}"
    )
    assert ob_a.resting_quantity(HFT_MM_BID_ORDER_ID) == 0.0
    assert ob_a.resting_quantity(HFT_MM_ASK_ORDER_ID) == 0.0
    mm = _mm()
    mm.make_market(ob_a, ob_b, ob_c)
    assert len(mm._active_orders) == DEFAULT_MAX_LEVELS * 2
    print(f"    HFT on A after off: bid={ob_a.resting_quantity(HFT_MM_BID_ORDER_ID):.0f} "
          f"ask={ob_a.resting_quantity(HFT_MM_ASK_ORDER_ID):.0f}")
    mb, ma = _mm_best_own_quotes(mm)
    print(f"    MM best own quotes (bid / ask): {_fmt_px(mb)} / {_fmt_px(ma)}")
    print(f"    MM active levels: {len(mm._active_orders)}, A best (book): {ob_a.best_bid[0]} / {ob_a.best_ask[0]}")

    # --- prob_one_sided=1: exactly one HFT side; MM still builds full ladder ---
    print("  [case 3] prob_off=0, prob_one_sided=1 (rng seed 0 → one side only)")
    ob_a, ob_b, ob_c = _books()
    hft.make_market_on_A(
        ob_a,
        ob_b,
        ob_c,
        prob_off=0.0,
        prob_one_sided=1.0,
        half_spread=0.00005,
        rng=random.Random(0),
    )
    print(
        "    HFT quote prices (bid / ask): "
        f"{_fmt_px(ob_a.resting_price(HFT_MM_BID_ORDER_ID))} / "
        f"{_fmt_px(ob_a.resting_price(HFT_MM_ASK_ORDER_ID))}"
    )
    qb = ob_a.resting_quantity(HFT_MM_BID_ORDER_ID)
    qa = ob_a.resting_quantity(HFT_MM_ASK_ORDER_ID)
    assert (qb == 0.0) != (qa == 0.0), "HFT should quote exactly one side"
    mm = _mm()
    mm.make_market(ob_a, ob_b, ob_c)
    assert len(mm._active_orders) == DEFAULT_MAX_LEVELS * 2
    print(f"    HFT resting bid/ask qty: {qb:.0f} / {qa:.0f}")
    mb, ma = _mm_best_own_quotes(mm)
    print(f"    MM best own quotes (bid / ask): {_fmt_px(mb)} / {_fmt_px(ma)}")
    print(f"    MM active levels: {len(mm._active_orders)}, A best (book): {ob_a.best_bid[0]} / {ob_a.best_ask[0]}")

    print("  test_making_phase_3: OK")


# %%%%%% Hedging Tests %%%%%%
def test_hedging():
    print("\n=== test_hedging ===")
    order_book_B = OrderBook(lambda_a0=5.0, alpha=0.05, theta=0.1, lambda_mo=2.0, v_unit=100_000)
    order_book_C = OrderBook(lambda_a0=5.0, alpha=0.05, theta=0.1, lambda_mo=2.0, v_unit=100_000)

    # Liquidity available on both venues
    order_book_B.add_limit_order(Order("B_BID_1", False, 1.1, 300_000))
    order_book_B.add_limit_order(Order("B_ASK_1", True, 1.2, 300_000))
    order_book_C.add_limit_order(Order("C_BID_1", False, 1.05, 400_000))
    order_book_C.add_limit_order(Order("C_ASK_1", True, 1.25, 400_000))

    # Start with inventory > 90% of q_max to trigger hedging
    mm = MarketMaker(
        EUR_quantity=950_000.0,
        USD_quantity=0.0,
        gamma=0.1,
        sigma=0.01,
        kappa=1.5,
        T=1.0,
        s0=1.15,  # initial mid price
    )

    order_B, order_C = mm.check_and_hedge(order_book_B, order_book_C, current_time=0.0)
    print(f"Hedge order to B: {order_B}")
    print(f"Hedge order to C: {order_C}")

    fills_B, fills_C = [], []
    if order_B is not None:
        fills_B = order_book_B.add_limit_order(order_B)
    if order_C is not None:
        fills_C = order_book_C.add_limit_order(order_C)

    total_hedged = sum(q for _, q in fills_B) + sum(q for _, q in fills_C)
    gross_cash_flow = sum(q * o.price for o, q in fills_B + fills_C)

    if mm.EUR_quantity > 0:
        mm.EUR_quantity -= total_hedged
        mm.USD_quantity += gross_cash_flow
    else:
        mm.EUR_quantity += total_hedged
        mm.USD_quantity -= gross_cash_flow

    print(f"Total hedged qty          : {total_hedged}")
    print(f"EUR inventory after hedge : {mm.EUR_quantity}")
    print(f"USD cash after hedge      : {mm.USD_quantity}")


# %%%%%% Main %%%%%%
if __name__ == "__main__":
    
    # test_making()

    test_making_phase_3()
    
    # test_hedging()