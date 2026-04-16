import polars as pl
from polars import col as c
from OrderBook import OrderBook, Order, PriceLevel
import math
from abc import ABC, abstractmethod
import time
import os
from typing import Optional

FEES_TAKER_B = 0.0002
FEES_TAKER_C = 0.0003
FLUSH_INTERVAL = 1_000
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARQUET_PATH_REALTIME = os.path.join(BASE_DIR, "metrics_realtime.parquet")
PARQUET_PATH_AGGREGATED = os.path.join(BASE_DIR, "metrics_aggregated.parquet")

# %%%%%% Price Grid Methods %%%%%%

class PriceGridStrategy(ABC):

    @abstractmethod
    def generate(self, problem: "UtilityProblem") -> tuple[list[float], list[float]]:
        pass


class NaivePriceGridStrategy(PriceGridStrategy):

    def __init__(self, max_levels: int = 10, tick_size: float = 0.0001):
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
                 delta_grid: float = 0.0001, 
                 geo_increment:float = 1.4, 
                 max_levels: int = 10, 
                 tick_size: float = 0.0001
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
            bids.append(ref_bid - increment)
            asks.append(ref_ask + increment)
            increment += round(self.delta_grid * (self.geo_increment ** i), 4) # round up to the closest tick

        return (bids, asks)
    

# %%%%%% Qunatity Grid Methods %%%%%%

class QuantityGridStrategy(ABC):

    @abstractmethod
    def generate(self, problem: "UtilityProblem") -> tuple[list[float], list[float]]:
        pass


class NaiveQuantityGridStrategy(QuantityGridStrategy):

    def __init__(self, max_levels: int = 10):
        self.max_levels = max_levels

    def generate(self, problem: "UtilityProblem") -> tuple[list[float], list[float]]:

        qty_per_level = int(problem.inventory_max / self.max_levels)
        bids = [qty_per_level] * self.max_levels
        asks = [qty_per_level] * self.max_levels
        return (bids, asks)


class GeometricQuantityGridStrategy(QuantityGridStrategy):

    def __init__(self, alpha: float = 0.7,max_levels: int = 10):
        self.alpha = alpha
        self.max_levels = max_levels

    def generate(self, problem: "UtilityProblem") -> tuple[list[float], list[float]]:

        bids, asks = [], []

        headroom = max(0, problem.inventory_max - abs(problem.inventory))
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
                 kapital: float = 1_000_000.0,
                 delta_threshold: float = 0.05,
                 fees_pips: float = 2.0,
                 price_grid_strategy: PriceGridStrategy = NaivePriceGridStrategy(),
                 quantity_grid_strategy: QuantityGridStrategy = NaiveQuantityGridStrategy(),
                 ):
        
        self.gamma = gamma
        self.sigma = sigma
        self.T_minus_t = self.T_minus_t = max(remaining_time, 0.001) # clamp
        self.inventory = inventory
        self.ref_price = ref_price
        self.kappa = kappa
        self.latency = latency
        self.fees_pips = fees_pips
        self.kapital = kapital
        self.delta_threshold = delta_threshold
        self.inventory_max = self.kapital * self.delta_threshold

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
        return self.fees_pips / 10_000.0

    @property
    def optimal_spread(self) -> float:
        return self.psi_Avellaneda_Stoikov + self.psi_snipe + self.psi_fees

    @property
    def best_ask(self) -> float:
        return self.reservation_price + self.optimal_spread / 2

    @property
    def best_bid(self) -> float:
        return self.reservation_price - self.optimal_spread / 2

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

    WEIGHT_B = 0.75
    WEIGHT_C = 0.25
    LATENCY_B = 0.200
    LATENCY_C = 0.170
    DELTA_TAU = 0.150

    def __init__(self, 
                 EUR_quantity: float, 
                 USD_quantity: float,
                 gamma: float,
                 sigma: float,
                 kappa: float,
                 T: float,
                 q_max:float,
                 s0: float
                 ):
        self.s0 = s0
        self.EUR_quantity = EUR_quantity
        self.USD_quantity = USD_quantity
        self.gamma = gamma
        self.sigma = sigma
        self.kappa = kappa
        self.T = T
        self.q_max = q_max
        self._t = 0.0
        self._step_count = 0
        self.id_cpt = 0
        self._initial_capital = EUR_quantity * s0 + USD_quantity

        # Accumulators
        self._realized_pnl = 0.0
        self._hedge_cost = 0.0

        # Orders
        self._active_orders: dict[str, list[tuple[str, float]]] = {}  # "bid_1.0850" -> [("MM_42", 300_000), ("MM_87", 150_000)]
        self._last_posted_mid: Optional[float] = None
        self._has_new_fills: bool = False
        self.epsilon: float = 0.00005  # 0.5 pip

        # Realtime Buffer : every step
        self.metrics_realtime = pl.DataFrame(schema={
            "timestamp": pl.Float64,
            "mid_A": pl.Float64,
            "mid_ref": pl.Float64,
            "EUR_quantity": pl.Float64,
            "USD_quantity": pl.Float64,
            "inventory_pct": pl.Float64,
            "hedge_regime": pl.String,
            "mtm_pnl": pl.Float64,
        })

        # Aggregated Buffer : every 100 steps
        self.metrics_aggregated = pl.DataFrame(schema={
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
        })

    # === protected methods ===
    
    def _ref_price(self, mid_B: float, mid_C: float) -> float:
        return self.WEIGHT_B * mid_B + self.WEIGHT_C * mid_C
    
    def _build_utility_problem(self, mid_B: float, mid_C: float) -> UtilityProblem:

        return UtilityProblem(
            gamma=self.gamma,
            sigma=self.sigma,
            remaining_time=self.T - self._t,
            inventory=self.EUR_quantity,
            ref_price=self._ref_price(mid_B, mid_C),
            kappa=self.kappa,
            latency=self.DELTA_TAU,
            price_grid_strategy=GeometricPriceGridStrategy(),
            quantity_grid_strategy=GeometricQuantityGridStrategy(),
        )

    # ===== Update inventory =====
    def update_inventory_from_fills(self, fills: list) -> None:
        for order, qty in fills:
            if not order.order_id.startswith("MM_"):
                continue

            # Mise à jour inventaire
            if order.side == "bid":
                self.EUR_quantity += qty
                self.USD_quantity -= qty * order.price
            else:
                self.EUR_quantity -= qty
                self.USD_quantity += qty * order.price

            # Mise à jour tracking FIFO
            side_prefix = "bid" if order.side == "bid" else "ask"
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
        """Flush les buffers en mémoire sur disque en mode append."""
        for path, df in [
            (PARQUET_PATH_REALTIME, self.metrics_realtime),
            (PARQUET_PATH_AGGREGATED, self.metrics_aggregated),
        ]:
            if df.is_empty():
                continue
            if os.path.exists(path):
                existing = pl.read_parquet(path)
                pl.concat([existing, df]).write_parquet(path)
            else:
                df.write_parquet(path)
        self.metrics_realtime = self.metrics_realtime.clear()
        self.metrics_aggregated = self.metrics_aggregated.clear()

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
            mid_ref = 0.75 * m_B + 0.25 * m_C
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
        abs_inv = abs(self.EUR_quantity)
        inv_pct = float(abs_inv / self.q_max) if self.q_max > 0 else 0.0
        if inv_pct < 0.70:
            regime = "Normal"
        elif inv_pct < 0.90:
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
        self.metrics_realtime = self.metrics_realtime.vstack(pl.DataFrame([row_rt], schema=self.metrics_realtime.schema))

        # 6. ENREGISTREMENT AGGREGATED (Tous les 100 steps)
        if self._step_count % 100 == 0:
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
                "fill_rate_bid": float(sum(f[1] for f in mm_fills if f[0].side == 'bid') / self.q_max),
                "fill_rate_ask": float(sum(f[1] for f in mm_fills if f[0].side == 'ask') / self.q_max),
                "spread_capture": float(spread_cap),
                "adverse_selection": float(spread_cap),
                "hft_snipe_count": int(hft_snipe_count),
                "hft_snipe_qty": float(hft_snipe_qty),
                "arb_opportunity_count": int(hft_snipe_count),
                "arb_opportunity_size": float(hft_snipe_qty),
            }
            self.metrics_aggregated = self.metrics_aggregated.vstack(pl.DataFrame([row_agg], schema=self.metrics_aggregated.schema))
        # 7. FLUSH PERIODIQUE
        if self._step_count % FLUSH_INTERVAL == 0:
            self._flush_to_parquet()

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

        utility_problem = self._build_utility_problem(
            mid_B=order_book_B.mid,
            mid_C=order_book_C.mid,
        )
        new_mid = utility_problem.ref_price

        # === Vérifier si repost nécessaire ===
        best_bid, _ = order_book_A.best_bid
        best_ask, _ = order_book_A.best_ask
        book_incomplete = best_ask is None or best_bid is None

        if self._last_posted_mid is not None:
            delta = abs(new_mid - self._last_posted_mid)
            if delta < self.epsilon and not self._has_new_fills and not book_incomplete:
                return

        bids_prices, ask_prices = utility_problem.get_price_grid()
        bids_qty, ask_qty = utility_problem.get_qty_grid()

        new_prices_bid = {f"bid_{round(p, 4)}": (p, bids_qty[i]) for i, p in enumerate(bids_prices)}
        new_prices_ask = {f"ask_{round(p, 4)}": (p, ask_qty[i]) for i, p in enumerate(ask_prices)}
        new_prices = {**new_prices_bid, **new_prices_ask}

        # === cancel levels that desapear ===
        keys_to_cancel = [key for key in self._active_orders if key not in new_prices]
        for key in keys_to_cancel:
            for order_id, _ in self._active_orders.pop(key):
                order_book_A.cancel(order_id)

        # === add or change level ===
        keys_to_cancel = [key for key in self._active_orders if key not in new_prices]
        for key in keys_to_cancel:
            for order_id, _ in self._active_orders.pop(key):
                order_book_A.cancel(order_id)

        # === cancel/repost systématique sur chaque niveau ===
        for key, (price, qty_target) in new_prices.items():
            side = "bid" if key.startswith("bid") else "ask"

            # Cancel les ordres existants sur ce niveau
            if key in self._active_orders:
                for order_id, _ in self._active_orders[key]:
                    order_book_A.cancel(order_id)
                self._active_orders[key] = []

            # Repost si qty > 0
            if qty_target > 0:
                order = Order(
                    order_id=f"MM_{self.id_cpt}",
                    side=side,
                    price=price,
                    quantity=qty_target,
                )
                self.id_cpt += 1
                order_book_A.add_limit_order(order)
                self._active_orders[key] = [(order.order_id, qty_target)]

        self._last_posted_mid = new_mid
        self._has_new_fills = False


    def check_and_hedge(self, order_book_B: OrderBook, order_book_C: OrderBook, current_time: float):
        """
        Check if inventory is too skewed and hedge if necessary.
        Hedge threshold: 90% of q_max.
        Picks the cheapest venue (net of taker fees), with partial fill split if needed.
        Returns (order_B, order_C), either can be None if not needed.
        """
        abs_inventory = abs(self.EUR_quantity)

        if abs_inventory < 0.9 * self.q_max:
            return None, None

        price_B_bid, qty_B_bid = order_book_B.best_bid
        price_B_ask, qty_B_ask = order_book_B.best_ask
        price_C_bid, qty_C_bid = order_book_C.best_bid
        price_C_ask, qty_C_ask = order_book_C.best_ask

        if any(p is None for p in [price_B_bid, price_B_ask, price_C_bid, price_C_ask]):
            return None, None

        # Bring inventory back to 50% of q_max
        hedge_qty = abs_inventory - 0.5 * self.q_max
        hedge_side = "ask" if self.EUR_quantity > 0 else "bid"

        # Best prices and available quantities on each venue
        if hedge_side == "ask":
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
            qty_primary = min(hedge_qty, qty_B)
            qty_secondary = min(hedge_qty - qty_primary, qty_C)
            price_primary, price_secondary = price_B, price_C
            primary = "B"
        else:
            qty_primary = min(hedge_qty, qty_C)
            qty_secondary = min(hedge_qty - qty_primary, qty_B)
            price_primary, price_secondary = price_C, price_B
            primary = "C"

        order_B = None
        order_C = None

        if primary == "B":
            if qty_primary > 0:
                order_B = Order(
                    order_id=f"hedge_B_{time.time_ns()}",
                    side=hedge_side,
                    price=float('inf') if hedge_side == 'bid' else 0.0,
                    quantity=qty_primary,
                    timestamp=current_time + 0.200,
                )
            if qty_secondary > 0:
                order_C = Order(
                    order_id=f"hedge_C_{time.time_ns()}",
                    side=hedge_side,
                    price=float('inf') if hedge_side == 'bid' else 0.0,
                    quantity=qty_secondary,
                    timestamp=current_time + 0.170,
                )
        else:
            if qty_primary > 0:
                order_C = Order(
                    order_id=f"hedge_C_{time.time_ns()}",
                    side=hedge_side,
                    price=float('inf') if hedge_side == 'bid' else 0.0,
                    quantity=qty_primary,
                    timestamp=current_time + 0.170,
                )
            if qty_secondary > 0:
                order_B = Order(
                    order_id=f"hedge_B_{time.time_ns()}",
                    side=hedge_side,
                    price=float('inf') if hedge_side == 'bid' else 0.0,
                    quantity=qty_secondary,
                    timestamp=current_time + 0.200,
                )

        return order_B, order_C


# %%%%%% Market Making Test %%%%%%
def test_making():
    print("=== test_making ===")
    order_book_A = OrderBook(lambda_a0=5.0, alpha=0.05, theta=0.1, lambda_mo=2.0, v_unit=100_000)
    order_book_B = OrderBook(lambda_a0=5.0, alpha=0.05, theta=0.1, lambda_mo=2.0, v_unit=100_000)
    order_book_C = OrderBook(lambda_a0=5.0, alpha=0.05, theta=0.1, lambda_mo=2.0, v_unit=100_000)

    # Simple books like in HFT.py
    order_book_B.add_limit_order(Order("B_BID_1", "bid", 1.1, 500_000))
    order_book_B.add_limit_order(Order("B_ASK_1", "ask", 1.2, 500_000))
    order_book_B.add_limit_order(Order("B_BID_2", "bid", 1.0, 1_000_000))
    order_book_B.add_limit_order(Order("B_ASK_2", "ask", 1.3, 1_000_000))
    order_book_C.add_limit_order(Order("C_BID_1", "bid", 1.1, 500_000))
    order_book_C.add_limit_order(Order("C_ASK_1", "ask", 1.2, 500_000))
    order_book_C.add_limit_order(Order("C_BID_2", "bid", 1.0, 1_000_000))
    order_book_C.add_limit_order(Order("C_ASK_2", "ask", 1.3, 1_000_000))

    mm = MarketMaker(
        EUR_quantity=0.0,
        USD_quantity=1_000_000.0,
        gamma=0.005,
        sigma=0.005,
        kappa=50,
        T=0.001,
        q_max=1_000_000.0,
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


# %%%%%% Hedging Tests %%%%%%
def test_hedging():
    print("\n=== test_hedging ===")
    order_book_B = OrderBook(lambda_a0=5.0, alpha=0.05, theta=0.1, lambda_mo=2.0, v_unit=100_000)
    order_book_C = OrderBook(lambda_a0=5.0, alpha=0.05, theta=0.1, lambda_mo=2.0, v_unit=100_000)

    # Liquidity available on both venues
    order_book_B.add_limit_order(Order("B_BID_1", "bid", 1.1, 300_000))
    order_book_B.add_limit_order(Order("B_ASK_1", "ask", 1.2, 300_000))
    order_book_C.add_limit_order(Order("C_BID_1", "bid", 1.05, 400_000))
    order_book_C.add_limit_order(Order("C_ASK_1", "ask", 1.25, 400_000))

    # Start with inventory > 90% of q_max to trigger hedging
    mm = MarketMaker(
        EUR_quantity=950_000.0,
        USD_quantity=0.0,
        gamma=0.1,
        sigma=0.01,
        kappa=1.5,
        T=1.0,
        q_max=1_000_000.0,
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
    
    test_making()
    
    test_hedging()