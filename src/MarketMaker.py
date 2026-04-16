import polars as pl
from polars import col as c
from OrderBook import OrderBook, Order, PriceLevel
import math
from abc import ABC, abstractmethod
import time
import os

FEES_TAKER_B = 0.0002
FEES_TAKER_C = 0.0003
FLUSH_INTERVAL = 10_000
PARQUET_PATH_REALTIME = "metrics_realtime.parquet"
PARQUET_PATH_AGGREGATED = "metrics_aggregated.parquet"

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
        self.T_minus_t = remaining_time
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
                 ):
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

        # Accumulators
        self._realized_pnl = 0.0
        self._hedge_cost = 0.0

        # Realtime Buffer : every step
        self.metrics_realtime = pl.DataFrame(schema={
            "timestamp": pl.Float64,
            "mid_A": pl.Float64,
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
        """
        Update EUR_quantity and USD_quantity from fills received on A.
        - A fill on a bid MM order : EUR_quantity++, USD_quantity--
        - A fill on a ask MM order : EUR_quantity--, USD_quantity++
        """
        for order, qty in fills:
            if not order.order_id.startswith("MM_"):
                continue
            if order.side == "bid":
                self.EUR_quantity += qty
                self.USD_quantity -= qty * order.price
            else:  # ask
                self.EUR_quantity -= qty
                self.USD_quantity += qty * order.price
   
    
    # ===== Metrics Methods =====
    def _flush_to_parquet(self):
        """Flush les buffers en mémoire sur disque en mode append."""
        for path, df in [
            (PARQUET_PATH_REALTIME, self.metrics_realtime),
            (PARQUET_PATH_AGGREGATED, self.metrics_aggregated),
        ]:
            if len(df) == 0:
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
        """
        Save metrics at each timestep.
        fills: list of tuples (passive_order, qty) from add_limit_order on A.
        """
        self._step_count += 1
        mid_A = order_book_A.mid
        best_bid_A, _ = order_book_A.best_bid
        best_ask_A, _ = order_book_A.best_ask

        # Inventory regime
        abs_inventory = abs(self.EUR_quantity)
        inventory_pct = abs_inventory / self.q_max
        if inventory_pct < 0.70:
            hedge_regime = "Normal"
        elif inventory_pct < 0.90:
            hedge_regime = "Alert"
        else:
            hedge_regime = "Hedge"

        # PnL MtM
        mtm_pnl = self.EUR_quantity * mid_A + self.USD_quantity

        # === Realtime row : every step ===
        self.metrics_realtime = self.metrics_realtime.vstack(
            pl.DataFrame([{
                "timestamp": timestamp,
                "mid_A": mid_A,
                "EUR_quantity": self.EUR_quantity,
                "USD_quantity": self.USD_quantity,
                "inventory_pct": inventory_pct,
                "hedge_regime": hedge_regime,
                "mtm_pnl": mtm_pnl,
            }], schema=self.metrics_realtime.schema)
        )

        # === Aggregated row : every 100 steps ===
        if self._step_count % 100 == 0:

            utility_problem = self._build_utility_problem(
                mid_B=order_book_B.mid,  
                mid_C=order_book_C.mid,
            )

            # Fills du MM
            mm_fills = [(o, q) for (o, q) in fills if o.order_id.startswith("MM_")]
            bid_fills = [(o, q) for (o, q) in mm_fills if o.side == 'bid']
            ask_fills = [(o, q) for (o, q) in mm_fills if o.side == 'ask']

            total_bid_qty = sum(q for _, q in bid_fills)
            total_ask_qty = sum(q for _, q in ask_fills)
            fill_rate_bid = total_bid_qty / self.q_max if bid_fills else 0.0
            fill_rate_ask = total_ask_qty / self.q_max if ask_fills else 0.0

            # Spread capture : distance between execution price and mid
            spread_capture = (
                sum(abs(o.price - mid_A) * q for o, q in mm_fills) / sum(q for _, q in mm_fills)
                if mm_fills else 0.0
            )

            # Adverse selection : même calcul — sera affiné avec mid post-fill
            adverse_selection = spread_capture

            # Realized PnL accumulated
            self._realized_pnl += sum(
                q * o.price * (1 if o.side == 'ask' else -1)
                for o, q in mm_fills
            )

            # Hedge cost accumulated
            hedge_fills = [(o, q) for (o, q) in fills if o.order_id.startswith("hedge")]
            self._hedge_cost += sum(
                q * o.price * FEES_TAKER_B
                for o, q in hedge_fills
            )

            # Arb opportunities : number of HFT orders as a proxy
            arb_opportunity_count = hft_snipe_count
            arb_opportunity_size = hft_snipe_qty

            self.metrics_aggregated = self.metrics_aggregated.vstack(
                pl.DataFrame([{
                    "timestamp": timestamp,
                    "mid_A": mid_A,
                    "reservation_price": utility_problem.reservation_price,
                    "optimal_spread": utility_problem.optimal_spread,
                    "spread_quoted": best_ask_A - best_bid_A,
                    "best_bid_A": best_bid_A,
                    "best_ask_A": best_ask_A,
                    "EUR_quantity": self.EUR_quantity,
                    "USD_quantity": self.USD_quantity,
                    "inventory_pct": inventory_pct,
                    "hedge_regime": hedge_regime,
                    "mtm_pnl": mtm_pnl,
                    "realized_pnl": self._realized_pnl,
                    "hedge_cost": self._hedge_cost,
                    "fill_rate_bid": fill_rate_bid,
                    "fill_rate_ask": fill_rate_ask,
                    "spread_capture": spread_capture,
                    "adverse_selection": adverse_selection,
                    "hft_snipe_count": hft_snipe_count,
                    "hft_snipe_qty": hft_snipe_qty,
                    "arb_opportunity_count": arb_opportunity_count,
                    "arb_opportunity_size": arb_opportunity_size,
                }], schema=self.metrics_aggregated.schema)
            )

        # === Flush every FLUSH_INTERVAL steps ===
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

    def make_market(self, order_book_A: OrderBook, order_book_B: OrderBook, order_book_C: OrderBook)-> None:
        # Pass limit orders on A given the state of B 200ms ago and C 170ms ago

        utility_problem = self._build_utility_problem(
            mid_B=order_book_B.mid,
            mid_C=order_book_C.mid,
        )

        bids_prices, ask_prices = utility_problem.get_price_grid()
        bids_qty, ask_qty = utility_problem.get_qty_grid()

        # create list of orders to pass to the order book A
        order_to_A = []

        #bids
        for i, price in enumerate(bids_prices):
            order_to_A.append(Order(
                order_id = f"MM_{self.id_cpt}",
                side = "bid",
                price = price,
                quantity = bids_qty[i]
            ))
            self.id_cpt += 1
        
        #asks
        for i, price in enumerate(ask_prices):
            order_to_A.append(Order(
                order_id = f"MM_{self.id_cpt}",
                side = "ask",
                price = price,
                quantity = ask_qty[i]
            ))
            self.id_cpt += 1

        # add orders in list to A
        order_book_A.add_limit_order_list(order_to_A)

        


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