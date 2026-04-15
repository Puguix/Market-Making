import polars as pl
from polars import col as c
from OrderBook import OrderBook, Order, PriceLevel
import math
from abc import ABC, abstractmethod 
import time

FEES_TAKER_B = 0.0002
FEES_TAKER_C = 0.0003

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
        self.max_levels = max_levels
        self.tick_size = tick_size

    def generate(self, problem: "UtilityProblem") -> tuple[list[float], list[float]]:

        ref_bid = problem.best_bid
        ref_ask = problem.best_ask
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

        for k in range(0, int(self.max_levels/2)): # quote 5 levels each side
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
        self.metrics = pl.DataFrame(schema={
            "EUR/USD": pl.Float64,
            "EUR_quantity": pl.Float64,
            "USD_quantity": pl.Float64,
            "PnL": pl.Float64,
            "avg_PnL": pl.Float64,
            "median_PnL": pl.Float64,
            "5th_percentile_PnL": pl.Float64,
            "95th_percentile_PnL": pl.Float64,
        })

        self.EUR_quantity = EUR_quantity
        self.USD_quantity = USD_quantity
        self.gamma = gamma
        self.sigma = sigma
        self.kappa = kappa
        self.T = T
        self.q_max = q_max
        self._t = 0.0

        # order id compteur
        self.id_cpt = 0

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
    
    # === calssic methods ===
    def get_EUR_quantity(self) -> float:
        return self.metrics.tail(1)["EUR_quantity"].item()

    def get_USD_quantity(self) -> float:
        return self.metrics.tail(1)["USD_quantity"].item()

    def compute_metrics(self, order_book: OrderBook) -> float:
        # Inventory value in EURO
        inventory_value = self.get_EUR_quantity() + self.get_USD_quantity() / order_book.get_midpoint()
        # PnL
        PnL = inventory_value - self.metrics.tail(1)["PnL"].item()
        # Avg PnL
        avg_PnL = self.metrics.tail(1)["avg_PnL"].item()
        # Median PnL
        median_PnL = self.metrics.tail(1)["median_PnL"].item()
        # 5th percentile PnL
        fifth_percentile_PnL = self.metrics.tail(1)["5th_percentile_PnL"].item()
        # 95th percentile PnL
        ninety_fifth_percentile_PnL = self.metrics.tail(1)["95th_percentile_PnL"].item()
        # Save metrics
        self.metrics = self.metrics.extend({
            "EUR_quantity": self.get_EUR_quantity(),
            "USD_quantity": self.get_USD_quantity(),
            "PnL": PnL,
            "avg_PnL": avg_PnL,
            "median_PnL": median_PnL,
            "5th_percentile_PnL": fifth_percentile_PnL,
            "95th_percentile_PnL": ninety_fifth_percentile_PnL,
        })

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