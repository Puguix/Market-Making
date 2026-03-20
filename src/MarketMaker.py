import polars as pl
from polars import col as c
from OrderBook import OrderBook
import math


class UtilityProblem:

    def __init__(self,
                 gamma: float,        
                 sigma: float,
                 remaining_time: float,
                 inventory: float,
                 ref_price: float,
                 kappa: float,
                 latency: float,
                 fees_pips: float = 2.0,
                 ):
        
        self.gamma = gamma
        self.sigma = sigma
        self.T_minus_t = remaining_time
        self.inventory = inventory
        self.ref_price = ref_price
        self.kappa = kappa
        self.latency = latency
        self.fees_pips = fees_pips

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

    def make_market(self, order_book_A: OrderBook, order_book_B: OrderBook, order_book_C: OrderBook):
        # Output the list of orders to submit or cancel given the 3 order books
        # and the inventory
        pass