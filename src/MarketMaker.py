import polars as pl
from polars import col as c

from OrderBook import OrderBook

class MarketMaker:
    """
    The market maker for exchange A.
    Keep trak of inventory and PnL.
    """

    def __init__(self, EUR_quantity: float, USD_quantity: float):
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