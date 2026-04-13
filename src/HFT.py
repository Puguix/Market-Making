from OrderBook import OrderBook


class HFT:
    """
    The High Frequency Traders that first arbitrage the price difference between exchanges.
    In phase 3, they make market on A.
    """

    def __init__(self):
        pass

    def snipe(self, order_book_A: OrderBook, order_book_B: OrderBook, order_book_C: OrderBook):
        # Pass orders based on current state of A, and states of B and C 50ms ago
        pass