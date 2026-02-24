from OrderBook import OrderBook, Level

class MarketSimulator:
    def __init__(self, order_book: OrderBook):
        self.order_book = order_book

    def get_midpoint(self) -> float:
        return self.order_book.get_midpoint()

    def get_spread(self) -> float:
        return self.order_book.get_spread()

    def get_best_bid(self) -> Level:
        return self.order_book.get_best_bid()

    def get_best_ask(self) -> Level:
        return self.order_book.get_best_ask()

    def simulate_single_step(self):
        pass

    def simulate_multiple_steps(self, steps: int):
        for _ in range(steps):
            self.simulate_single_step()