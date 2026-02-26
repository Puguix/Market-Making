from dataclasses import dataclass

@dataclass
class Level:
    price: float
    quantity: float

class OrderBook:
    def __init__(self, levels: int = 10):
        self.levels = levels
        self.bids = []
        self.asks = []

    def init_dummy_order_book(self, price: float = 100, quantity: float = 100):
        self.bids = [Level(price - (i + 1), quantity//self.levels) for i in range(self.levels)]
        self.asks = [Level(price + (i + 1), quantity//self.levels) for i in range(self.levels)]

    def display(self):
        print(f"{'BIDS':>18} | {'ASKS':<18}")
        print(f"{'Qty':>8} {'Price':>8} | {'Price':<8} {'Qty':<8}")
        print("-" * 38)
        bid_levels = [level for level in self.bids if level.quantity > 0]
        ask_levels = [level for level in self.asks if level.quantity > 0]
        max_levels = max(len(bid_levels), len(ask_levels))
        for i in range(max_levels):
            bid_str = ""
            ask_str = ""
            if i < len(bid_levels):
                bid = bid_levels[i]
                bid_str = f"{bid.quantity:8.2f} {bid.price:8.2f}"
            else:
                bid_str = " " * 17
            if i < len(ask_levels):
                ask = ask_levels[i]  # Display asks from highest to lowest
                ask_str = f"{ask.price:<8.2f} {ask.quantity:<8.2f}"
            print(f"{bid_str} | {ask_str}")
        print(f"\nMid: {self.get_midpoint():.2f}  Spread: {self.get_spread():.2f}")
        print()

    def get_best_bid(self) -> Level:
        return self.bids[0]
    
    def get_best_ask(self) -> Level:
        return self.asks[0]
    
    def get_midpoint(self) -> float:
        return (self.get_best_bid().price + self.get_best_ask().price) / 2
    
    def get_spread(self) -> float:
        return self.get_best_ask().price - self.get_best_bid().price

    def add_limit_order(self, price: float, quantity: float, is_bid: bool):
        if is_bid:
            # Hit asks
            while price > self.get_best_ask().price and quantity > 0:
                quantity_taken = min(quantity, self.get_best_ask().quantity)
                quantity -= quantity_taken
                self.asks[0].quantity -= quantity_taken
                if self.get_best_ask().quantity == 0:
                    self.asks.pop(0)
            if quantity > 0:
                # Add to bids
                for i in range(len(self.bids)):
                    if self.bids[i].price < price:
                        self.bids.insert(i, Level(price, quantity))
                        break
                    elif self.bids[i].price == price:
                        self.bids[i].quantity += quantity
                        break
                    elif i == len(self.bids) - 1:
                        self.bids.append(Level(price, quantity))
                        break
                # Trim bids to number of levels
                while self.bids.length > self.levels:
                    self.bids.pop(-1)
        else:
            # Hit bids
            while price < self.get_best_bid().price and quantity > 0:
                quantity_taken = min(quantity, self.get_best_bid().quantity)
                quantity -= quantity_taken
                self.bids[0].quantity -= quantity_taken
                if self.get_best_bid().quantity == 0:
                    self.bids.pop(0)
            if quantity > 0:
                # Add to asks
                for i in range(len(self.asks)):
                    if self.asks[i].price > price:
                        self.asks.insert(i, Level(price, quantity))
                        break
                    elif self.asks[i].price == price:
                        self.asks[i].quantity += quantity
                        break
                    elif i == len(self.asks) - 1:
                        self.asks.append(Level(price, quantity))
                        break
                # Trim bids to number of levels
                while self.asks.length > self.levels:
                    self.asks.pop(-1)
    
    def add_market_order(self, quantity: float, is_bid: bool):
        if is_bid:
            while quantity > 0:
                quantity_taken = min(quantity, self.get_best_ask().quantity)
                quantity -= quantity_taken
                self.asks[0].quantity -= quantity_taken
                if self.get_best_ask().quantity == 0:
                    self.asks.pop(0)
        else:
            while quantity > 0:
                quantity_taken = min(quantity, self.get_best_bid().quantity)
                quantity -= quantity_taken
                self.bids[0].quantity -= quantity_taken
                if self.get_best_bid().quantity == 0:
                    self.bids.pop(0)

    def cancel_order(self, price: float, quantity: float, is_bid: bool):
        if is_bid:
            for level in self.bids:
                if level.price == price:
                    level.quantity = max(0, level.quantity - quantity)
                    break
        else:
            for level in self.asks:
                if level.price == price:
                    level.quantity = max(0, level.quantity - quantity)
                    break