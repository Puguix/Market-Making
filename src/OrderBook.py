from collections import deque
from sortedcontainers import SortedDict
import time
from dataclasses import dataclass, field


@dataclass
class Order:
    order_id: str
    side: str          # 'bid' or 'ask'
    price: float
    quantity: float
    timestamp: float = field(default_factory=time.time)


class PriceLevel:
    """FIFO for orders at a certain price — price-time priority."""

    def __init__(self):
        self.queue: deque[Order] = deque()
        self.total_qty: float = 0.0

    def add(self, order: Order):
        self.queue.append(order)
        self.total_qty += order.quantity

    def cancel(self, order_id: str) -> bool:
        for i, o in enumerate(self.queue):
            if o.order_id == order_id:
                self.total_qty -= o.quantity
                del self.queue[i]
                return True
        return False

    def consume(self, qty: float) -> list[tuple[Order, float]]:
        """Fills orders by priority (head first) until qty is empty. Return fills"""
        fills = []
        remaining = qty
        while self.queue and remaining > 0:
            head = self.queue[0]
            fill_qty = min(head.quantity, remaining)
            fills.append((head, fill_qty))
            remaining -= fill_qty
            head.quantity -= fill_qty
            self.total_qty -= fill_qty
            if head.quantity <= 0:
                self.queue.popleft()
        return fills

    def __bool__(self):
        return bool(self.queue)


class OrderBook:
    """
    Asks : SortedDict: price -> PriceLevel (best ask in head)
    Bids : SortedDict: price -> PriceLevel (best bid in head)
    """

    def __init__(self):
        self.asks: SortedDict = SortedDict()
        self.bids: SortedDict = SortedDict(lambda p: -p)
        self._orders: dict[str, Order] = {}  # id → order

    def add_limit_order(self, order: Order) -> list[tuple[Order, float]]:
        fills = []
        if order.side == 'bid':
            fills = self._match(order, self.asks, lambda ask_p: ask_p <= order.price)
            if order.quantity > 0:
                self._insert(order, self.bids)
        else:
            fills = self._match(order, self.bids, lambda bid_p: bid_p >= order.price)
            if order.quantity > 0:
                self._insert(order, self.asks)
        return fills

    def add_market_order(self, side: str, qty: float) -> list[tuple[Order, float]]:
        dummy = Order("__market__", side, float('inf') if side == 'bid' else 0.0, qty)
        return self.add_limit_order(dummy)

    def _insert(self, order: Order, side: SortedDict):
        if order.price not in side:
            side[order.price] = PriceLevel()
        side[order.price].add(order)
        self._orders[order.order_id] = order

    def _match(self, aggressor: Order, opposite: SortedDict, crossable) -> list:
        fills = []
        for price in list(opposite.keys()):
            if not crossable(price) or aggressor.quantity <= 0:
                break
            level: PriceLevel = opposite[price]
            level_fills = level.consume(aggressor.quantity)
            for (passive, qty) in level_fills:
                aggressor.quantity -= qty
                fills.append((passive, qty))
                if passive.quantity <= 0:
                    self._orders.pop(passive.order_id, None)
            if not level:
                del opposite[price]
        return fills

    def cancel(self, order_id: str) -> bool:
        order = self._orders.pop(order_id, None)
        if not order:
            return False
        side = self.bids if order.side == 'bid' else self.asks
        if order.price in side:
            side[order.price].cancel(order_id)
            if not side[order.price]:
                del side[order.price]
        return True

    @property
    def best_bid(self):
        return next(iter(self.bids), None)

    @property
    def best_ask(self):
        return next(iter(self.asks), None)

    @property
    def spread(self):
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid

    @property
    def mid(self):
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2

    def snapshot(self, depth: int = 5) -> dict:
        bids = [(p, self.bids[p].total_qty) for p in list(self.bids)[:depth]]
        asks = [(p, self.asks[p].total_qty) for p in list(self.asks)[:depth]]
        return {"bids": bids, "asks": asks, "mid": self.mid, "spread": self.spread}

    def print_book(self, depth: int = 5):
        snap = self.snapshot(depth)
        print(f"\n{'─'*40}")
        print(f"mid={snap['mid']:.5f} | spread={snap['spread']:.5f}")
        print(f"{'─'*40}")
        print(f"  {'PRICE':>12}  {'QTY':>14}  SIDE")
        print(f"{'─'*40}")
        for price, qty in reversed(snap['asks']):
            print(f"  {price:>12.5f}  {qty:>14,.0f}  ASK")
        print(f"  {'--- mid ---':>28}")
        for price, qty in snap['bids']:
            print(f"  {price:>12.5f}  {qty:>14,.0f}  BID")
        print(f"{'─'*40}\n")


# Test
if __name__ == "__main__":

    ob = OrderBook()

    # Populate bids
    ob.add_limit_order(Order("B1", "bid", 1.08500, 1_000_000))
    ob.add_limit_order(Order("B2", "bid", 1.08500,   500_000))
    ob.add_limit_order(Order("B3", "bid", 1.08480, 2_000_000))
    ob.add_limit_order(Order("B4", "bid", 1.08460, 3_000_000))
    ob.add_limit_order(Order("B5", "bid", 1.08440, 1_500_000))
    ob.add_limit_order(Order("B6", "bid", 1.08420, 2_500_000))

    # Populate asks
    ob.add_limit_order(Order("A1", "ask", 1.08520,   800_000))
    ob.add_limit_order(Order("A2", "ask", 1.08540, 1_500_000))
    ob.add_limit_order(Order("A3", "ask", 1.08560, 2_000_000))
    ob.add_limit_order(Order("A4", "ask", 1.08580, 1_000_000))
    ob.add_limit_order(Order("A5", "ask", 1.08600, 3_000_000))

    print("=== Book initial ===")
    ob.print_book()

    # Cancel an order
    print(">>> cancel B3")
    ob.cancel("B3")
    ob.print_book()

    # Limit order that matches B1 (FIFO : B1 before B2)
    print(">>> Limit ask 1.08500 pour 900k (matche B1 en FIFO)")
    fills = ob.add_limit_order(Order("X1", "ask", 1.08500, 900_000))
    for passive, qty in fills:
        print(f"    fill: order {passive.order_id} @ {passive.price:.5f} x {qty:,.0f}")
    ob.print_book()

    # Market order buy
    print(">>> Market buy 2M")
    fills = ob.add_market_order("bid", 2_000_000)
    for passive, qty in fills:
        print(f"    fill: order {passive.order_id} @ {passive.price:.5f} x {qty:,.0f}")
    ob.print_book()
