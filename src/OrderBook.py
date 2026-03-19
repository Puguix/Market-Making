from collections import deque
from sortedcontainers import SortedDict
import time
import math
import random
from dataclasses import dataclass, field

from PoissonSimulation import ArrivalIntensity, PoissonGenerator



# %%%%%%%%%%%%%%%% OBJECTS %%%%%%%%%%%%%%%%%%%

@dataclass
class Order:
    order_id: str
    side: str          # 'bid' or 'ask'
    price: float
    quantity: float
    timestamp: float = field(default_factory=time.time)


class PriceLevel:
    """
    Price level implementation. Tracking order at each price level and their coming order on FIFO base.

    Attributes:
        queue : queue of oders
        total_qty: the total quantity of orders at the price level
    """

    def __init__(self):
        self.queue: deque[Order] = deque()
        self.total_qty: float = 0.0

    def add(self, order: Order)->None:
        """
        Adding an Order to the price level.
        """
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
        """
        Method that consume a certain quatity at this price level by FIFO.

        Parameters:
            qty (float): quatity of order to consume at this price level.
        
        Returns:
            a List of tuples containing an Order and the filled quantity for each.
        """
        fills = []
        remaining = qty
        # as long as there is order at the price level and the quantity is not filled.
        while self.queue and remaining > 0:
            head = self.queue[0]
            fill_qty = min(head.quantity, remaining)
            fills.append((head, fill_qty))
            remaining -= fill_qty
            head.quantity -= fill_qty
            self.total_qty -= fill_qty

            # if No more quantity for this order, go to next in the queue
            if head.quantity <= 0:
                self.queue.popleft()
        return fills

    def __bool__(self):
        return bool(self.queue)


class OrderBook:
    """
    A class for Order Book Implementation.

    Attributes:
        asks (SortedDict): A dictionnary of asks oders always sorted.
        bids (SortedDict): The same
    """

    def __init__(self):
        self.asks: SortedDict = SortedDict() # empty sorted dict
        self.bids: SortedDict = SortedDict(lambda p: -p) # invertly sorted dict (we want to see higher prices first for bids)
        self._orders: dict[str, Order] = {}  # id → order

    # === protected methods ===
    def _insert(self, order: Order, side: SortedDict[float, PriceLevel])->None:
        """
        Insert an order (side) in the order book by modifying directly "side"
        """
        if order.price not in side:
            side[order.price] = PriceLevel()
        side[order.price].add(order)
        self._orders[order.order_id] = order

    def _match(self, aggressor: Order, opposite: SortedDict, crossable) -> list:
        """
        
        """
        fills = []
        # loop on orderbook prices (keys)
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
 
    def _shift_prices(self, delta: float)->None:
        if not delta:
            return

        def shift_side(side_dict: SortedDict, key_func=None) -> SortedDict:
            if not side_dict:
                return side_dict
            items = list(side_dict.items())
            new_side = SortedDict(key_func) if key_func else SortedDict()
            for price, level in items:
                new_price = price + delta
                for o in level.queue:
                    o.price = new_price
                if new_price in new_side:
                    target_level: PriceLevel = new_side[new_price]
                    target_level.queue.extend(level.queue)
                    target_level.total_qty += level.total_qty
                else:
                    new_side[new_price] = level
            return new_side

        self.bids = shift_side(self.bids, lambda p: -p)
        self.asks = shift_side(self.asks)

    # === Classic methods ===
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

    def evolve_one_step(
        self,
        new_mid: float,
        dt: float,
        lambda_a0: float,
        alpha: float,
        theta: float,
        lambda_mo: float,
        v_unit: float,
        mo_buy_prob: float = 0.5,
    ):
        """
        Discrete-time evolution of the synthetic order book around a given mid-price,
        following the queue dynamics of Section 2.2.
        """
        if self.best_bid is None or self.best_ask is None:
            return

        current_mid = self.mid
        if current_mid is not None and new_mid is not None:
            delta_mid = new_mid - current_mid
            if delta_mid:
                self._shift_prices(delta_mid)

        # Limit order arrivals and cancellations at each level
        p_cancel = 1.0 - math.exp(-theta * dt)

        for side_name, side_dict in (("bid", self.bids), ("ask", self.asks)):
            for price, level in list(side_dict.items()):
                # Limit order arrivals λ_a(δ) = λ_a0 * exp(-α δ), δ in pips,
                # simulated with a Poisson generator as in PoissonSimulation.
                delta_pips = abs(price - new_mid) * 10_000.0
                arr_int = ArrivalIntensity(
                    spread=delta_pips,
                    alpha=alpha,
                    lambda_0=lambda_a0 * dt,  # expected arrivals over this dt at δ=0
                )
                n_arrivals = PoissonGenerator(arr_int).generate()
                for _ in range(n_arrivals):
                    order_id = f"sim_LO_{side_name}_{time.time_ns()}"
                    order = Order(
                        order_id=order_id,
                        side=side_name,
                        price=price,
                        quantity=v_unit,
                    )
                    self._insert(order, side_dict)

                # Cancellations: each resting order independently cancelled with p_cancel
                to_cancel = []
                for o in list(level.queue):
                    if random.random() < p_cancel:
                        to_cancel.append(o.order_id)
                for oid in to_cancel:
                    self.cancel(oid)

        # Market order arrivals at best levels with intensity λ_MO,
        # also simulated with the Poisson generator.
        mo_int = ArrivalIntensity(
            spread=0.0,
            alpha=0.0,
            lambda_0=lambda_mo * dt,
        )
        n_mo = PoissonGenerator(mo_int).generate()
        for _ in range(n_mo):
            side = "bid" if random.random() < mo_buy_prob else "ask"
            self.add_market_order(side, v_unit)

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

# %%%%%%%%%%%%% Func %%%%%%%%%%%%%%
def test_basic():
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


def test_advanced():
    random.seed(42)

    ob = OrderBook()
    base_mid = 1.08500
    tick = 0.0001  # 1 pip

    # Build a deep book: many levels on each side around base_mid
    level_count = 50
    for i in range(1, level_count + 1):
        bid_price = base_mid - i * tick
        ask_price = base_mid + i * tick

        bid_qty = random.randint(1, 10) * 100_000
        ask_qty = random.randint(1, 10) * 100_000

        ob.add_limit_order(Order(f"HB_B{i}", "bid", bid_price, bid_qty))
        ob.add_limit_order(Order(f"HB_A{i}", "ask", ask_price, ask_qty))

    print("=== Heavy book initial snapshot ===")
    ob.print_book(depth=10)

    # Simulate several evolution steps with a slowly drifting mid-price
    dt = 0.01
    lambda_a0 = 5.0
    alpha = 0.05
    theta = 0.1
    lambda_mo = 2.0
    v_unit = 100_000

    new_mid = base_mid
    for step in range(100):
        # Small random walk in mid-price (for testing)
        new_mid += 0.00001 * random.gauss(0.0, 1.0)
        ob.evolve_one_step(
            new_mid=new_mid,
            dt=dt,
            lambda_a0=lambda_a0,
            alpha=alpha,
            theta=theta,
            lambda_mo=lambda_mo,
            v_unit=v_unit,
            mo_buy_prob=0.5,
        )
        if (step + 1) % 25 == 0:
            print(f"=== Snapshot after {step + 1} steps ===")
            ob.print_book(depth=10)




# %%%%%%% TEST %%%%%%%%%%
if __name__ == "__main__":
    print(">>> Running basic order book test")
    # test_basic()

    print(">>> Running advanced book evolution test")
    test_advanced()

