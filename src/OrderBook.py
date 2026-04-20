from collections import OrderedDict
from sortedcontainers import SortedDict
import math
import random
import numpy as np
from time import perf_counter
from dataclasses import dataclass, field
from typing import Optional

from PoissonSimulation import ArrivalIntensity, PoissonGenerator

from config import (
    LAMBDA_A0_B, ALPHA_B, THETA_B, LAMBDA_MO_B, V_UNIT_B,
    LAMBDA_A0_C, ALPHA_C, THETA_C, LAMBDA_MO_C, V_UNIT_C,
    ORGANIC_BOOK_DEPTH_LEVELS, ORGANIC_BOOK_LEVEL_QTY, DEFAULT_TICK_SIZE,
    ORDERBOOK_DEFAULT_LEVELS, ORDERBOOK_DEFAULT_TICK_SIZE,
    ORDERBOOK_DEFAULT_MO_BUY_PROB, ORDERBOOK_ADVANCED_TEST_LEVEL_COUNT,
    ORDERBOOK_ADVANCED_TEST_STEPS, ORDERBOOK_ADVANCED_TEST_SNAPSHOT_INTERVAL,
    ORDERBOOK_ADVANCED_TEST_RANDOM_SEED, ORDERBOOK_ADVANCED_TEST_DT,
    ORDERBOOK_ADVANCED_TEST_DRIFT_SCALE, ORDERBOOK_ADVANCED_TEST_BOOK_DEPTH
)

# %%%%%%%%%%%%%%%% OBJECTS %%%%%%%%%%%%%%%%%%%

@dataclass
class Order:
    order_id: str
    is_ask: bool
    price: float
    quantity: float
    # timestamp: float = field(default_factory=time.time)


class PriceLevel:
    """
    Price level implementation. Tracking order at each price level and their coming order on FIFO base.

    Attributes:
        queue : queue of oders
        total_qty: the total quantity of orders at the price level
    """

    def __init__(self):
        # FIFO storage with O(1) cancellation by order_id.
        self.queue: OrderedDict[str, Order] = OrderedDict()
        self.total_qty: float = 0.0

    def add(self, order: Order)->None:
        """
        Adding an Order to the price level.
        """
        self.queue[order.order_id] = order
        self.total_qty += order.quantity

    def cancel(self, order_id: str) -> bool:
        order = self.queue.pop(order_id, None)
        if order is None:
            return False
        self.total_qty -= order.quantity
        return True

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
            head = next(iter(self.queue.values()))
            fill_qty = min(head.quantity, remaining)
            fills.append((head, fill_qty))
            remaining -= fill_qty
            head.quantity -= fill_qty
            self.total_qty -= fill_qty

            # if No more quantity for this order, go to next in the queue
            if head.quantity <= 0:
                self.queue.popitem(last=False)
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

    def __init__(self, lambda_a0: float, alpha: float, theta: float, lambda_mo: float, v_unit: float):
        self.asks: SortedDict = SortedDict() # empty sorted dict
        self.bids: SortedDict = SortedDict(lambda p: -p) # invertly sorted dict (we want to see higher prices first for bids)
        self._orders: dict[str, Order] = {}  # id → order
        self._order_id_seq: int = 0
        self.lambda_a0 = lambda_a0
        self.alpha = alpha
        self.theta = theta
        self.lambda_mo = lambda_mo
        self.v_unit = v_unit

    def _new_order_id(self, prefix: str) -> str:
        self._order_id_seq += 1
        return f"{prefix}_{self._order_id_seq}"

    def copy(self) -> "OrderBook":
        """Structural copy for ring-buffer evolution (avoids copy.deepcopy)."""
        ob = OrderBook(self.lambda_a0, self.alpha, self.theta, self.lambda_mo, self.v_unit)
        ob._order_id_seq = self._order_id_seq
        for _price, level in self.bids.items():
            for o in level.queue.values():
                ob._insert(Order(o.order_id, o.is_ask, o.price, o.quantity), ob.bids)
        for _price, level in self.asks.items():
            for o in level.queue.values():
                ob._insert(Order(o.order_id, o.is_ask, o.price, o.quantity), ob.asks)
        return ob

    def build_organic_book(
        self,
        mid: float,
        depth_levels: int = ORGANIC_BOOK_DEPTH_LEVELS,
        level_qty: float = ORGANIC_BOOK_LEVEL_QTY,
        tick_size: float = DEFAULT_TICK_SIZE,
    ) -> "OrderBook":
        for i in range(1, depth_levels + 1):
            self.add_limit_order(Order(f"B{i}", False, mid - i * tick_size, level_qty))
            self.add_limit_order(Order(f"A{i}", True, mid + i * tick_size, level_qty))
        return self

    # === protected methods ===
    def _insert(self, order: Order, side: SortedDict[float, PriceLevel])->None:
        """
        Insert an order (side) in the order book by modifying directly "side"
        """
        if order.price not in side:
            side[order.price] = PriceLevel()
        side[order.price].add(order)
        self._orders[order.order_id] = order

    def _match(self, aggressor: Order, opposite: SortedDict, crossable) -> list: # crossable is a function that return a boolean
        """
        If the limit order is crossing the opposite side, we match the LO with the other LO to execute the trades.

        Attributes:
            aggressor: the LO
            opposite: SortedDict of bids or ask orders in the order book
            crossable: a function that return true if our order cross the opposite side

        Return:
            A list of tuples containing the orders at which we executed our aggressor LO and the quantity.
        """
        fills = []
        # loop on orderbook prices (keys)
        for price in list(opposite.keys()):
            if not crossable(price) or aggressor.quantity <= 0: # if crossable we break, if all qty filled we break
                break
            level: PriceLevel = opposite[price]
            level_fills = level.consume(aggressor.quantity) # return (Orders : filled qty at each order, for this price level)
            for (passive, qty) in level_fills: # (passive= the orders, qty= quantity filled)
                aggressor.quantity -= qty
                fills.append((passive, qty))
                if passive.quantity <= 0:
                    self._orders.pop(passive.order_id, None)
            if not level:
                del opposite[price]
        return fills
 
    def _shift_prices(self, delta: float)->None:
        """
        Method that shift the order book for a certain delta in the mid price variation.
        """

        if not delta:
            return

        def shift_side(side_dict: SortedDict, key_func=None) -> SortedDict:
            if not side_dict:
                return side_dict
            items = list(side_dict.items())
            new_side = SortedDict(key_func) if key_func else SortedDict()
            for price, level in items:
                # Keep all quoted levels on a 4-decimal grid after mid shifts.
                new_price = round(float(price + delta), 4)
                for o in level.queue.values():
                    o.price = new_price
                if new_price in new_side:
                    target_level: PriceLevel = new_side[new_price]
                    target_level.queue.update(level.queue)
                    target_level.total_qty += level.total_qty
                else:
                    new_side[new_price] = level
            return new_side

        self.bids = shift_side(self.bids, lambda p: -p)
        self.asks = shift_side(self.asks)

    # === Classic methods ===
    def add_limit_order(self, order: Order) -> list[tuple[Order, float]]:

        """
        A method that enable to add an order in the order book.

        Attributes:
            order (Oder): The LO you want to add in the order book.

        Returns:
            The filled order if any (if there is no crossable order it should return an empty list)
        """
        fills = []
        if order.is_ask:
            fills = self._match(order, self.bids, lambda bid_p: bid_p >= order.price)
            if order.quantity > 0:
                self._insert(order, self.asks)
        else:
            fills = self._match(order, self.asks, lambda ask_p: ask_p <= order.price) # checking if there are prices to cross
            if order.quantity > 0: # still order to place ?
                self._insert(order, self.bids) # insert bid
        return fills

    def add_market_order(self, is_ask: bool, qty: float) -> list[tuple[Order, float]]:
        dummy = Order("__market__", is_ask, float('inf') if not is_ask else 0.0, qty)
        if is_ask:
            return self._match(dummy, self.bids, lambda bid_p: bid_p >= dummy.price)
        else:
            return self._match(dummy, self.asks, lambda ask_p: ask_p <= dummy.price)

    def add_market_order_from_LO(self, order: Order) -> list[tuple[Order, float]]:
        return self._match(order, self.asks if order.is_ask else self.bids, lambda p: p >= order.price if order.is_ask else p <= order.price)

    def cancel(self, order_id: str) -> bool:
        order = self._orders.get(order_id)
        if not order:
            return False
        side = self.asks if order.is_ask else self.bids
        level = side.get(order.price)
        if level is None:
            return False

        cancelled = level.cancel(order_id)
        if not cancelled:
            return False

        self._orders.pop(order_id, None)
        if not level:
            del side[order.price]
        return True

    def resting_quantity(self, order_id: str) -> float:
        """Remaining visible quantity for a resting limit order id, or 0 if absent."""
        o = self._orders.get(order_id)
        return float(o.quantity) if o else 0.0

    def resting_price(self, order_id: str) -> Optional[float]:
        """Limit price for a resting order id, or None if not in the book."""
        o = self._orders.get(order_id)
        return float(o.price) if o else None

    @property
    def best_bid(self):
        if not self.bids:
            return None, 0.0
        best_price = next(iter(self.bids))
        return best_price, self.bids[best_price].total_qty

    @property
    def best_ask(self):
        if not self.asks:
            return None, 0.0
        best_price = next(iter(self.asks))
        return best_price, self.asks[best_price].total_qty

    @property
    def spread(self):
        bid = self.best_bid[0]
        ask = self.best_ask[0]
        if bid is None or ask is None:
            return None
        return ask - bid

    @property
    def mid(self):
        bid = self.best_bid[0]
        ask = self.best_ask[0]
        if bid is None or ask is None:
            return None
        return (bid + ask) / 2

    def evolve_one_step(
        self,
        new_mid: float,
        dt: float,
        mo_buy_prob: float = ORDERBOOK_DEFAULT_MO_BUY_PROB,
        n_levels: int = ORDERBOOK_DEFAULT_LEVELS,
        tick: float = ORDERBOOK_DEFAULT_TICK_SIZE,
        _timing: Optional[dict] = None,
    ) -> tuple["OrderBook", list]:
        _t = perf_counter()

        def _lap(section: str) -> None:
            nonlocal _t
            if _timing is None:
                return
            now = perf_counter()
            _timing[section] = (now - _t) * 1000.0
            _t = now

        # 1. Shift all existing prices to follow the new mid
        current_mid = self.mid
        if current_mid is not None and new_mid is not None:
            delta_mid = new_mid - current_mid
            if delta_mid:
                self._shift_prices(delta_mid)
        _lap("1_shift_prices")

        # 2. Prune levels too far from new_mid to keep book size bounded
        max_depth = n_levels * tick
        for price in list(self.bids.keys()):
            if new_mid - price > max_depth:
                for o in self.bids[price].queue.values():
                    self._orders.pop(o.order_id, None)
                del self.bids[price]
        for price in list(self.asks.keys()):
            if price - new_mid > max_depth:
                for o in self.asks[price].queue.values():
                    self._orders.pop(o.order_id, None)
                del self.asks[price]
        _lap("2_prune_far_levels")

        # 3. Cancellations on existing levels
        p_cancel = 1.0 - math.exp(-self.theta * dt)
        for side_name, side_dict in (("bid", self.bids), ("ask", self.asks)):
            for price, level in list(side_dict.items()):
                to_cancel = [o.order_id for o in list(level.queue.values()) if random.random() < p_cancel]
                for oid in to_cancel:
                    self.cancel(oid)
        _lap("3_cancellations")

        # 4. LO arrivals on the fixed grid around new_mid
        # Batched Poisson: same intensity as ArrivalIntensity(spread=k, ...) with lambda_0 = lambda_a0 * dt,
        # i.e. mu_k = (lambda_a0 * dt) * exp(-alpha * k). Bid and ask are independent draws per level k.
        lo_bid_inserts = 0
        lo_ask_inserts = 0
        if n_levels > 0:
            mu_scale = self.lambda_a0 * dt
            ks = np.arange(1, n_levels + 1, dtype=np.float64)
            mus = mu_scale * np.exp(-self.alpha * ks)
            n_bids = np.random.poisson(mus)
            n_asks = np.random.poisson(mus)
            for idx, k in enumerate(range(1, n_levels + 1)):
                bid_price = round(new_mid - k * tick, 4)
                ask_price = round(new_mid + k * tick, 4)
                n_arr_bid = int(n_bids[idx])
                n_arr_ask = int(n_asks[idx])
                lo_bid_inserts += n_arr_bid
                for _ in range(n_arr_bid):
                    self._insert(Order(self._new_order_id("sim_LO_bid"), False, bid_price, self.v_unit), self.bids)
                lo_ask_inserts += n_arr_ask
                for _ in range(n_arr_ask):
                    self._insert(Order(self._new_order_id("sim_LO_ask"), True, ask_price, self.v_unit), self.asks)
        _lap("4_lo_grid_arrivals")
        if _timing is not None:
            _timing["count_lo_bid_inserts"] = lo_bid_inserts
            _timing["count_lo_ask_inserts"] = lo_ask_inserts

        # 5. Market order arrivals
        current_spread = self.spread
        if current_spread is None:
            print(f"[OrderBook] WARNING: spread is None (book one-sided or empty), falling back to 1 tick. "
                f"best_bid={self.best_bid[0]}, best_ask={self.best_ask[0]}")
            current_spread = tick
        n_mo = PoissonGenerator(ArrivalIntensity(spread=current_spread, alpha=self.alpha, lambda_0=self.lambda_mo * dt)).generate()
        _lap("5a_mo_poisson_draw")

        mo_fills = []
        for _ in range(n_mo):
            mo_fills.extend(self.add_market_order(random.random() > mo_buy_prob, self.v_unit))
        _lap("5b_mo_execute_loop")
        if _timing is not None:
            _timing["count_mo"] = n_mo

        # 6. Empty or one-sided book: Poisson LOs can leave a side empty; mid is then undefined.
        #    Seed inner quotes around new_mid (same grid as step 4), respecting any existing top of book.
        if new_mid is not None:
            inner_bid = round(new_mid - tick, 4)
            inner_ask = round(new_mid + tick, 4)
            bid_px, _ = self.best_bid
            ask_px, _ = self.best_ask
            if bid_px is None:
                tb = inner_bid if ask_px is None else round(min(inner_bid, ask_px - tick), 4)
                self._insert(Order(self._new_order_id("sim_LO_bid"), False, tb, self.v_unit), self.bids)
                bid_px, _ = self.best_bid
            if ask_px is None:
                ta = inner_ask if bid_px is None else round(max(inner_ask, bid_px + tick), 4)
                self._insert(Order(self._new_order_id("sim_LO_ask"), True, ta, self.v_unit), self.asks)
        _lap("6_seed_inner_quotes")

        return self, mo_fills

    def snapshot(self, depth: int = 5) -> dict:
        bids = [(p, self.bids[p].total_qty) for p in list(self.bids)[:depth]]
        asks = [(p, self.asks[p].total_qty) for p in list(self.asks)[:depth]]
        return {"bids": bids, "asks": asks, "mid": self.mid, "spread": self.spread}

    def print_book(self, depth: int = 5):
        snap = self.snapshot(depth)
        mid_str = f"{snap['mid']:.5f}" if snap['mid'] is not None else "None"
        spread_str = f"{snap['spread']:.5f}" if snap['spread'] is not None else "None"
        print(f"\n{'─'*40}")
        print(f"mid={mid_str} | spread={spread_str}")
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
    ob = OrderBook(lambda_a0=LAMBDA_A0_B, alpha=ALPHA_B, theta=THETA_B, lambda_mo=LAMBDA_MO_B, v_unit=V_UNIT_B)

    # Populate bids
    ob.add_limit_order(Order("B1", False, 1.08500, 1_000_000))
    ob.add_limit_order(Order("B2", False, 1.08500,   500_000))
    ob.add_limit_order(Order("B3", False, 1.08480, 2_000_000))
    ob.add_limit_order(Order("B4", False, 1.08460, 3_000_000))
    ob.add_limit_order(Order("B5", False, 1.08440, 1_500_000))
    ob.add_limit_order(Order("B6", False, 1.08420, 2_500_000))

    # Populate asks
    ob.add_limit_order(Order("A1", True, 1.08520,   800_000))
    ob.add_limit_order(Order("A2", True, 1.08540, 1_500_000))
    ob.add_limit_order(Order("A3", True, 1.08560, 2_000_000))
    ob.add_limit_order(Order("A4", True, 1.08580, 1_000_000))
    ob.add_limit_order(Order("A5", True, 1.08600, 3_000_000))

    print("=== Book initial ===")
    ob.print_book()

    # Cancel an order
    print(">>> cancel B3")
    ob.cancel("B3")
    ob.print_book()

    # Limit order that matches B1 (FIFO : B1 before B2)
    print(">>> Limit ask 1.08500 pour 900k (matche B1 en FIFO)")
    fills = ob.add_limit_order(Order("X1", True, 1.08500, 900_000))
    for passive, qty in fills:
        print(f"    fill: order {passive.order_id} @ {passive.price:.5f} x {qty:,.0f}")
    ob.print_book()

    # Market order buy
    print(">>> Market buy 2M")
    fills = ob.add_market_order(False, 2_000_000)
    for passive, qty in fills:
        print(f"    fill: order {passive.order_id} @ {passive.price:.5f} x {qty:,.0f}")
    ob.print_book()


def test_advanced():
    random.seed(ORDERBOOK_ADVANCED_TEST_RANDOM_SEED)

    ob = OrderBook(lambda_a0=LAMBDA_A0_B, alpha=ALPHA_B, theta=THETA_B, lambda_mo=LAMBDA_MO_B, v_unit=V_UNIT_B)
    base_mid = 1.08500
    tick = ORDERBOOK_DEFAULT_TICK_SIZE  # 1 pip

    # Build a deep book: many levels on each side around base_mid
    level_count = ORDERBOOK_ADVANCED_TEST_LEVEL_COUNT
    for i in range(1, level_count + 1):
        bid_price = base_mid - i * tick
        ask_price = base_mid + i * tick

        bid_qty = random.randint(1, 10) * 100_000
        ask_qty = random.randint(1, 10) * 100_000

        ob.add_limit_order(Order(f"HB_B{i}", False, bid_price, bid_qty))
        ob.add_limit_order(Order(f"HB_A{i}", True, ask_price, ask_qty))

    print("=== Heavy book initial snapshot ===")
    ob.print_book(depth=10)

    # Simulate several evolution steps with a slowly drifting mid-price
    dt = ORDERBOOK_ADVANCED_TEST_DT

    new_mid = base_mid
    for step in range(ORDERBOOK_ADVANCED_TEST_STEPS):
        # Small random walk in mid-price (for testing)
        new_mid += ORDERBOOK_ADVANCED_TEST_DRIFT_SCALE * random.gauss(0.0, 1.0)
        ob.evolve_one_step(
            new_mid=new_mid,
            dt=dt,
            mo_buy_prob=ORDERBOOK_DEFAULT_MO_BUY_PROB,
        )
        if (step + 1) % ORDERBOOK_ADVANCED_TEST_SNAPSHOT_INTERVAL == 0:
            print(f"=== Snapshot after {step + 1} steps ===")
            ob.print_book(depth=ORDERBOOK_ADVANCED_TEST_BOOK_DEPTH)




# %%%%%%% TEST %%%%%%%%%%
if __name__ == "__main__":
    print(">>> Running basic order book test")
    # test_basic()

    print(">>> Running advanced book evolution test")
    test_advanced()

