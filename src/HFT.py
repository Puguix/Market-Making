import random
from typing import Optional

from OrderBook import OrderBook, Order

from config import (
    FEES_TAKER_A, FEES_MAKER_A, FEES_TAKER_B, FEES_MAKER_B, FEES_TAKER_C, FEES_MAKER_C,
    LAMBDA_A0_B, ALPHA_B, THETA_B, LAMBDA_MO_B, V_UNIT_B,
    MARKET_MAKER_WEIGHT_B, MARKET_MAKER_WEIGHT_C,
    HFT_PROB_OFF, HFT_PROB_ONE_SIDED,
)

HFT_MM_BID_ORDER_ID = "__hft_mm_bid__"
HFT_MM_ASK_ORDER_ID = "__hft_mm_ask__"
HFT_QUOTE_DECIMALS = 4


def _round_quote_price(price: float, decimals: int = HFT_QUOTE_DECIMALS) -> float:
    return round(price, decimals)


class HFT:
    """
    The High Frequency Traders that first arbitrage the price difference between exchanges.
    In phase 3, they make market on A (tight quotes from a B/C fair, with optional outages).
    """

    def __init__(self):
        pass

    def snipe(self, order_book_A: OrderBook, order_book_B: OrderBook, order_book_C: OrderBook):
        # Pass orders based on current state of A, and states of B and C 50ms ago
        orders_A, orders_B, orders_C = [], [], []

        best_bid_A, qty_bid_A = order_book_A.best_bid
        best_ask_A, qty_ask_A = order_book_A.best_ask
        best_bid_B, qty_bid_B = order_book_B.best_bid
        best_ask_B, qty_ask_B = order_book_B.best_ask
        best_bid_C, qty_bid_C = order_book_C.best_bid
        best_ask_C, qty_ask_C = order_book_C.best_ask

        if any(p is None for p in [best_bid_A, best_ask_A, best_bid_B, best_ask_B, best_bid_C, best_ask_C]):
            return [], [], []

        if best_bid_A * (1 - FEES_TAKER_A) > best_ask_B * (1 + FEES_TAKER_B):
            # A bid over B ask -> buy B sell A
            qty = min(qty_bid_A, qty_ask_B)
            orders_A.append(Order("__snipe__", True, best_bid_A, qty))
            orders_B.append(Order("__snipe__", False, best_ask_B, qty))
        if best_ask_A * (1 + FEES_TAKER_A) < best_bid_B * (1 - FEES_TAKER_B):
            # A ask under B bid -> sell B buy A
            qty = min(qty_ask_A, qty_bid_B)
            orders_A.append(Order("__snipe__", False, best_ask_A, qty))
            orders_B.append(Order("__snipe__", True, best_bid_B, qty))
        if best_bid_A * (1 - FEES_TAKER_A) > best_ask_C * (1 + FEES_TAKER_C):
            # A bid over C ask -> buy C sell A
            qty = min(qty_bid_A, qty_ask_C)
            orders_A.append(Order("__snipe__", True, best_bid_A, qty))
            orders_C.append(Order("__snipe__", False, best_ask_C, qty))
        if best_ask_A * (1 + FEES_TAKER_A) < best_bid_C * (1 - FEES_TAKER_C):
            # A ask under C bid -> sell C buy A
            qty = min(qty_ask_A, qty_bid_C)
            orders_A.append(Order("__snipe__", False, best_ask_A, qty))
            orders_C.append(Order("__snipe__", True, best_bid_C, qty))
        if best_bid_B * (1 - FEES_TAKER_B) > best_ask_C * (1 + FEES_TAKER_C):
            # B bid over C ask -> buy C sell B
            qty = min(qty_bid_B, qty_ask_C)
            orders_B.append(Order("__snipe__", True, best_bid_B, qty))
            orders_C.append(Order("__snipe__", False, best_ask_C, qty))
        if best_ask_B * (1 + FEES_TAKER_B) < best_bid_C * (1 - FEES_TAKER_C):
            # B ask under C bid -> sell C buy B
            qty = min(qty_ask_B, qty_bid_C)
            orders_B.append(Order("__snipe__", False, best_ask_B, qty))
            orders_C.append(Order("__snipe__", True, best_bid_C, qty))

        return orders_A, orders_B, orders_C

    def make_market_on_A(
        self,
        order_book_A: OrderBook,
        order_book_B: OrderBook,
        order_book_C: OrderBook,
        *,
        half_spread: float = 0.00005,
        half_spread_bid: Optional[float] = None,
        half_spread_ask: Optional[float] = None,
        quote_quantity: float = 500_000.0,
        weight_b: float = MARKET_MAKER_WEIGHT_B,
        weight_c: float = MARKET_MAKER_WEIGHT_C,
        prob_off: float = HFT_PROB_OFF,
        prob_one_sided: float = HFT_PROB_ONE_SIDED,
        price_decimals: int = HFT_QUOTE_DECIMALS,
        rng: Optional[random.Random] = None,
    ) -> list[tuple[Order, float]]:
        """
        Refresh aggressive HFT liquidity on A from a B/C fair mid (same weighting as the MM ref).

        Cancels any prior HFT quote orders on A, then either goes dark (prob_off), or reposts
        tight limit bids/asks straddling the fair. With prob_one_sided, only one side is quoted
        (side chosen at random).

        Parameters
        ----------
        half_spread : float
            Half the quoted spread in price space: bid = fair - half_spread, ask = fair + half_spread
            before rounding to ``price_decimals`` (tightness control).
        half_spread_bid, half_spread_ask : optional
            Asymmetric tightness; when None, each side uses ``half_spread``.
        price_decimals : int
            Bid and ask prices are ``round(..., price_decimals)`` (default 4).
        prob_off : float
            Probability the HFT pulls all quotes this step (servers / risk-off).
        prob_one_sided : float
            When quoting, probability of quoting only bid or only ask (random side).
        rng : optional ``random.Random`` for reproducible backtests.

        Returns
        -------
        list[tuple[Order, float]]
            Aggressor-side fill tuples from ``add_limit_order`` if new quotes cross the book.
        """
        r = rng if rng is not None else random
        fills: list[tuple[Order, float]] = []

        order_book_A.cancel(HFT_MM_BID_ORDER_ID)
        order_book_A.cancel(HFT_MM_ASK_ORDER_ID)

        if r.random() < prob_off:
            return fills

        mid_B = order_book_B.mid
        mid_C = order_book_C.mid
        if mid_B is None or mid_C is None:
            return fills

        fair = weight_b * mid_B + weight_c * mid_C
        hb = half_spread if half_spread_bid is None else half_spread_bid
        ha = half_spread if half_spread_ask is None else half_spread_ask

        bid_price = _round_quote_price(fair - hb, price_decimals)
        ask_price = _round_quote_price(fair + ha, price_decimals)
        if bid_price >= ask_price:
            inc = 10 ** (-price_decimals)
            ask_price = _round_quote_price(bid_price + inc, price_decimals)

        post_bid = True
        post_ask = True
        if r.random() < prob_one_sided:
            if r.random() < 0.5:
                post_ask = False
            else:
                post_bid = False

        if post_bid:
            fills.extend(
                order_book_A.add_limit_order(
                    Order(HFT_MM_BID_ORDER_ID, False, bid_price, quote_quantity)
                )
            )
        if post_ask:
            fills.extend(
                order_book_A.add_limit_order(
                    Order(HFT_MM_ASK_ORDER_ID, True, ask_price, quote_quantity)
                )
            )
        return fills


def test_snipe() -> None:
    order_book_A = OrderBook(lambda_a0=LAMBDA_A0_B, alpha=ALPHA_B, theta=THETA_B, lambda_mo=LAMBDA_MO_B, v_unit=V_UNIT_B)
    order_book_B = OrderBook(lambda_a0=LAMBDA_A0_B, alpha=ALPHA_B, theta=THETA_B, lambda_mo=LAMBDA_MO_B, v_unit=V_UNIT_B)
    order_book_C = OrderBook(lambda_a0=LAMBDA_A0_B, alpha=ALPHA_B, theta=THETA_B, lambda_mo=LAMBDA_MO_B, v_unit=V_UNIT_B)

    # Exchange A reference book
    order_book_A.add_limit_order(Order("A_BID_1", False, 1.1000, 300_000))
    order_book_A.add_limit_order(Order("A_ASK_1", True, 1.1004, 200_000))

    # Exchange B has a cheap ask -> should trigger A bid vs B ask arbitrage
    order_book_B.add_limit_order(Order("B_BID_1", False, 1.0985, 250_000))
    order_book_B.add_limit_order(Order("B_ASK_1", True, 1.0990, 150_000))

    # Exchange C has an expensive bid -> should trigger A ask vs C bid arbitrage
    order_book_C.add_limit_order(Order("C_BID_1", False, 1.1012, 120_000))
    order_book_C.add_limit_order(Order("C_ASK_1", True, 1.1016, 250_000))

    hft = HFT()
    orders_A, orders_B, orders_C = hft.snipe(order_book_A, order_book_B, order_book_C)
    print("[test_snipe] Orders sent to A:", orders_A)
    print("[test_snipe] Orders sent to B:", orders_B)
    print("[test_snipe] Orders sent to C:", orders_C)


def test_making() -> None:
    """Post tight HFT quotes on an empty A from the same B/C ladder as test_snipe; then pull liquidity."""
    order_book_A = OrderBook(lambda_a0=LAMBDA_A0_B, alpha=ALPHA_B, theta=THETA_B, lambda_mo=LAMBDA_MO_B, v_unit=V_UNIT_B)
    order_book_B = OrderBook(lambda_a0=LAMBDA_A0_B, alpha=ALPHA_B, theta=THETA_B, lambda_mo=LAMBDA_MO_B, v_unit=V_UNIT_B)
    order_book_C = OrderBook(lambda_a0=LAMBDA_A0_B, alpha=ALPHA_B, theta=THETA_B, lambda_mo=LAMBDA_MO_B, v_unit=V_UNIT_B)

    order_book_B.add_limit_order(Order("B_BID_1", False, 1.0985, 250_000))
    order_book_B.add_limit_order(Order("B_ASK_1", True, 1.0990, 150_000))
    order_book_C.add_limit_order(Order("C_BID_1", False, 1.1012, 120_000))
    order_book_C.add_limit_order(Order("C_ASK_1", True, 1.1016, 250_000))

    mid_B = order_book_B.mid
    mid_C = order_book_C.mid
    assert mid_B is not None and mid_C is not None
    fair = MARKET_MAKER_WEIGHT_B * mid_B + MARKET_MAKER_WEIGHT_C * mid_C
    half_spread = 0.00005
    dec = HFT_QUOTE_DECIMALS
    exp_bid = _round_quote_price(fair - half_spread, dec)
    exp_ask = _round_quote_price(fair + half_spread, dec)
    if exp_bid >= exp_ask:
        exp_ask = _round_quote_price(exp_bid + 10 ** (-dec), dec)

    hft = HFT()
    rng = random.Random(123)
    fills = hft.make_market_on_A(
        order_book_A,
        order_book_B,
        order_book_C,
        half_spread=half_spread,
        prob_off=0.0,
        prob_one_sided=0.0,
        rng=rng,
    )
    bb, qb = order_book_A.best_bid
    ba, qa = order_book_A.best_ask
    print("[test_making] fair (B/C weighted):", fair)
    print("[test_making] expected bid / ask:", exp_bid, exp_ask)
    print("[test_making] book A best bid / ask:", bb, ba, "qty", qb, qa)
    print("[test_making] crossing fills from post:", fills)
    assert bb == exp_bid and ba == exp_ask

    hft.make_market_on_A(
        order_book_A,
        order_book_B,
        order_book_C,
        prob_off=1.0,
        rng=rng,
    )
    assert order_book_A.best_bid[0] is None and order_book_A.best_ask[0] is None
    print("[test_making] after prob_off=1.0: book A is empty (HFT pulled quotes)")

    one_side_book = OrderBook(lambda_a0=LAMBDA_A0_B, alpha=ALPHA_B, theta=THETA_B, lambda_mo=LAMBDA_MO_B, v_unit=V_UNIT_B)
    hft.make_market_on_A(
        one_side_book,
        order_book_B,
        order_book_C,
        half_spread=half_spread,
        prob_off=0.0,
        prob_one_sided=1.0,
        rng=random.Random(0),
    )
    bb1, _ = one_side_book.best_bid
    ba1, _ = one_side_book.best_ask
    assert (bb1 is None) != (ba1 is None), "expected exactly one side quoted"
    print("[test_making] one-sided (prob_one_sided=1, seed=0): bid=", bb1, "ask=", ba1)


if __name__ == "__main__":
    test_snipe()
    print()
    # test_making()