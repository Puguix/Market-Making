from OrderBook import OrderBook, Order

FEES_TAKER_A = 0.0004
FEES_MAKER_A = 0.0001
FEES_TAKER_B = 0.0002
FEES_MAKER_B = 0.00009
FEES_TAKER_C = 0.0003
FEES_MAKER_C = 0.00009


class HFT:
    """
    The High Frequency Traders that first arbitrage the price difference between exchanges.
    In phase 3, they make market on A.
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

        if best_bid_A - FEES_TAKER_A > best_ask_B + FEES_TAKER_B:
            # A bid over B ask -> buy B sell A
            qty = min(qty_bid_A, qty_ask_B)
            orders_A.append(Order("__snipe__", "ask", best_bid_A, qty))
            orders_B.append(Order("__snipe__", "bid", best_ask_B, qty))
        if best_ask_A + FEES_TAKER_A < best_bid_B - FEES_TAKER_B:
            # A ask under B bid -> sell B buy A
            qty = min(qty_ask_A, qty_bid_B)
            orders_A.append(Order("__snipe__", "bid", best_ask_A, qty))
            orders_B.append(Order("__snipe__", "ask", best_bid_B, qty))
        if best_bid_A - FEES_TAKER_A > best_ask_C + FEES_TAKER_C:
            # A bid over C ask -> buy C sell A
            qty = min(qty_bid_A, qty_ask_C)
            orders_A.append(Order("__snipe__", "ask", best_bid_A, qty))
            orders_C.append(Order("__snipe__", "bid", best_ask_C, qty))
        if best_ask_A + FEES_TAKER_A < best_bid_C - FEES_TAKER_C:
            # A ask under C bid -> sell C buy A
            qty = min(qty_ask_A, qty_bid_C)
            orders_A.append(Order("__snipe__", "bid", best_ask_A, qty))
            orders_C.append(Order("__snipe__", "ask", best_bid_C, qty))

        return orders_A, orders_B, orders_C


if __name__ == "__main__":
    order_book_A = OrderBook(lambda_a0=5.0, alpha=0.05, theta=0.1, lambda_mo=2.0, v_unit=100000)
    order_book_B = OrderBook(lambda_a0=5.0, alpha=0.05, theta=0.1, lambda_mo=2.0, v_unit=100000)
    order_book_C = OrderBook(lambda_a0=5.0, alpha=0.05, theta=0.1, lambda_mo=2.0, v_unit=100000)

    # Exchange A reference book
    order_book_A.add_limit_order(Order("A_BID_1", "bid", 1.1000, 300_000))
    order_book_A.add_limit_order(Order("A_ASK_1", "ask", 1.1004, 200_000))

    # Exchange B has a cheap ask -> should trigger A bid vs B ask arbitrage
    order_book_B.add_limit_order(Order("B_BID_1", "bid", 1.0985, 250_000))
    order_book_B.add_limit_order(Order("B_ASK_1", "ask", 1.0990, 150_000))

    # Exchange C has an expensive bid -> should trigger A ask vs C bid arbitrage
    order_book_C.add_limit_order(Order("C_BID_1", "bid", 1.1012, 120_000))
    order_book_C.add_limit_order(Order("C_ASK_1", "ask", 1.1016, 250_000))

    hft = HFT()
    orders_A, orders_B, orders_C = hft.snipe(order_book_A, order_book_B, order_book_C)
    print("Orders sent to A:", orders_A)
    print("Orders sent to B:", orders_B)
    print("Orders sent to C:", orders_C)