import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from OrderBook import OrderBook


def test_order_book():
    ob = OrderBook()
    ob.init_dummy_order_book(levels=10, price=100, quantity=100)

    print("Order book after dummy init:")
    ob.display()
    print()

    # Limit buy at 95 (resting)
    ob.add_limit_order(95, 50, is_bid=True)
    print("After limit buy 50 @ 95:")
    ob.display()
    print()

    # Market sell 30
    ob.add_market_order(30, is_bid=False)
    print("After market sell 30:")
    ob.display()
    print()

    # Limit sell that crosses (sell at 92)
    ob.add_limit_order(92, 20, is_bid=False)
    print("After limit sell 20 @ 92:")
    ob.display()


if __name__ == "__main__":
    test_order_book()
