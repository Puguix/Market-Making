import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from OrderBook import OrderBook, Order
from MarketMaker import MarketMaker
from EURUSDPriceSimulator import EURUSDPriceSimulator
from MarketSimulator import MarketSimulator
from HFT import HFT

def build_book(mid, n_levels=5):
    ob = OrderBook(lambda_a0=5.0, alpha=0.05, theta=0.1, lambda_mo=10.0, v_unit=100_000)
    tick = 0.0001
    for i in range(1, n_levels + 1):
        ob.add_limit_order(Order(f"B{i}", "bid", mid - i * tick, 500_000))
        ob.add_limit_order(Order(f"A{i}", "ask", mid + i * tick, 500_000))
    return ob

mid = 1.0850
sim = MarketSimulator(
    order_book_A=OrderBook(lambda_a0=5.0, alpha=0.05, theta=0.1, lambda_mo=2.0, v_unit=10_000),  # vide
    order_book_B=build_book(mid),
    order_book_C=build_book(mid),
    market_maker=MarketMaker(
        EUR_quantity=0.0, USD_quantity=1_000_000.0,
        gamma=0.05,
        sigma=8.33e-6,
        kappa=5000,
        T=24*3600,
        q_max=1_000_000.0,
        s0=mid
    ),
    price_simulator=EURUSDPriceSimulator(s0=mid, dt_seconds=0.01, seed=42),
    hft=HFT(),
)

sim.simulate_multiple_steps(steps=100, generate_200ms_history=True)

# Vérifications basiques
assert len(sim.data) == 100, f"Attendu 100 rows, got {len(sim.data)}"
assert sim.order_book_A.best_bid[0] < sim.order_book_A.best_ask[0], "Crossed book sur A"
print(f"Steps simulés : {len(sim.data)}")
print(f"Trades accumulés : {len(sim.all_trades)}")
print(f"Top 3 trades : {sim.get_top_trades(3)}")
print(f"Inventaire MM : {sim.market_maker.EUR_quantity:.0f} EUR")
nav = sim.market_maker.EUR_quantity * sim.order_book_A.mid + sim.market_maker.USD_quantity
pnl = nav - sim.market_maker._initial_capital
print(f"NAV : {nav:.2f} USD")
print(f"MtM PnL : {pnl:.2f} USD")

print(f"Ordres actifs sur A : {len(sim.order_book_A._orders)}")
print(f"Best bid A : {sim.order_book_A.best_bid}")
print(f"Best ask A : {sim.order_book_A.best_ask}")
print(f"Mid A : {sim.order_book_A.mid:.5f}")
print(f"Mid B actuel : {sim.get_B_midpoint():.5f}")
print(f"Mid C actuel : {sim.get_C_midpoint():.5f}")

print(f"Trades sur A : {len([t for t in sim.all_trades if t['exchange'] == 'A'])}")
print(f"Trades sur B : {len([t for t in sim.all_trades if t['exchange'] == 'B'])}")
print(f"Trades sur C : {len([t for t in sim.all_trades if t['exchange'] == 'C'])}")