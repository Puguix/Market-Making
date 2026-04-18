import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from OrderBook import OrderBook, Order
from MarketMaker import MarketMaker
from EURUSDPriceSimulator import EURUSDPriceSimulator
from MarketSimulator import MarketSimulator
from HFT import HFT
from config import (
    LAMBDA_A0_B, ALPHA_B, THETA_B, LAMBDA_MO_B, V_UNIT_B, DEFAULT_TICK_SIZE,
    BACKTEST_MID_START, BACKTEST_MM_EUR_QUANTITY, BACKTEST_MM_USD_QUANTITY,
    BACKTEST_MM_GAMMA, PRICE_SIM_DEFAULT_DT_SECONDS, SMOKE_TEST_N_LEVELS,
    SMOKE_TEST_STEPS, SMOKE_TEST_SEED, SMOKE_TEST_MM_SIGMA,
    SMOKE_TEST_MM_KAPPA, SMOKE_TEST_MM_HORIZON, SMOKE_TEST_MM_Q_MAX
)

def build_book(mid, n_levels=SMOKE_TEST_N_LEVELS):
    ob = OrderBook(lambda_a0=LAMBDA_A0_B, alpha=ALPHA_B, theta=THETA_B, lambda_mo=LAMBDA_MO_B, v_unit=V_UNIT_B)
    tick = DEFAULT_TICK_SIZE
    for i in range(1, n_levels + 1):
        ob.add_limit_order(Order(f"B{i}", "bid", mid - i * tick, 500_000))
        ob.add_limit_order(Order(f"A{i}", "ask", mid + i * tick, 500_000))
    return ob

mid = BACKTEST_MID_START
sim = MarketSimulator(
    order_book_A=OrderBook(lambda_a0=LAMBDA_A0_B, alpha=ALPHA_B, theta=THETA_B, lambda_mo=LAMBDA_MO_B, v_unit=V_UNIT_B),  # vide
    order_book_B=build_book(mid),
    order_book_C=build_book(mid),
    market_maker=MarketMaker(
        EUR_quantity=BACKTEST_MM_EUR_QUANTITY, USD_quantity=BACKTEST_MM_USD_QUANTITY,
        gamma=BACKTEST_MM_GAMMA,
        sigma=SMOKE_TEST_MM_SIGMA,
        kappa=SMOKE_TEST_MM_KAPPA,
        T=SMOKE_TEST_MM_HORIZON,
        q_max=SMOKE_TEST_MM_Q_MAX,
        s0=mid
    ),
    price_simulator=EURUSDPriceSimulator(s0=mid, dt_seconds=PRICE_SIM_DEFAULT_DT_SECONDS, seed=SMOKE_TEST_SEED),
    hft=HFT(),
)

sim.simulate_multiple_steps(steps=SMOKE_TEST_STEPS, generate_200ms_history=True)

# Vérifications basiques
assert len(sim.data) == SMOKE_TEST_STEPS, f"Attendu {SMOKE_TEST_STEPS} rows, got {len(sim.data)}"
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