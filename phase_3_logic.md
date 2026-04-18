
In Phase 3, the market maker’s role shifts from a primary liquidity provider to a fallback liquidity system. Because HFTs now dominate the top of the book with tighter spreads, the strategy must pivot from active competition to passive risk management and "tail-risk" provisioning.
+3

Based on the project requirements, here is how the quoting strategy should adapt:

1. Shift in Order Book Positioning
Since HFTs are aggressively quoting at the best bid and offer (BBO), your orders will likely sit further back in the book.


Avoid "Pennying": You should not attempt to outprice HFTs for small trades, as they have a significant speed advantage (50ms vs. your 170–200ms latency).
+1


Focus on Depth: Your 10 levels of quotes should be structured to capture large orders that exceed the thin liquidity provided by HFTs.
+1


Safety Spreads: Because you are slower, your quotes should incorporate a wider "buffer" to avoid being "picked off" by HFTs when prices move on Exchanges B and C.
+2

2. Tail-Risk and Volatility Response
The primary objective in Phase 3 is to provide liquidity during market stress.
+1


Asymmetric Quoting: If market volatility spikes or HFTs go "one-sided" (only quoting one side of the book), you must be prepared to absorb the resulting flow.
+2

Liquidity "Black Hole" Protection: Your model should detect when HFTs go offline or pull their quotes. In these scenarios, you become the sole price discovery mechanism on Exchange A and should widen spreads significantly to compensate for the increased risk.
+2

3. Inventory-Centric Management
With HFTs handling "normal" flow, your fills will likely be infrequent but large.

Skewed Quotes: Use your inventory levels to aggressively skew your quotes. If you are long EUR, you should lower your bid and ask prices to discourage further buys and encourage sells, even if this places you far away from the HFT-driven mid-price.
+1


Externalization Strategy: Since you are the fallback, you are more likely to hit your 90% delta risk limit quickly during a large event. Your strategy must prioritize efficient hedging on Exchanges B and C (paying the taker fees of 0.02%–0.03%) when your internal warehouse is full.
+1

4. Behavioral Adaptation
Scenario	HFT Behavior	Your Quoting Strategy
Normal Markets	
Tight spreads, high competition.

Deep book placement; focus on capturing "whale" trades.

Volatility Spikes	
HFTs may widen or vanish.
+1

Maintain presence but widen spreads to protect capital.
+2

HFT Outage	
Order book empties.
+1

Revert to Phase 2 style "primary" market making but with high risk-premium spreads.
+2

In summary, the Phase 3 strategy is less about "making the market" and more about "insuring the market" against liquidity droughts, while managing the adverse selection inherent in being the slowest participant.
+1



# Implementation

Summary of what was implemented:

OrderBook.resting_quantity(order_id)
Returns remaining size for that id (used to infer HFT size on A via HFT_MM_BID_ORDER_ID / HFT_MM_ASK_ORDER_ID).

config.py — Phase 3 parameters
Tunable constants: latency buffer (pips), base depth (ticks), HFT size scaling, fallback spread multiple, minimum HFT size, fallback depth/buffer scaling, EMA α for A’s spread, inventory ramp window, skew strength, side depth asymmetry, and MARKET_MAKER_PHASE3_HEDGE_LEG_TRIGGER = 0.88 for earlier B/C hedging.

UtilityProblem
spread_safety_addon: added to optimal_spread (latency / adverse‑selection cushion vs HFTs on B/C).
quote_depth_bid_delta / quote_depth_ask_delta: push the geometric grid away from the touch (bids lower, asks higher) so the 10 levels sit behind the theoretical A‑S quotes; depth grows with log1p(hft_q / reference_qty).
MarketMaker
hedge_leg_trigger: min(0.90, 0.88) = 0.88 — check_and_hedge uses this so risk is reduced before the old 90% leg cap.
_ema_spread_A: EMA of A’s spread for “wide spread” detection.
_build_utility_problem(..., *, spread_safety_addon, quote_depth_bid_delta, quote_depth_ask_delta, inventory_skew_multiplier) — passes Phase 3 knobs into UtilityProblem.
plan_phase3_quote_actions(order_book_A, B, C) -> (list[str], list[Order])
Inventory: leg_share → pressure in [0.78, 0.90]; inventory_skew_multiplier ramps Avellaneda–Stoikov skew; side depth tilts bid vs ask depth so that when long EUR bids sit deeper and asks move in (favor shrinking EUR), and the opposite when short.
Fallback “tighten”: if HFT resting size on A is below threshold or A’s spread > EMA × multiplier → shrink depth and latency buffer so the MM moves closer to the touch.
Returns all MM order ids to cancel (levels dropped or repriced) and new Order objects (still MM_{id} ids).
apply_quote_plan(order_book_A, order_ids_to_cancel, orders_to_submit) — cancels ids, clears _active_orders, posts submits, runs update_inventory_from_fills on any crosses.
make_market — epsilon check unchanged; then plan_phase3_quote_actions + apply_quote_plan. If B/C mids are missing, all MM orders on A are cancelled and _active_orders cleared.
make_market remains the entry point used by MarketSimulator; the explicit cancel/post lists are produced by plan_phase3_quote_actions and executed by apply_quote_plan.

If you want the simulator to call make_market_on_A before the MM so “normal” depth uses real HFT size, wire that in MarketSimulator next; with the current sim (snipe only), hft_q is often 0, so the “HFT disappeared” branch will often fire and the MM will quote tighter until HFT posting exists on A.