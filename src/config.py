import os

# Base directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Absolute directory of this config module.
PROJECT_ROOT = os.path.dirname(BASE_DIR)  # Project root (parent of src).
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")  # Central folder for generated artifacts.
os.makedirs(OUTPUT_DIR, exist_ok=True)
PARQUET_PATH_REALTIME = os.path.join(OUTPUT_DIR, "metrics_realtime.parquet")  # Output file for step-by-step metrics.
PARQUET_PATH_AGGREGATED = os.path.join(OUTPUT_DIR, "metrics_aggregated.parquet")  # Output file for aggregated backtest metrics.
BACKTEST_REPORT_PATH = os.path.join(OUTPUT_DIR, "full_backtest_report.png")  # Path for the generated report image.
PARQUET_PATH_FILLS_LOG = os.path.join(OUTPUT_DIR, "metrics_fills_log.parquet")  # Output file storing detailed fill events.

# Fees constants
FEES_TAKER_A = 0.0004  # Taker fee rate on venue A.
FEES_MAKER_A = 0.0001  # Maker fee rate on venue A.
FEES_TAKER_B = 0.0002  # Taker fee rate on venue B.
FEES_MAKER_B = 0.00009  # Maker fee rate on venue B.
FEES_TAKER_C = 0.0003  # Taker fee rate on venue C.
FEES_MAKER_C = 0.00009  # Maker fee rate on venue C.

# MarketMaker constants
FLUSH_INTERVAL = 1_000  # Number of records buffered before writing to disk.
MARKET_MAKER_WEIGHT_B = 0.75  # Weight of venue B in fair-value estimation.
MARKET_MAKER_WEIGHT_C = 0.25  # Weight of venue C in fair-value estimation.
MARKET_MAKER_LATENCY_B = 0.200  # Assumed decision latency for venue B data (seconds).
MARKET_MAKER_LATENCY_C = 0.170  # Assumed decision latency for venue C data (seconds).
MARKET_MAKER_DELTA_TAU = 0.150  # Time horizon used in MM quote adjustment logic.
INVENTORY_REGIME_NORMAL_THRESHOLD = 0.75  # Inventory ratio boundary for normal regime.
MARKET_MAKER_HEDGE_THRESHOLD = 0.90  # Inventory ratio where hedging behavior is triggered.
MARKET_MAKER_EPSILON = 0.00005  # Small numerical floor used in MM stability checks.
MARKET_MAKER_AGGREGATION_STEPS = 100  # Steps per aggregation window for metrics/features.
MARKET_MAKER_DEFAULT_QUOTE_PHASE = 3  # Default quoting mode used by the market maker.

# Phase 3 — fallback MM on A (behind HFT, latency buffer, stress / wide-spread reaction)
MARKET_MAKER_PHASE3_HEDGE_LEG_TRIGGER = 0.88  # Leg-usage threshold to trigger hedge-focused behavior.
MM_PHASE3_LATENCY_SAFETY_BUFFER_PIPS = 1.5  # Extra quote buffer to account for latency risk (pips).
MM_PHASE3_DEEPEN_BASE_TICKS = 4.0  # Base number of ticks to pull back quotes in fallback mode.
MM_PHASE3_TICK = 0.0001  # Tick size used by phase 3 depth calculations.
MM_PHASE3_HFT_QTY_REFERENCE = 500_000.0  # Reference HFT quantity for scaling quote adjustments.
MM_PHASE3_MAX_EXTRA_TICKS_FROM_HFT = 6.0  # Cap on additional widening driven by HFT pressure.
MM_PHASE3_FALLBACK_SPREAD_MULTIPLIER = 2.2  # Spread multiplier used under fallback/wide-spread conditions.
MM_PHASE3_HFT_MIN_TOTAL_QTY = 25_000.0  # Minimum observed HFT size to activate HFT-aware adjustments.
MM_PHASE3_FALLBACK_DEPTH_PULL_TICKS = 3.0  # Extra depth pullback (ticks) in fallback mode.
MM_PHASE3_FALLBACK_BUFFER_SCALE = 0.45  # Scaling factor for fallback latency/depth buffer.
MM_PHASE3_EMA_SPREAD_ALPHA = 0.02  # EMA smoothing factor for spread tracking.
MM_PHASE3_INVENTORY_RAMP_START_LEG = 0.78  # Inventory ratio where skew ramp starts.
MM_PHASE3_INVENTORY_RAMP_END_LEG = 0.90  # Inventory ratio where skew ramp reaches max.
MM_PHASE3_INVENTORY_SKEW_STRENGTH = 2.0  # Maximum intensity of inventory-based skew.
MM_PHASE3_SIDE_DEPTH_INV_SCALE = 0.55  # Inventory-driven scaling for side-specific quote depth.

# Strategy constants
DEFAULT_MAX_LEVELS = 10  # Default number of quoted depth levels.
DEFAULT_TICK_SIZE = 0.0001  # Default minimum price increment.
DEFAULT_GEO_INCREMENT = 1.4  # Geometric progression factor between successive levels.
DEFAULT_GEO_QTY_ALPHA = 0.7  # Size-decay coefficient across quote levels.
MIN_REMAINING_TIME = 0.001  # Minimum horizon clamp for time-dependent formulas.
PIPS_TO_PRICE_SCALE = 10_000.0  # Conversion factor from price delta to pips.
DEFAULT_FEES_PIPS = 2.0  # Default fee estimate expressed in pips.
DEFAULT_INITIAL_CAPITAL = 1_000_000.0  # Initial capital used in PnL/capital simulations.

# Backtest / simulator constants
BACKTEST_DEFAULT_DAYS = 1  # Default number of simulated days.
BACKTEST_DEFAULT_STEPS = 5_000  # Default number of simulation steps.
BACKTEST_DEFAULT_DT = 0.01  # Time step used by the simulator (seconds).
BACKTEST_MID_START = 1.15  # Initial mid price at simulation start.
BACKTEST_MM_USD_QUANTITY = 500_000.0  # Starting USD inventory for market maker.
BACKTEST_MM_EUR_QUANTITY = 500_000.0 / BACKTEST_MID_START  # Starting EUR inventory converted from USD notionals.
BACKTEST_MM_GAMMA = 0.002  # Risk aversion parameter in Avellaneda-Stoikov logic.
BACKTEST_MM_SIGMA = 8.33e-6  # Volatility estimate used for quote optimization.
BACKTEST_MM_KAPPA = 80_000  # Liquidity/arrival sensitivity parameter.
ORGANIC_BOOK_DEPTH_LEVELS = 10  # Number of organic (non-agent) levels per side.
ORGANIC_BOOK_LEVEL_QTY = 500_000  # Quantity placed at each organic book level.
SIMULATOR_BUFFER_B_SIZE = 21  # Rolling buffer size for venue B features.
SIMULATOR_BUFFER_C_SIZE = 18  # Rolling buffer size for venue C features.
SIMULATOR_MM_B_LOOKBACK_STEPS = 20  # Lookback window for MM signals on venue B.
SIMULATOR_MM_C_LOOKBACK_STEPS = 17  # Lookback window for MM signals on venue C.
SIMULATOR_HFT_LOOKBACK_STEPS = 5  # Lookback window for HFT feature calculations.
SIMULATOR_HEDGE_LOOKBACK_B = 20  # Hedge signal lookback horizon for venue B.
SIMULATOR_HEDGE_LOOKBACK_C = 17  # Hedge signal lookback horizon for venue C.
SIMULATOR_PROGRESS_LOG_INTERVAL = 500  # Step interval for progress logging.
SIMULATOR_TOP_TRADES_COUNT = 10  # Number of top trades shown in summaries.
SIMULATOR_STEP_DT = 0.01  # Per-step simulation timestep (seconds).
SIMULATOR_RANDOM_BUY_PROB = 0.5  # Probability of random organic buy flow.
SIMULATOR_ORGANIC_LAMBDA_SCALE = 0.01  # Scaling for organic order arrival intensity.
SIMULATOR_LOG_PROGRESS_SCALE = 100  # Divider used to normalize progress output.
# Phase: 1 = organic + MM only (no HFT). 2 = + HFT cross-venue snipe. 3 = + HFT make_market_on_A (uses HFT_PROB_*).
SIMULATOR_DEFAULT_PHASE = 1  # Default simulator mode/feature set.
SIMULATOR_PHASE1_ORGANIC_LAMBDA_MULTIPLIER = 3  # Organic flow boost used in phase 1.

# Backtest plotting constants
PLOT_FIGSIZE = (20, 14)  # Default matplotlib figure size for reports.
PLOT_GRIDSPEC_ROWS = 4  # Number of grid rows in the report layout.
PLOT_GRIDSPEC_COLS = 2  # Number of grid columns in the report layout.
PLOT_DPI = 300  # Rendering DPI for saved report images.
INVENTORY_ALERT_LOW_LINE_PCT = 25  # Lower alert guide line for inventory plots (% of max).
INVENTORY_ALERT_HIGH_LINE_PCT = 75  # Upper alert guide line for inventory plots (% of max).
INVENTORY_HEDGE_LOW_LINE_PCT = 10  # Lower hedge threshold line on inventory charts.
INVENTORY_HEDGE_HIGH_LINE_PCT = 90  # Upper hedge threshold line on inventory charts.
INVENTORY_YMAX_PCT = 100  # Max y-axis inventory percentage for plotting.
PIPS_MULTIPLIER = 10_000  # Conversion factor from price units to pips.
HFT_MARKER_SIZE_DIVISOR = 1_000  # Scaling divisor for HFT marker sizes on charts.
TABLE_FONT_SIZE = 10  # Font size used for summary tables.
TABLE_SCALE_X = 1  # Horizontal scale factor for matplotlib tables.
TABLE_SCALE_Y = 1.5  # Vertical scale factor for matplotlib tables.
FILL_RATE_YMAX_MULTIPLIER = 1.5  # Multiplier for dynamic fill-rate chart upper bound.
FALLBACK_FILL_RATE_YMAX = 1  # Fallback fill-rate y-axis max when data is sparse.

# OrderBook evolution/test constants
ORDERBOOK_DEFAULT_LEVELS = 20  # Default number of levels in order book tests.
ORDERBOOK_DEFAULT_TICK_SIZE = 0.0001  # Default order book tick size.
ORDERBOOK_DEFAULT_MO_BUY_PROB = 0.5  # Default market-order buy probability.
ORDERBOOK_ADVANCED_TEST_LEVEL_COUNT = 50  # Depth used in advanced order book tests.
ORDERBOOK_ADVANCED_TEST_STEPS = 5000  # Number of simulation steps in advanced tests.
ORDERBOOK_ADVANCED_TEST_SNAPSHOT_INTERVAL = 25  # Steps between order book snapshots.
ORDERBOOK_ADVANCED_TEST_RANDOM_SEED = 42  # Seed for deterministic advanced tests.
ORDERBOOK_ADVANCED_TEST_DT = 0.01  # Time increment for advanced order book tests.
ORDERBOOK_ADVANCED_TEST_DRIFT_SCALE = 0.00001  # Drift magnitude applied in advanced tests.
ORDERBOOK_ADVANCED_TEST_BOOK_DEPTH = 10  # Effective displayed depth for advanced tests.

# OrderBook constants
LAMBDA_A0_B = 5.0  # Baseline limit-order arrival intensity on venue B.
ALPHA_B = 0.05  # Distance-decay coefficient for venue B arrivals.
THETA_B = 0.1  # Mean-reversion strength for venue B order book dynamics.
LAMBDA_MO_B = 0.5  # Baseline market-order intensity on venue B.
V_UNIT_B = 100_000  # Standard order size unit on venue B.

LAMBDA_A0_C = 5.0  # Baseline limit-order arrival intensity on venue C.
ALPHA_C = 0.05  # Distance-decay coefficient for venue C arrivals.
THETA_C = 0.1  # Mean-reversion strength for venue C order book dynamics.
LAMBDA_MO_C = 0.5  # Baseline market-order intensity on venue C.
V_UNIT_C = 100_000  # Standard order size unit on venue C.

# Exchange A defaults
LAMBDA_A0_A = 50.0  # Baseline limit-order arrival intensity on venue A.
ALPHA_A = 0.05  # Distance-decay coefficient for venue A arrivals.
THETA_A = 0.1  # Mean-reversion strength for venue A order book dynamics.
LAMBDA_MO_A = .05  # Baseline market-order intensity on venue A.
V_UNIT_A = 50_000  # Standard order size unit on venue A.

# Price simulator defaults
PRICE_SIM_DEFAULT_S0 = 1.0850  # Default starting FX price for simulator.
PRICE_SIM_DEFAULT_DT_SECONDS = 0.01  # Simulation timestep for price process.
PRICE_SIM_DEFAULT_BAR_SIGMA_PIPS = 5.0  # Intraday diffusion volatility (pips).
PRICE_SIM_DEFAULT_LAMBDA_JUMP_PER_DAY = 4.0  # Expected number of jump events per day.
PRICE_SIM_DEFAULT_SIGMA_JUMP_PIPS = 7.5  # Jump-size volatility (pips).
PRICE_SIM_DEFAULT_RHO_EPS = 0.9  # Correlation of venue-specific micro-noise shocks.
PRICE_SIM_DEFAULT_SIGMA_EPS_PIPS = 0.3  # Noise scale on venues B and C (pips).
SECONDS_PER_HOUR = 3600.0  # Seconds in one hour.
HOURS_PER_DAY = 24.0  # Hours in one day.
SECONDS_PER_DAY = 86_400  # Seconds in one day.
PRICE_SIM_DEFAULT_DAY_STEPS = 8_640_000  # Number of simulation ticks in one day.
TOKYO_ACTIVITY = 0.6  # Relative flow/activity multiplier during Tokyo session.
LONDON_OPEN_ACTIVITY = 1.4  # Relative flow/activity multiplier at London open.
LONDON_MID_ACTIVITY = 1.0  # Relative flow/activity multiplier mid-London session.
OVERLAP_ACTIVITY = 1.5  # Relative flow/activity multiplier during session overlap.
POST_LONDON_ACTIVITY = 0.8  # Relative flow/activity multiplier after London peak.
OVERNIGHT_ACTIVITY = 0.5  # Relative flow/activity multiplier overnight.

# Poisson model defaults
ARRIVAL_INTENSITY_DEFAULT_SPREAD = 0.01  # Reference spread used in intensity model.
ARRIVAL_INTENSITY_DEFAULT_ALPHA = 0.1  # Sensitivity of fills to quote distance.
ARRIVAL_INTENSITY_DEFAULT_LAMBDA_0 = 0.5  # Baseline order-arrival intensity.

# Constraint for Optim
MAX_SPREAD_PIPS   = 5.0  # Max spread allowed by optimizer (pips).
MIN_QTY_QUOTED    = 100_000  # Minimum quote size required by optimizer.
HEDGE_THRESHOLD   = 0.9  # Inventory threshold for hedge constraint/trigger.

# HFT constants
HFT_PROB_OFF = 0.01  # Probability of HFT pausing/turning off quoting.
HFT_PROB_ONE_SIDED = 0.01  # Probability of HFT quoting only one side.