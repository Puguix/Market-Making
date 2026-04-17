import os

# Base directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARQUET_PATH_REALTIME = os.path.join(BASE_DIR, "metrics_realtime.parquet")
PARQUET_PATH_AGGREGATED = os.path.join(BASE_DIR, "metrics_aggregated.parquet")

# Fees / HFT constants
FEES_TAKER_A = 0.0004
FEES_MAKER_A = 0.0001
FEES_TAKER_B = 0.0002
FEES_MAKER_B = 0.00009
FEES_TAKER_C = 0.0003
FEES_MAKER_C = 0.00009

# MarketMaker constants
FLUSH_INTERVAL = 1_000

# OrderBook constants
LAMBDA_A0_B = 5.0
ALPHA_B = 0.05
THETA_B = 0.1
LAMBDA_MO_B = 2.0
V_UNIT_B = 100_000

LAMBDA_A0_C = 5.0
ALPHA_C = 0.05
THETA_C = 0.1
LAMBDA_MO_C = 2.0
V_UNIT_C = 100_000
