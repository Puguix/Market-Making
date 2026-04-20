# Market-Making

Market Making project for the Electronic Markets course of Dauphine's 203 master.

The subject is [here](Project%202%20-%20Market%20Making.pdf).

Make sure to install the required packages:

```
pip install -r requirements.txt
```

Run a full simulation with `BacktestRunner.py`. You can also test modules directly by running them.

## Structure of the code

- `BacktestRunner.py`: main entry point. It runs the full simulation, collects metrics, and creates the report.
- `MarketSimulator.py`: orchestrates the simulation loop and interactions between market maker, HFT, and order books.
- `MarketMaker.py`: contains the market-making logic (quote generation, inventory control, and hedging decisions).
- `HFT.py`: contains the HFT logic (opportunistic cross-venue actions and optional market making on venue A).
- `OrderBook.py`: implements the limit order book mechanics (limit orders, market orders, cancellations, and matching).
- `EURUSDPriceSimulator.py`: models EURUSD mid-price evolution across the day, including noise for venues B and C.
- `PoissonSimulation.py`: models order arrival intensity using a Poisson-style process.
- `Calibration.py`: helper module to calibrate or tune model parameters.
- `config.py`: central place for constants and configuration values used by all modules.
- `tests/smoke_test.py`: quick smoke test to validate that the core simulation pipeline runs correctly.

## Inputs / Outputs

### Inputs

- `seed`: controls randomness for reproducible runs. Same seed + same parameters gives the same run.
- `phase`: selects strategy setup (for example market-maker only vs market-maker + HFT behaviors).
- `steps`: controls how many simulation ticks are used to represent one trading day.

No matter the number of `steps`, the simulation still represents a full day.  
Changing `steps` changes the time resolution only (more steps = finer granularity, fewer steps = coarser granularity).

### Output files

- `output/full_backtest_report.png`: main visual report with PnL, inventory, spreads, fill rates, and trade summaries.
- `output/metrics_realtime.parquet`: per-step metrics (time series) generated during the run.
- `output/metrics_aggregated.parquet`: aggregated metrics and final statistics for the run.
- `output/metrics_fills_log.parquet`: detailed fill-level log (fills, side, price, quantity, and related context).
