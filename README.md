# Market-Making

Market Making project for the Electronic Markets course of Dauphine's 203 master.

The subject is [here](Project%202%20-%20Market%20Making.pdf).

Make sure to install the required packages:

```
pip install -r requirements.txt
```

Run a full simulation from the project root with:

- macOS / Linux:
```
python3 src/BacktestRunner.py
```
- Windows (PowerShell or CMD):
```
py -3 src\BacktestRunner.py
```

You can also test modules directly by running them in the same way (replace the script path).

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
- `steps`: number of simulation ticks executed (each tick is `dt`, typically 10 ms). The EURUSD simulator still maps a full synthetic day cycle (Tokyo -> London -> New York) over those ticks. So with `dt=10ms`, `5000` steps simulate `50s` of engine time, while the price regime progression is compressed across that run.

### Output files

- `output/full_backtest_report.png`: main visual report with PnL, inventory, spreads, fill rates, and trade summaries.
- `output/metrics_realtime.parquet`: per-step metrics (time series) generated during the run.
- `output/metrics_aggregated.parquet`: aggregated metrics and final statistics for the run.
- `output/metrics_fills_log.parquet`: detailed fill-level log (fills, side, price, quantity, and related context).

## Limitations

With `dt=10ms`, a full day would be `8_640_000` steps. If your computer can run `5_000` steps in `5s` (ours cannot), it will take you `2h30` to run the simulation (if nothing crashes in the meantime). So beware of your inputs!!
