import os
import optuna
from optuna.trial import FrozenTrial
from BacktestRunner import BacktestRunner
from MarketMaker import MarketMaker, GeometricPriceGridStrategy, GeometricQuantityGridStrategy
from config import PARQUET_PATH_REALTIME, PARQUET_PATH_AGGREGATED, MAX_SPREAD_PIPS, MIN_QTY_QUOTED, HEDGE_THRESHOLD, DEFAULT_GEO_QTY_ALPHA, DEFAULT_GEO_INCREMENT, DEFAULT_TICK_SIZE
import polars as pl
import numpy as np

def logging_callback(study: optuna.Study, trial: FrozenTrial):
    if trial.number % 10 == 0:
        if trial.value is None:
            print(f"Trial {trial.number:>4} | no value (failed or infeasible)")
            return

        try:
            best = study.best_trial
            print(
                f"Trial {trial.number:>4} | "
                f"Sharpe={-trial.value:.4f} | "
                f"Best so far: {-best.value:.4f} "
                f"(gamma={best.params['gamma']:.4f}, kappa={best.params['kappa']:.1f})"
            )
        except ValueError:
            print(f"Trial {trial.number:>4} | Sharpe={-trial.value:.4f} | No feasible best trial yet.")


def check_constraints(df_agg: pl.DataFrame, params: dict) -> list[float]:

    # list of constraints
    constraints = []

    # MAX SPread
    # avg_spread_pips = df_agg["spread_quoted"].mean() * 10_000
    # constraints.append(float(avg_spread_pips - MAX_SPREAD_PIPS))

    # min qty quoted
    # avg_fill = (df_agg["fill_rate_bid"].mean() + df_agg["fill_rate_ask"].mean()) / 2
    
    # constraints.append(float(0.001 - avg_fill))

    return constraints # convention on optuna : if constraint > 0 there is a violation


def objective(trial: optuna.Trial) -> float:

    # --- PARMAS GRIS ---
    gamma         = trial.suggest_float("gamma", 0.01, 0.5, log=True)
    kappa         = trial.suggest_float("kappa", 10_000.0, 500_000.0, log=True)
    # delta_grid    = trial.suggest_float("delta_grid", 0.00005, 0.0005)
    # geo_increment = trial.suggest_float("geo_increment", 1.1, 2.5)
    # qty_alpha     = trial.suggest_float("qty_alpha", 0.4, 0.9)
    # --------------------

    # Run Simulation
    trial_id = trial.number
    runner = BacktestRunner(
        steps=10_000
    )

    try:
        sim = runner.run_simulation_with_params(
            gamma=gamma,
            kappa=kappa,
            hedge_threshold=HEDGE_THRESHOLD,
            delta_grid=DEFAULT_TICK_SIZE,
            geo_increment=DEFAULT_GEO_INCREMENT,
            qty_alpha=DEFAULT_GEO_QTY_ALPHA,
            # delta_grid=delta_grid,
            # geo_increment=geo_increment,
            # qty_alpha=qty_alpha,
        )
    except Exception as e:
        # Fail to converge
        print(f"Trial {trial_id} failed with error: {e}")
        trial.set_user_attr("constraints", [999.0, 999.0]) # high penalty
        return 999.0

    # Check if parquet files exist before reading
    if not os.path.exists(PARQUET_PATH_REALTIME) or not os.path.exists(PARQUET_PATH_AGGREGATED):
        print(f"Trial {trial_id}: parquet files not found")
        trial.set_user_attr("constraints", [999.0, 999.0])
        return 999.0
    
    df_rt  = pl.read_parquet(PARQUET_PATH_REALTIME)
    df_agg = pl.read_parquet(PARQUET_PATH_AGGREGATED)

    if df_agg.is_empty() or df_rt.is_empty():
        trial.set_user_attr("constraints", [999.0, 999.0])
        return 999.0

    pnl = df_rt["mtm_pnl"].to_numpy()
    mean = np.mean(pnl)
    std = np.std(pnl)
    sharpe = mean / (std + 1e-9)

    # checking constraints
    constraint_values = check_constraints(df_agg, {})
    trial.set_user_attr("constraints", constraint_values)

    # logging
    trial.set_user_attr("avg_spread_pips", float(df_agg["spread_quoted"].mean() * 10_000))
    trial.set_user_attr("avg_inventory_pct", float(df_agg["inventory_pct"].mean()))
    trial.set_user_attr("total_snipes", int(df_agg["hft_snipe_count"].sum()))
    trial.set_user_attr("final_pnl", float(pnl[-1]))

    return -sharpe




if __name__ == "__main__":

    sampler = optuna.samplers.TPESampler(
        seed=42,
        n_startup_trials=20,   # exploration aléatoire avant de fitter le modèle
        multivariate=True,     # capture les corrélations entre paramètres
        constant_liar=True,    # réduit les collisions en parallèle
    )

    study = optuna.create_study(
        study_name="mm_calibration_v1",
        direction="minimize",
        sampler=sampler,
        # storage="sqlite:///mm_optim.db",   # reprise automatique si crash
        # load_if_exists=True,
    )

    # # Injection de contraintes dans le sampler
    # study.sampler = optuna.samplers.TPESampler(
    #     seed=42,
    #     multivariate=True,
    #     constant_liar=True,
    #     constraints_func=lambda t: t.user_attrs.get("constraints", [0.0, 0.0])
    # )

    study.optimize(
        objective,
        n_trials=50,
        n_jobs=1,           # 4 simulations en parallèle
        callbacks=[logging_callback],
        gc_after_trial=True,
    )

    # ── Résultats ──────────────────────────────────────────────────
    print("\n=== Best Params ===")
    print(study.best_params)
    print(f"Sharpe optimal : {-study.best_value:.4f}")

    # Importance des features
    importance = optuna.importance.get_param_importances(study)
    print("\n=== Params Importance ===")
    for param, score in importance.items():
        print(f"  {param:<20} {score:.4f}")


    # Backtest with optmial params
    best_params = study.best_params
    runner = BacktestRunner(steps=50_000)
    sim = runner.run_simulation_with_params(
        gamma=best_params["gamma"],
        kappa=best_params["kappa"],
        hedge_threshold=HEDGE_THRESHOLD,
        # delta_grid=best_params["delta_grid"],
        # geo_increment=best_params["geo_increment"],
        # qty_alpha=best_params["qty_alpha"],
        delta_grid=DEFAULT_TICK_SIZE,
        geo_increment=DEFAULT_GEO_INCREMENT,
        qty_alpha=DEFAULT_GEO_QTY_ALPHA,
    )

    runner.analyze_and_plot(sim)