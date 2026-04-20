import os
from typing import Optional

import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math

# Import de tes classes existantes
from OrderBook import OrderBook, Order
from MarketMaker import MarketMaker, PARQUET_PATH_AGGREGATED, PARQUET_PATH_REALTIME, PARQUET_PATH_FILLS_LOG
from EURUSDPriceSimulator import EURUSDPriceSimulator
from MarketSimulator import MarketSimulator
from HFT import HFT
from config import (
    BACKTEST_DEFAULT_STEPS, BACKTEST_DEFAULT_DT, BACKTEST_MID_START,
    BACKTEST_MM_EUR_QUANTITY, BACKTEST_MM_USD_QUANTITY, BACKTEST_MM_GAMMA,
    BACKTEST_MM_SIGMA, BACKTEST_MM_KAPPA,
    LAMBDA_A0_A, ALPHA_A, THETA_A, LAMBDA_MO_A, V_UNIT_A,
    FILL_RATE_YMAX_MULTIPLIER, FALLBACK_FILL_RATE_YMAX,
    INVENTORY_ALERT_LOW_LINE_PCT, INVENTORY_ALERT_HIGH_LINE_PCT,
    INVENTORY_HEDGE_LOW_LINE_PCT, INVENTORY_HEDGE_HIGH_LINE_PCT,
    INVENTORY_YMAX_PCT, PIPS_MULTIPLIER,
    HFT_MARKER_SIZE_DIVISOR, PLOT_FIGSIZE, PLOT_GRIDSPEC_ROWS,
    PLOT_GRIDSPEC_COLS, PLOT_DPI, TABLE_FONT_SIZE, TABLE_SCALE_X,
    TABLE_SCALE_Y, BACKTEST_REPORT_PATH, LAMBDA_A0_B, ALPHA_B, THETA_B,
    LAMBDA_MO_B, V_UNIT_B, LAMBDA_A0_C, ALPHA_C, THETA_C, LAMBDA_MO_C, V_UNIT_C,
    SIMULATOR_HEDGE_LOOKBACK_B, SIMULATOR_HEDGE_LOOKBACK_C,
    SIMULATOR_BUFFER_B_SIZE, SIMULATOR_BUFFER_C_SIZE,
    SIMULATOR_DEFAULT_PHASE,
)

import subprocess
import sys

def _open_file(path: str) -> None:
    """Ouvre un fichier avec le visualiseur par défaut (Mac, Windows, Linux)."""
    if sys.platform == "darwin":
        subprocess.Popen(["open", path])
    elif sys.platform == "win32":
        os.startfile(path)
    else:
        subprocess.Popen(["xdg-open", path])

# Mandatory comment as per instructions:
# This implementation follows a heuristic-first market making approach under latency constraints.

class BacktestRunner:
    def __init__(
        self,
        steps: int = BACKTEST_DEFAULT_STEPS,
        dt: float = BACKTEST_DEFAULT_DT,
        phase: int = SIMULATOR_DEFAULT_PHASE,
    ):
        if phase not in (1, 2, 3):
            raise ValueError(f"phase must be 1, 2, or 3, got {phase!r}")
        self.steps = steps
        self.dt = dt
        self.mid_start = BACKTEST_MID_START
        self.phase = int(phase)

        # Chemins des fichiers
        self.paths = ["metrics_realtime.parquet", "metrics_aggregated.parquet"]
        self._cleanup()

    def _cleanup(self):
        """Supprime les anciens fichiers pour repartir à neuf."""
        for p in [PARQUET_PATH_REALTIME, PARQUET_PATH_AGGREGATED, PARQUET_PATH_FILLS_LOG]:
            if os.path.exists(p):
                os.remove(p)
                print(f">>> Nettoyage : {os.path.basename(p)} supprimé.")

    def run_simulation(self):
        print(f">>> Démarrage de la simulation ({self.steps} steps, phase={self.phase})...")
        
        # 1. Setup des OrderBooks
        ob_A = OrderBook(lambda_a0=LAMBDA_A0_A, alpha=ALPHA_A, theta=THETA_A, lambda_mo=LAMBDA_MO_A, v_unit=V_UNIT_A)
        ob_B = OrderBook(lambda_a0=LAMBDA_A0_B, alpha=ALPHA_B, theta=THETA_B, lambda_mo=LAMBDA_MO_B, v_unit=V_UNIT_B).build_organic_book(self.mid_start)
        ob_C = OrderBook(lambda_a0=LAMBDA_A0_C, alpha=ALPHA_C, theta=THETA_C, lambda_mo=LAMBDA_MO_C, v_unit=V_UNIT_C).build_organic_book(self.mid_start)

        # 2. Setup du Market Maker (Phase 1 Heuristique)
        mm_quote_phase = 3 if self.phase == 3 else 1
        mm = MarketMaker(
            EUR_quantity=BACKTEST_MM_EUR_QUANTITY,
            USD_quantity=BACKTEST_MM_USD_QUANTITY,
            gamma=BACKTEST_MM_GAMMA,
            sigma=BACKTEST_MM_SIGMA,
            kappa=BACKTEST_MM_KAPPA,
            T=self.steps * self.dt,
            s0=self.mid_start,
            quote_phase=mm_quote_phase,
        )

        # 3. Setup du Simulateur
        price_simulator = EURUSDPriceSimulator(s0=self.mid_start, dt_seconds=self.dt)
        sim = MarketSimulator(
            order_book_A=ob_A,
            order_book_B=ob_B,
            order_book_C=ob_C,
            market_maker=mm,
            price_simulator=price_simulator,
            hft=HFT(),
            phase=self.phase,
        )

        # 4. Set up de 200ms de data sur B et C
        price_simulator.generate_prices(SIMULATOR_BUFFER_B_SIZE + self.steps)
        sim.simulate_200ms_history()

        # 5. Make the market on A
        mm.make_market(
            sim.order_book_A,
            sim.order_books_B[(sim.current_idx_B - SIMULATOR_HEDGE_LOOKBACK_B) % SIMULATOR_BUFFER_B_SIZE],
            sim.order_books_C[(sim.current_idx_C - SIMULATOR_HEDGE_LOOKBACK_C) % SIMULATOR_BUFFER_C_SIZE],
        )

        # 6. Execution
        sim.simulate_n_steps(n_steps=self.steps)
        sim.market_maker._flush_to_parquet() 
        print(f">>> Fichiers écrits dans : {os.path.dirname(PARQUET_PATH_AGGREGATED)}")
        print(">>> Simulation terminée. Analyse des données...")
        return sim

    def _generate_mtm_aging(self, ax, df_rt: pl.DataFrame) -> None:
        """
        Compute and plot MtM P&L as a function of time since trade inception.
        Reads fills_log.parquet, joins asof with df_rt on mid_ref, aggregates by horizon.
        """
        from config import PARQUET_PATH_FILLS_LOG

        if not os.path.exists(PARQUET_PATH_FILLS_LOG):
            ax.axis('off')
            ax.text(0.5, 0.5, "fills_log.parquet introuvable", ha='center', va='center')
            return

        df_fills = pl.read_parquet(PARQUET_PATH_FILLS_LOG)
        if len(df_fills) == 0:
            ax.axis('off')
            ax.text(0.5, 0.5, "Aucun fill enregistré", ha='center', va='center')
            return

        df_rt_ts = df_rt.select(["timestamp", "mid_ref"]).sort("timestamp")
        horizons_steps = [0, 2, 5, 10, 20, 50, 100]

        dt = self.dt

        # Explode : une ligne par (fill, horizon)
        rows = []
        for h in horizons_steps:
            tmp = df_fills.with_columns([
                pl.lit(h).alias("horizon_steps"),
                (pl.col("timestamp_fill") + h * dt).alias("horizon_timestamp"),
                # bid=+1 (on a acheté, on gagne si mid monte), ask=-1
                ((pl.col("is_ask").cast(pl.Int8) * -2) + 1).cast(pl.Float64).alias("sign"),
            ])
            rows.append(tmp)

        df_exploded = pl.concat(rows).sort("horizon_timestamp")

        # join_asof : mid_ref le plus proche au timestamp horizon
        df_joined = df_exploded.join_asof(
            df_rt_ts,
            left_on="horizon_timestamp",
            right_on="timestamp",
            strategy="nearest",
        )

        # MtM en pips par unité
        df_joined = df_joined.with_columns([
            ((pl.col("mid_ref") - pl.col("fill_price")) * pl.col("sign") * 10_000).alias("mtm_pips"),
        ])

        # Agrégation par horizon
        df_agg = df_joined.group_by("horizon_steps").agg([
            pl.col("mtm_pips").mean().alias("mean"),
            pl.col("mtm_pips").median().alias("median"),
            pl.col("mtm_pips").quantile(0.05).alias("p5"),
            pl.col("mtm_pips").quantile(0.95).alias("p95"),
        ]).sort("horizon_steps")

        x = df_agg["horizon_steps"].to_numpy() * dt * 1000  # ms
        ax.plot(x, df_agg["mean"].to_numpy(), label="Mean", color="#1f77b4", linewidth=2)
        ax.plot(x, df_agg["median"].to_numpy(), label="Median", color="#ff7f0e", linewidth=2, linestyle="--")
        ax.fill_between(x, df_agg["p5"].to_numpy(), df_agg["p95"].to_numpy(), alpha=0.2, color="#1f77b4", label="P5–P95")
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_xlabel("Time since trade inception (ms)")
        ax.set_ylabel("MtM P&L (pips / unit)")
        ax.set_title("MtM P&L as a Function of Time Since Trade Inception", fontsize=14, fontweight='bold')
        ax.legend()

    def _generate_tables_report(self, sim) -> None:
        fig = plt.figure(figsize=(14, 10))
        fig.patch.set_facecolor('white')

        # --- Stats table ---
        ax1 = fig.add_axes([0.02, 0.55, 0.96, 0.38])  # [left, bottom, width, height]
        ax1.axis('off')
        fig.text(0.5, 0.95, "Summary Statistics", ha='center', va='top',
                fontsize=13, fontweight='bold')

        stats = sim.market_maker.compute_summary_stats()
        if not stats.is_empty():
            n = len(stats.columns)
            mid = math.ceil(n / 2)
            col1 = [(col, stats[col][0]) for col in stats.columns[:mid]]
            col2 = [(col, stats[col][0]) for col in stats.columns[mid:]]
            while len(col2) < len(col1):
                col2.append(("", ""))
            fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)
            cell_text = [[k1, fmt(v1), k2, fmt(v2)] for (k1, v1), (k2, v2) in zip(col1, col2)]
            tbl = ax1.table(
                cellText=cell_text,
                colLabels=["Métrique", "Valeur", "Métrique", "Valeur"],
                loc='center', cellLoc='left',
                colWidths=[0.3, 0.2, 0.3, 0.2],
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(10)
            tbl.scale(1.0, 2.0)

        # --- Top 10 trades table ---
        ax2 = fig.add_axes([0.15, 0.02, 0.70, 0.45])
        ax2.axis('off')
        fig.text(0.5, 0.50, "Top 10 Trades by Size (Exchange A)", ha='center', va='top',
                fontsize=13, fontweight='bold')

        top_trades = sim.get_top_trades(10)
        if top_trades:
            cell_text = []
            for i, t in enumerate(top_trades, 1):
                side = "ASK (sell)" if t["Is Ask"] else "BID (buy)"
                cell_text.append([str(i), f"{t['price']:.5f}", f"{t['quantity']:,.0f}", side])
            tbl2 = ax2.table(
                cellText=cell_text,
                colLabels=["Rank", "Price", "Quantity (EUR)", "Side"],
                loc='center', cellLoc='center',
                colWidths=[0.08, 0.25, 0.25, 0.25],
            )
            tbl2.auto_set_font_size(False)
            tbl2.set_fontsize(10)
            tbl2.scale(1.0, 2.0)

        tables_path = BACKTEST_REPORT_PATH.replace(".png", "_tables.png")
        plt.savefig(tables_path, dpi=PLOT_DPI)
        print(f">>> Tables sauvegardées sous '{tables_path}'")
        plt.close('all')

    def analyze_and_plot(self, sim):
        path_rt = PARQUET_PATH_REALTIME
        path_agg = PARQUET_PATH_AGGREGATED
        
        if not os.path.exists(path_agg):
            print(f"ERREUR : Le fichier {path_agg} est introuvable.")
            return

        sns.set_theme(style="whitegrid")
        df_rt = pl.read_parquet(path_rt)
        df_agg = pl.read_parquet(path_agg)

        fig = plt.figure(figsize=PLOT_FIGSIZE)
        gs = fig.add_gridspec(4, 2, hspace=0.45, wspace=0.35,
              top=0.97, bottom=0.03, left=0.06, right=0.97)

        # --- GRAPH 1: PnL Evolution ---
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(df_agg["timestamp"], df_agg["mtm_pnl"], label="MtM PnL", color="#1f77b4")
        ax1.fill_between(df_agg["timestamp"], df_agg["mtm_pnl"], alpha=0.2)
        ax1.set_title("Evolution du PnL Total (USD)", fontsize=14, fontweight='bold')
        ax1.set_ylabel("PnL ($)")

        # --- GRAPH 2: Inventory ---
        ax2 = fig.add_subplot(gs[0, 1])
        inventory_value_usd = df_rt["EUR_quantity"] * df_rt["mid_ref"] + df_rt["USD_quantity"]
        eur_share_pct = 100 * (df_rt["EUR_quantity"] * df_rt["mid_ref"]) / inventory_value_usd
        ax2.plot(df_rt["timestamp"], eur_share_pct, color="#ff7f0e", label="EUR share in inventory value (%)")
        ax2.axhline(INVENTORY_ALERT_LOW_LINE_PCT, color='orange', linestyle='--', alpha=0.7, label="Alert Low (25%)")
        ax2.axhline(INVENTORY_ALERT_HIGH_LINE_PCT, color='orange', linestyle='--', alpha=0.7, label="Alert High (75%)")
        ax2.axhline(INVENTORY_HEDGE_LOW_LINE_PCT, color='red', linestyle='--', alpha=0.8, label="Hedge Low (10%)")
        ax2.axhline(INVENTORY_HEDGE_HIGH_LINE_PCT, color='red', linestyle='--', alpha=0.8, label="Hedge High (90%)")
        ax2.set_title("Part EUR dans la Valeur d'Inventaire (%)", fontsize=14, fontweight='bold')
        ax2.set_ylim(0, INVENTORY_YMAX_PCT)
        ax2.legend(fontsize=7, loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0)

        # --- GRAPH 3: Price & Reservation Price ---
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(df_agg["timestamp"], df_agg["mid_A"], label="Mid A", color="black", alpha=0.3)
        ax3.plot(df_agg["timestamp"], df_agg["mid_B"], label="Mid B", color="blue", alpha=0.3)
        ax3.plot(df_agg["timestamp"], df_agg["reservation_price"], label="Reservation Price", color="red", linestyle="--")
        ax3.set_title("Mid Price vs Reservation Price (Inventory Skew)", fontsize=14, fontweight='bold')
        ax3.legend()

        # --- GRAPH 4: Spread ---
        ax4 = fig.add_subplot(gs[1, 1])
        spread_b = (df_agg["best_ask_B"] - df_agg["best_bid_B"]) * PIPS_MULTIPLIER
        ax4.plot(df_agg["timestamp"], df_agg["spread_quoted"] * PIPS_MULTIPLIER, label="Spread A (pips)", color="green")
        ax4.plot(df_agg["timestamp"], spread_b, label="Spread B bid-ask (pips)", color="purple", alpha=0.8)
        ax4.set_title("Spread A et Spread B (Pips)", fontsize=14, fontweight='bold')
        ax4.set_ylabel("Pips")
        ax4.legend()

        # --- GRAPH 5: Sniping Activity ---
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.scatter(df_agg["timestamp"], df_agg["hft_snipe_count"],
                    s=df_agg["hft_snipe_qty"] / HFT_MARKER_SIZE_DIVISOR,
                    alpha=0.5, c="red", label="HFT Snipes")
        ax5.set_title("Activité de Sniping HFT (Taille = Volume)", fontsize=14, fontweight='bold')
        ax5.set_ylabel("Nombre d'attaques")

        # --- GRAPH 6: Fill Rates ---
        ax6 = fig.add_subplot(gs[2, 1])
        avg_fills = [df_agg["fill_rate_bid"].mean(), df_agg["fill_rate_ask"].mean()]
        ax6.bar(["Bid Fill Rate", "Ask Fill Rate"], avg_fills, color=["#1f77b4", "#d62728"])
        ax6.set_title("Taux d'exécution moyen (Fill Rates)", fontsize=14, fontweight='bold')
        ax6.set_ylim(0, max(avg_fills) * FILL_RATE_YMAX_MULTIPLIER if any(avg_fills) else FALLBACK_FILL_RATE_YMAX)

        # --- GRAPH 7: MtM aging ---
        ax7 = fig.add_subplot(gs[3, :])
        self._generate_mtm_aging(ax7, df_rt)

        plt.tight_layout()
        plt.savefig(BACKTEST_REPORT_PATH, dpi=PLOT_DPI, bbox_inches='tight')
        print(f">>> Rapport graphs sauvegardé sous '{BACKTEST_REPORT_PATH}'")
        plt.close('all')  

        # --- Fichier séparé pour les tables ---
        self._generate_tables_report(sim)

        # Ouvre les deux fichiers PNG automatiquement
        import subprocess
        tables_path = BACKTEST_REPORT_PATH.replace(".png", "_tables.png")
        _open_file(BACKTEST_REPORT_PATH)
        _open_file(tables_path)


    def run_simulation_with_params(
        self,
        gamma: float,
        kappa: float,
        hedge_threshold: float,
        delta_grid: float,
        geo_increment: float,
        qty_alpha: float,
        phase: int = SIMULATOR_DEFAULT_PHASE,
        quote_phase: Optional[int] = None,
    ) -> "MarketSimulator":
        if phase not in (1, 2, 3):
            raise ValueError(f"phase must be 1, 2, or 3, got {phase!r}")
        if quote_phase is None:
            quote_phase = 1 if phase == 1 else 3
        if quote_phase not in (1, 3):
            raise ValueError(f"quote_phase must be 1 or 3, got {quote_phase!r}")

        self._cleanup()

        ob_A = OrderBook(lambda_a0=LAMBDA_A0_A, alpha=ALPHA_A, theta=THETA_A, lambda_mo=LAMBDA_MO_A, v_unit=V_UNIT_A)
        ob_B = OrderBook(lambda_a0=LAMBDA_A0_B, alpha=ALPHA_B, theta=THETA_B, lambda_mo=LAMBDA_MO_B, v_unit=V_UNIT_B).build_organic_book(self.mid_start)
        ob_C = OrderBook(lambda_a0=LAMBDA_A0_C, alpha=ALPHA_C, theta=THETA_C, lambda_mo=LAMBDA_MO_C, v_unit=V_UNIT_C).build_organic_book(self.mid_start)

        mm = MarketMaker(
            EUR_quantity=BACKTEST_MM_EUR_QUANTITY,
            USD_quantity=BACKTEST_MM_USD_QUANTITY,
            gamma=gamma,
            sigma=BACKTEST_MM_SIGMA, # to estimate from simulated prices
            kappa=kappa,
            T=self.steps * self.dt,
            s0=self.mid_start,
            quote_phase=quote_phase,
        )
        mm.hedge_threshold = hedge_threshold

        price_sim = EURUSDPriceSimulator(s0=self.mid_start, dt_seconds=self.dt, seed=42)
        price_sim.generate_prices(SIMULATOR_BUFFER_B_SIZE + self.steps)  # génère d'abord

        sim = MarketSimulator(
            order_book_A=ob_A,
            order_book_B=ob_B,
            order_book_C=ob_C,
            market_maker=mm,
            price_simulator=price_sim,
            hft=HFT(),
            phase=phase,
        )

        sim.simulate_200ms_history()

        # Make Market on a to initialize it
        mm.make_market(
            sim.order_book_A,
            sim.order_books_B[(sim.current_idx_B - SIMULATOR_HEDGE_LOOKBACK_B) % SIMULATOR_BUFFER_B_SIZE],
            sim.order_books_C[(sim.current_idx_C - SIMULATOR_HEDGE_LOOKBACK_C) % SIMULATOR_BUFFER_C_SIZE],
        )

        sim.simulate_n_steps(n_steps=self.steps)

        sim.market_maker._flush_to_parquet()
        return sim
    

if __name__ == "__main__":
    runner = BacktestRunner(steps=50_000, phase=1)
    simulator = runner.run_simulation()
    runner.analyze_and_plot(simulator)