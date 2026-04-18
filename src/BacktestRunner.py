import os
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Import de tes classes existantes
from OrderBook import OrderBook
from MarketMaker import MarketMaker, PARQUET_PATH_AGGREGATED, PARQUET_PATH_REALTIME
from EURUSDPriceSimulator import EURUSDPriceSimulator
from MarketSimulator import MarketSimulator
from HFT import HFT
from config import (
    BACKTEST_DEFAULT_STEPS, BACKTEST_DEFAULT_DT, BACKTEST_MID_START,
    BACKTEST_MM_EUR_QUANTITY, BACKTEST_MM_USD_QUANTITY, BACKTEST_MM_GAMMA,
    BACKTEST_MM_SIGMA, BACKTEST_MM_KAPPA,
    LAMBDA_A0_A, ALPHA_A, THETA_A, LAMBDA_MO_A, V_UNIT_A,
    FILL_RATE_YMAX_MULTIPLIER, FALLBACK_FILL_RATE_YMAX,
    INVENTORY_ALERT_LINE_PCT, INVENTORY_YMAX_PCT, PIPS_MULTIPLIER,
    HFT_MARKER_SIZE_DIVISOR, PLOT_FIGSIZE, PLOT_GRIDSPEC_ROWS,
    PLOT_GRIDSPEC_COLS, PLOT_DPI, TABLE_FONT_SIZE, TABLE_SCALE_X,
    TABLE_SCALE_Y, BACKTEST_REPORT_PATH, LAMBDA_A0_B, ALPHA_B, THETA_B,
    LAMBDA_MO_B, V_UNIT_B, LAMBDA_A0_C, ALPHA_C, THETA_C, LAMBDA_MO_C, V_UNIT_C,
    SIMULATOR_HEDGE_LOOKBACK_B, SIMULATOR_HEDGE_LOOKBACK_C,
    SIMULATOR_BUFFER_B_SIZE, SIMULATOR_BUFFER_C_SIZE
)

# Mandatory comment as per instructions:
# This implementation follows a heuristic-first market making approach under latency constraints.

class BacktestRunner:
    def __init__(self, steps: int = BACKTEST_DEFAULT_STEPS, dt: float = BACKTEST_DEFAULT_DT):
        self.steps = steps
        self.dt = dt
        self.mid_start = BACKTEST_MID_START
        
        # Chemins des fichiers
        self.paths = ["metrics_realtime.parquet", "metrics_aggregated.parquet"]
        self._cleanup()

    def _cleanup(self):
        """Supprime les anciens fichiers pour repartir à neuf."""
        for p in [PARQUET_PATH_REALTIME, PARQUET_PATH_AGGREGATED]:
            if os.path.exists(p):
                os.remove(p)
                print(f">>> Nettoyage : {os.path.basename(p)} supprimé.")

    def run_simulation(self):
        print(f">>> Démarrage de la simulation ({self.steps} steps)...")
        
        # 1. Setup des OrderBooks
        ob_A = OrderBook(lambda_a0=LAMBDA_A0_A, alpha=ALPHA_A, theta=THETA_A, lambda_mo=LAMBDA_MO_A, v_unit=V_UNIT_A)
        ob_B = OrderBook(lambda_a0=LAMBDA_A0_B, alpha=ALPHA_B, theta=THETA_B, lambda_mo=LAMBDA_MO_B, v_unit=V_UNIT_B).build_organic_book(self.mid_start)
        ob_C = OrderBook(lambda_a0=LAMBDA_A0_C, alpha=ALPHA_C, theta=THETA_C, lambda_mo=LAMBDA_MO_C, v_unit=V_UNIT_C).build_organic_book(self.mid_start)

        # 2. Setup du Market Maker (Phase 1 Heuristique)
        mm = MarketMaker(
            EUR_quantity=BACKTEST_MM_EUR_QUANTITY,
            USD_quantity=BACKTEST_MM_USD_QUANTITY,
            gamma=BACKTEST_MM_GAMMA,
            sigma=BACKTEST_MM_SIGMA,
            kappa=BACKTEST_MM_KAPPA,
            T=self.steps * self.dt / 86_400 / 365,
            s0=self.mid_start
        )

        # 3. Setup du Simulateur
        price_simulator = EURUSDPriceSimulator(s0=self.mid_start, dt_seconds=self.dt)
        sim = MarketSimulator(
            order_book_A=ob_A,
            order_book_B=ob_B,
            order_book_C=ob_C,
            market_maker=mm,
            price_simulator=price_simulator,
            hft=HFT()
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
        gs = fig.add_gridspec(PLOT_GRIDSPEC_ROWS, PLOT_GRIDSPEC_COLS)

        # --- GRAPH 1: PnL Evolution ---
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(df_agg["timestamp"], df_agg["mtm_pnl"], label="MtM PnL", color="#1f77b4")
        ax1.fill_between(df_agg["timestamp"], df_agg["mtm_pnl"], alpha=0.2)
        ax1.set_title("Evolution du PnL Total (USD)", fontsize=14, fontweight='bold')
        ax1.set_ylabel("PnL ($)")

        # --- GRAPH 2: Inventory & Skew ---
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(df_rt["timestamp"], df_rt["inventory_pct"] * 100, color="#ff7f0e", label="Inventory % Usage")
        ax2.axhline(INVENTORY_ALERT_LINE_PCT, color='r', linestyle='--', alpha=0.5, label="Alert Threshold")
        ax2.set_title("Utilisation de la Limite d'Inventaire (%)", fontsize=14, fontweight='bold')
        ax2.set_ylim(0, INVENTORY_YMAX_PCT)
        ax2.legend()

        # --- GRAPH 3: Price & Reservation Price ---
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(df_agg["timestamp"], df_agg["mid_A"], label="Mid A", color="black", alpha=0.3)
        ax3.plot(df_agg["timestamp"], df_agg["reservation_price"], label="Reservation Price", color="red", linestyle="--")
        ax3.set_title("Mid Price vs Reservation Price (Inventory Skew)", fontsize=14, fontweight='bold')
        ax3.legend()

        # --- GRAPH 4: Quoted Spread vs Sniping ---
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(df_agg["timestamp"], df_agg["spread_quoted"] * PIPS_MULTIPLIER, label="Spread A (pips)", color="green")
        ax4.set_title("Spread Quoté sur A (Pips)", fontsize=14, fontweight='bold')
        ax4.set_ylabel("Pips")

        # --- GRAPH 5: Sniping Activity ---
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.scatter(df_agg["timestamp"], df_agg["hft_snipe_count"], s=df_agg["hft_snipe_qty"] / HFT_MARKER_SIZE_DIVISOR,
                    alpha=0.5, c="red", label="HFT Snipes")
        ax5.set_title("Activité de Sniping HFT (Taille = Volume)", fontsize=14, fontweight='bold')
        ax5.set_ylabel("Nombre d'attaques")

        # --- GRAPH 6: Fill Rates ---
        ax6 = fig.add_subplot(gs[2, 1])
        avg_fills = [df_agg["fill_rate_bid"].mean(), df_agg["fill_rate_ask"].mean()]
        ax6.bar(["Bid Fill Rate", "Ask Fill Rate"], avg_fills, color=["#1f77b4", "#d62728"])
        ax6.set_title("Taux d'exécution moyen (Fill Rates)", fontsize=14, fontweight='bold')
        ax6.set_ylim(0, max(avg_fills) * FILL_RATE_YMAX_MULTIPLIER if any(avg_fills) else FALLBACK_FILL_RATE_YMAX)

        # --- TABLEAU DE STATS (Bas du graph) ---
        ax_table = fig.add_subplot(gs[3, :])
        ax_table.axis('off')
        
        stats = sim.market_maker.compute_summary_stats()
        if not stats.is_empty():
            cell_text = []
            for col in stats.columns:
                val = stats[col][0]
                if isinstance(val, float): cell_text.append([col, f"{val:.4f}"])
                else: cell_text.append([col, str(val)])
            
            the_table = ax_table.table(cellText=cell_text, colLabels=["Métrique", "Valeur"], 
                                      loc='center', cellLoc='left')
            the_table.auto_set_font_size(False)
            the_table.set_fontsize(TABLE_FONT_SIZE)
            the_table.scale(TABLE_SCALE_X, TABLE_SCALE_Y)

        plt.tight_layout()
        plt.savefig(BACKTEST_REPORT_PATH, dpi=PLOT_DPI)
        print(">>> Rapport sauvegardé sous 'full_backtest_report.png'")
        plt.show()

if __name__ == "__main__":
    runner = BacktestRunner(steps=5_000)
    simulator = runner.run_simulation()
    runner.analyze_and_plot(simulator)