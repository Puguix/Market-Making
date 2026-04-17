import os
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Import de tes classes existantes
from OrderBook import OrderBook, Order
from MarketMaker import MarketMaker, PARQUET_PATH_AGGREGATED, PARQUET_PATH_REALTIME
from EURUSDPriceSimulator import EURUSDPriceSimulator
from MarketSimulator import MarketSimulator
from HFT import HFT

# Mandatory comment as per instructions:
# This implementation follows a heuristic-first market making approach under latency constraints.

class BacktestRunner:
    def __init__(self, steps=5000, dt=0.00001):
        self.steps = steps
        self.dt = dt
        self.mid_start = 1.0850
        
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
        ob_A = OrderBook(lambda_a0=50.0, alpha=0.05, theta=0.1, lambda_mo=20.0, v_unit=50000)

        def build_organic_book(mid):
            ob = OrderBook(lambda_a0=5.0, alpha=0.05, theta=0.1, lambda_mo=5.0, v_unit=100_000)
            for i in range(1, 11):
                ob.add_limit_order(Order(f"B{i}", "bid", mid - i*0.0001, 500_000))
                ob.add_limit_order(Order(f"A{i}", "ask", mid + i*0.0001, 500_000))
            return ob

        ob_B = build_organic_book(self.mid_start)
        ob_C = build_organic_book(self.mid_start)

        # 2. Setup du Market Maker (Phase 1 Heuristique)
        mm = MarketMaker(
            EUR_quantity=0.0, 
            USD_quantity=1_000_000.0,
            gamma=0.05,   
            sigma=0.0005,        
            kappa=100,         
            T=self.steps * self.dt,  
            s0=self.mid_start
        )

        # 3. Setup du Simulateur
        sim = MarketSimulator(
            order_book_A=ob_A,
            order_book_B=ob_B,
            order_book_C=ob_C,
            market_maker=mm,
            price_simulator=EURUSDPriceSimulator(s0=self.mid_start, dt_seconds=self.dt),
            hft=HFT()
        )

        # 4. Exécution
        sim.simulate_multiple_steps(steps=self.steps, generate_200ms_history=True)
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

        fig = plt.figure(figsize=(18, 14))
        gs = fig.add_gridspec(4, 2)

        # --- GRAPH 1: PnL Evolution ---
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(df_agg["timestamp"], df_agg["mtm_pnl"], label="MtM PnL", color="#1f77b4")
        ax1.fill_between(df_agg["timestamp"], df_agg["mtm_pnl"], alpha=0.2)
        ax1.set_title("Evolution du PnL Total (USD)", fontsize=14, fontweight='bold')
        ax1.set_ylabel("PnL ($)")

        # --- GRAPH 2: Inventory & Skew ---
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(df_rt["timestamp"], df_rt["inventory_pct"] * 100, color="#ff7f0e", label="Inventory % Usage")
        ax2.axhline(70, color='r', linestyle='--', alpha=0.5, label="Alert Threshold")
        ax2.set_title("Utilisation de la Limite d'Inventaire (%)", fontsize=14, fontweight='bold')
        ax2.set_ylim(0, 100)
        ax2.legend()

        # --- GRAPH 3: Price & Reservation Price ---
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(df_agg["timestamp"], df_agg["mid_A"], label="Mid A", color="black", alpha=0.3)
        ax3.plot(df_agg["timestamp"], df_agg["reservation_price"], label="Reservation Price", color="red", linestyle="--")
        ax3.set_title("Mid Price vs Reservation Price (Inventory Skew)", fontsize=14, fontweight='bold')
        ax3.legend()

        # --- GRAPH 4: Quoted Spread vs Sniping ---
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(df_agg["timestamp"], df_agg["spread_quoted"] * 10000, label="Spread A (pips)", color="green")
        ax4.set_title("Spread Quoté sur A (Pips)", fontsize=14, fontweight='bold')
        ax4.set_ylabel("Pips")

        # --- GRAPH 5: Sniping Activity ---
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.scatter(df_agg["timestamp"], df_agg["hft_snipe_count"], s=df_agg["hft_snipe_qty"]/1000, 
                    alpha=0.5, c="red", label="HFT Snipes")
        ax5.set_title("Activité de Sniping HFT (Taille = Volume)", fontsize=14, fontweight='bold')
        ax5.set_ylabel("Nombre d'attaques")

        # --- GRAPH 6: Fill Rates ---
        ax6 = fig.add_subplot(gs[2, 1])
        avg_fills = [df_agg["fill_rate_bid"].mean(), df_agg["fill_rate_ask"].mean()]
        ax6.bar(["Bid Fill Rate", "Ask Fill Rate"], avg_fills, color=["#1f77b4", "#d62728"])
        ax6.set_title("Taux d'exécution moyen (Fill Rates)", fontsize=14, fontweight='bold')
        ax6.set_ylim(0, max(avg_fills)*1.5 if any(avg_fills) else 1)

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
            the_table.set_fontsize(10)
            the_table.scale(1, 1.5)

        plt.tight_layout()
        plt.savefig("full_backtest_report.png", dpi=300)
        print(">>> Rapport sauvegardé sous 'full_backtest_report.png'")
        plt.show()

if __name__ == "__main__":
    # On simule 5000 steps = 50 secondes de marché à 10ms/step
    runner = BacktestRunner(steps=5000)
    simulator = runner.run_simulation()
    runner.analyze_and_plot(simulator)