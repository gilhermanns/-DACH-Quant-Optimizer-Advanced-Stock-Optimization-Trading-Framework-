import pandas as pd
import numpy as np
from main import main as run_strategy_cli
from backtester import InstitutionalBacktester
from data_fetcher import AdvancedDACHFetcher
from strategies.alpha_ensemble_hybrid import AlphaEnsembleHybrid
from strategies.dach_midcap_liquidity import DACHMidCapLiquidity
from strategies.dax_momentum_adaptive import DAXMomentumAdaptive
from strategies.swiss_regime_svm import SwissRegimeSVM
from strategies.lstm_deep_alpha import LSTMDeepAlpha
from strategies.cross_asset_stat_arb import CrossAssetStatArb
from strategies.tda_crash_detection import TDACrashDetection

def run_dashboard():
    print("🚀 Running Global DACH Strategy Comparison Dashboard...")
    
    # Define the core strategies and their primary assets
    strategies = [
        ("Alpha Ensemble", AlphaEnsembleHybrid, "DAX"),
        ("Mid-Cap Liquidity", DACHMidCapLiquidity, "MDAX"),
        ("Swiss SVM Regime", SwissRegimeSVM, "SMI"),
        ("Deep Alpha (LSTM)", LSTMDeepAlpha, "SAP.DE"),
    ]
    
    results = []
    fetcher = AdvancedDACHFetcher()
    engine = InstitutionalBacktester()
    
    for name, strat_class, asset in strategies:
        print(f"  -> Backtesting {name} on {asset}...")
        try:
            data = fetcher.get_data([asset], "2021-01-01", "2025-12-23")
            if asset not in data:
                raise Exception(f"No data for {asset}")
            
            _, metrics = engine.run(strat_class, data[asset])
            
            results.append({
                "Strategy": name,
                "Asset": asset,
                "Sharpe": metrics['Sharpe'],
                "CAGR (%)": metrics['CAGR'],
                "MaxDD (%)": metrics['MaxDD']
            })
        except Exception as e:
            print(f"  -> FAILED to run {name}: {e}")
            results.append({"Strategy": name, "Asset": asset, "Sharpe": 0, "CAGR (%)": 0, "MaxDD (%)": 0})

    df = pd.DataFrame(results)
    print("\n" + "="*70)
    print("                      STRATEGY COMPARISON DASHBOARD")
    print("="*70)
    print(df.to_string(index=False))
    print("="*70)

if __name__ == "__main__":
    run_dashboard()
