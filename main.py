import argparse
from data_fetcher import AdvancedDACHFetcher
from backtester import InstitutionalBacktester
from strategies.alpha_ensemble_hybrid import AlphaEnsembleHybrid
from strategies.dach_midcap_liquidity import DACHMidCapLiquidity
from strategies.dax_momentum_adaptive import DAXMomentumAdaptive
from strategies.swiss_regime_svm import SwissRegimeSVM
from strategies.lstm_deep_alpha import LSTMDeepAlpha
from strategies.cross_asset_stat_arb import CrossAssetStatArb
from strategies.tda_crash_detection import TDACrashDetection

def main():
    parser = argparse.ArgumentParser(description="gilhermanns-QuantPortfolio: Professional Edition")
    parser.add_argument("--strategy", type=str, default="ensemble", 
                        choices=["ensemble", "midcap", "momentum", "regime", "lstm", "arb", "tda"])
    parser.add_argument("--asset", type=str, default="DAX", help="Ticker or Index (DAX, MDAX, SMI, SAP.DE)")
    parser.add_argument("--start", type=str, default="2021-01-01")
    parser.add_argument("--end", type=str, default="2025-12-23")
    
    args = parser.parse_args()
    
    print(f"--- [SYSTEM] Initializing {args.strategy.upper()} Strategy ---")
    
    fetcher = AdvancedDACHFetcher()
    data = fetcher.get_data([args.asset], args.start, args.end)
    
    if args.asset not in data:
        print(f"--- [ERROR] Data retrieval failed for {args.asset} ---")
        return

    engine = InstitutionalBacktester()
    
    strat_map = {
        "ensemble": AlphaEnsembleHybrid,
        "midcap": DACHMidCapLiquidity,
        "momentum": DAXMomentumAdaptive,
        "regime": SwissRegimeSVM,
        "lstm": LSTMDeepAlpha,
        "arb": CrossAssetStatArb,
        "tda": TDACrashDetection,
    }
    
    strat_class = strat_map.get(args.strategy, AlphaEnsembleHybrid)
    
    print(f"--- [RUN] Backtesting {args.asset} | {args.start} to {args.end} ---")
    cerebro, metrics = engine.run(strat_class, data[args.asset])
    
    print("\n" + "="*40)
    print("      INSTITUTIONAL PERFORMANCE REPORT")
    print("="*40)
    for k, v in metrics.items():
        print(f"{k:20}: {v}")
    print("="*40)

if __name__ == "__main__":
    main()
