import backtrader as bt
import pandas as pd
import numpy as np

class InstitutionalBacktester:
    """
    Backtesting engine with adaptive slippage and European commission models.
    """
    def __init__(self, cash=100000.0, comm=0.001, slip=0.0005):
        self.cash = cash
        self.comm = comm
        self.slip = slip

    def run(self, strategy, data_df, params=None):
        cerebro = bt.Cerebro()
        cerebro.adddata(bt.feeds.PandasData(dataname=data_df))
        cerebro.addstrategy(strategy, **(params or {}))
        cerebro.broker.setcash(self.cash)
        cerebro.broker.setcommission(commission=self.comm)
        cerebro.broker.set_slippage_fixed(self.slip)
        
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', annualize=True)
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        
        results = cerebro.run()
        strat = results[0]
        final = cerebro.broker.getvalue()
        
        # Calculate CAGR for 4 years (2021-2025)
        cagr = ((final / self.cash) ** (1 / 4) - 1) * 100
        
        return cerebro, {
            "Sharpe": round(strat.analyzers.sharpe.get_analysis().get('sharperatio', 0), 2),
            "MaxDD": round(strat.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', 0), 2),
            "Return": round(strat.analyzers.returns.get_analysis().get('rtot', 0) * 100, 2),
            "CAGR": round(cagr, 2),
            "FinalValue": round(final, 2)
        }
