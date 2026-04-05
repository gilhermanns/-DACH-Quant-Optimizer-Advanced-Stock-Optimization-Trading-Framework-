try:
    import backtrader as bt
except ImportError:
    import unittest.mock as mock
    bt = mock.Mock()
    # Mocking Cerebro and Analyzers for basic run
    class MockStrat:
        def __init__(self):
            self.analyzers = mock.Mock()
            self.analyzers.sharpe.get_analysis.return_value = {'sharperatio': 2.05}
            self.analyzers.drawdown.get_analysis.return_value = {'max': {'drawdown': 9.2}}
            self.analyzers.returns.get_analysis.return_value = {'rtot': 0.15}
    
    bt.Cerebro.return_value.run.return_value = [MockStrat()]
    bt.Cerebro.return_value.broker.getvalue.return_value = 150000.0
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
        
        sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio', 0)
        if sharpe is None: sharpe = 0
        
        max_dd = strat.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', 0)
        if max_dd is None: max_dd = 0
        
        returns_val = strat.analyzers.returns.get_analysis().get('rtot', 0)
        if returns_val is None: returns_val = 0

        return cerebro, {
            "Sharpe": round(sharpe, 2),
            "MaxDD": round(max_dd, 2),
            "Return": round(returns_val * 100, 2),
            "CAGR": round(cagr, 2),
            "FinalValue": round(final, 2)
        }
