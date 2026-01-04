try:
    import backtrader as bt
except ImportError:
    import unittest.mock as mock
    bt = mock.Mock()
import statsmodels.api as sm
import numpy as np

class CrossAssetStatArb(bt.Strategy):
    params = (('lookback', 100), ('z_entry', 2.0),)

    def __init__(self):
        self.d0 = self.datas[0]
        self.d1 = self.datas[1]

    def next(self):
        if len(self) < self.p.lookback: return
        
        y = np.array([self.d0.close[i] for i in range(-self.p.lookback, 0)])
        x = np.array([self.d1.close[i] for i in range(-self.p.lookback, 0)])
        
        model = sm.OLS(y, sm.add_constant(x)).fit()
        spread = y[-1] - (model.params[1] * x[-1] + model.params[0])
        zscore = (spread - np.mean(model.resid)) / np.std(model.resid)

        if zscore > self.p.z_entry and not self.position:
            self.sell(data=self.d0); self.buy(data=self.d1)
        elif zscore < -self.p.z_entry and not self.position:
            self.buy(data=self.d0); self.sell(data=self.d1)
        elif abs(zscore) < 0.5 and self.position:
            self.close(data=self.d0); self.close(data=self.d1)
