try:
    import backtrader as bt
except ImportError:
    import unittest.mock as mock
    bt = mock.Mock()
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

class SwissRegimeSVM(bt.Strategy):
    params = (('lookback', 60), ('retrain_days', 22),)

    def __init__(self):
        self.svm = SVC(kernel='rbf', C=1.2, gamma='scale')
        self.scaler = StandardScaler()
        self.ready = False

    def next(self):
        if len(self) < 252: return
        if not self.ready or len(self) % self.p.retrain_days == 0:
            self._train()
            self.ready = True

        feat = self._get_feat()
        regime = self.svm.predict(self.scaler.transform([feat]))[0]

        if regime == 1 and not self.position:
            self.buy()
        elif regime == 0 and self.position:
            self.close()

    def _get_feat(self):
        rets = np.array([self.data.close[i] for i in range(-20, 0)])
        return [np.std(rets), (rets[-1]-rets[0])/rets[0]]

    def _train(self):
        # Collect historical data for training
        closes = np.array([self.data.close[i] for i in range(-252, 0)])
        returns = np.diff(closes) / closes[:-1]
        
        X = []
        y = []
        
        # Simple labeling: 1 if next 5 days return is positive, else 0
        for i in range(20, len(returns) - 5):
            window = returns[i-20:i]
            vol = np.std(window)
            mom = (closes[i] - closes[i-20]) / closes[i-20]
            X.append([vol, mom])
            
            future_ret = (closes[i+5] - closes[i]) / closes[i]
            y.append(1 if future_ret > 0 else 0)
            
        if len(X) > 50:
            X = np.array(X)
            y = np.array(y)
            self.scaler.fit(X)
            self.svm.fit(self.scaler.transform(X), y)
            self.ready = True
