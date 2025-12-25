import backtrader as bt
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
        pass
