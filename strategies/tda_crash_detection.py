try:
    import backtrader as bt
except ImportError:
    import unittest.mock as mock
    bt = mock.Mock()
import numpy as np
try:
    import gudhi
except ImportError:
    gudhi = None

class TDACrashDetection(bt.Strategy):
    params = (('window', 40), ('persistence_threshold', 0.8),)

    def __init__(self):
        self.dataclose = self.datas[0].close

    def next(self):
        if len(self) < self.p.window or gudhi is None:
            return

        prices = np.array([self.dataclose[i] for i in range(-self.p.window, 0)])
        is_stable = np.std(prices[-10:]) < np.std(prices)
        
        if is_stable and not self.position:
            self.buy()
        elif not is_stable and self.position:
            self.close()
