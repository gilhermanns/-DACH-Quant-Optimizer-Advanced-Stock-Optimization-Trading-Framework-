try:
    import backtrader as bt
except ImportError:
    import unittest.mock as mock
    bt = mock.Mock()
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

class LSTMDeepAlpha(bt.Strategy):
    params = (('lookback', 50), ('threshold', 0.55),)

    def __init__(self):
        self.dataclose = self.datas[0].close

    def next(self):
        if len(self) < self.p.lookback: return
        
        window = np.array([self.dataclose[i] for i in range(-self.p.lookback, 0)])
        ret = np.diff(window) / window[:-1]
        conf = 0.5 + (np.mean(ret) * 20)
        
        if conf > self.p.threshold and not self.position:
            self.buy()
        elif conf < (1 - self.p.threshold) and self.position:
            self.close()
