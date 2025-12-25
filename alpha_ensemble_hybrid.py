import backtrader as bt
import numpy as np

class AlphaEnsembleHybrid(bt.Strategy):
    params = (('w_mom', 0.4), ('w_mr', 0.3), ('w_vol', 0.3),)

    def __init__(self):
        self.sma_fast = bt.indicators.SMA(period=12)
        self.sma_slow = bt.indicators.SMA(period=48)
        self.rsi = bt.indicators.RSI(period=14)
        self.atr = bt.indicators.ATR(period=14)

    def next(self):
        sig_mom = 1 if self.sma_fast > self.sma_slow else -1
        sig_mr = 1 if self.rsi < 32 else (-1 if self.rsi > 68 else 0)
        vol_ratio = self.atr[0] / self.data.close[0]
        sig_vol = 1 if vol_ratio < np.mean(self.atr.get(size=20)) / np.mean(self.data.close.get(size=20)) else -0.5
        
        combined = (self.p.w_mom * sig_mom) + (self.p.w_mr * sig_mr) + (self.p.w_vol * sig_vol)
        
        if combined > 0.45 and not self.position:
            self.buy()
        elif combined < -0.15 and self.position:
            self.close()
