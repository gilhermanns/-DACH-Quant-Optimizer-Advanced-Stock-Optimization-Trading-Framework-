import backtrader as bt

class DAXMomentumAdaptive(bt.Strategy):
    params = (('fast', 10), ('slow', 50), ('atr_mult', 2.5),)

    def __init__(self):
        self.fast_ma = bt.indicators.SMA(period=self.p.fast)
        self.slow_ma = bt.indicators.SMA(period=self.p.slow)
        self.atr = bt.indicators.ATR(period=14)
        self.stop_price = None

    def next(self):
        if not self.position:
            if self.fast_ma > self.slow_ma:
                self.buy()
                self.stop_price = self.data.close[0] - (self.p.atr_mult * self.atr[0])
        else:
            new_stop = self.data.close[0] - (self.p.atr_mult * self.atr[0])
            self.stop_price = max(self.stop_price, new_stop)
            
            if self.data.close < self.stop_price or self.fast_ma < self.slow_ma:
                self.close()
