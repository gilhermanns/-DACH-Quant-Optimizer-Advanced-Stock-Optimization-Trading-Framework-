try:
    import backtrader as bt
except ImportError:
    import unittest.mock as mock
    bt = mock.Mock()

class DACHMidCapLiquidity(bt.Strategy):
    params = (('period', 22), ('std_dev', 2.1),)

    def __init__(self):
        self.bband = bt.indicators.BollingerBands(period=self.p.period, devfactor=self.p.std_dev)
        self.vol_ma = bt.indicators.SMA(self.data.volume, period=20)

    def next(self):
        if not self.position:
            if self.data.close < self.bband.lines.bot and self.data.volume[0] > 1.5 * self.vol_ma[0]:
                self.buy()
        else:
            if self.data.close > self.bband.lines.mid:
                self.close()
