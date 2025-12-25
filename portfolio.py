import numpy as np
import pandas as pd
from pypfopt import HRPOpt, expected_returns, risk_models

class InstitutionalPortfolio:
    """
    Hierarchical Risk Parity (HRP) Portfolio Management.
    Optimized for cross-asset DACH portfolios to minimize tail risk.
    """
    def __init__(self, returns_df):
        self.returns = returns_df

    def optimize(self):
        """
        Computes HRP weights.
        HRP is superior to Mean-Variance as it doesn't require matrix inversion.
        """
        hrp = HRPOpt(self.returns)
        weights = hrp.optimize()
        return dict(weights)

    def rebalance(self, current_weights, target_weights):
        """
        Calculates necessary adjustments to reach target allocation.
        """
        adjustments = {}
        for asset, target in target_weights.items():
            diff = target - current_weights.get(asset, 0)
            if abs(diff) > 0.02: # 2% rebalancing threshold
                adjustments[asset] = diff
        return adjustments
