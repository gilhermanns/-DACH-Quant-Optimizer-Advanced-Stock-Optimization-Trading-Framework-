# Institutional Quant Framework: Limitations & Scope

This framework is designed for quantitative research and strategy backtesting. While it implements advanced alpha-generating logic, users should be aware of the following limitations.

## 1. Execution & Slippage
- **Model vs. Reality**: The backtester uses an adaptive slippage model, but real-world execution in mid-cap stocks (MDAX/SDAX) can vary significantly during periods of low liquidity.
- **Latency**: The framework does not account for HFT-level latency or co-location advantages.

## 2. Data Dependencies
- **Source Quality**: The default `data_fetcher` uses public APIs. For institutional production, it is recommended to integrate with premium data providers (e.g., Bloomberg, Reuters) to ensure corporate action adjustments (dividends, splits) are handled with 100% accuracy.
- **Look-ahead Bias**: While the strategies are designed to be causal, users must ensure that custom features do not inadvertently introduce look-ahead bias during the feature engineering phase.

## 3. Model Risk
- **Overfitting**: ML-based strategies (LSTM, SVM) are susceptible to overfitting on historical regimes. Regular walk-forward optimization and stress testing are mandatory.
- **Regime Shifts**: The "Swiss Regime SVM" is trained on historical SMI dynamics. A fundamental shift in Swiss monetary policy or global trade could render historical classifications obsolete.

## 4. Regulatory & Compliance
- This software does not include compliance modules for MiFID II or other regional financial regulations. Users are responsible for ensuring their trading activities comply with local laws.

---
*Disclaimer: Quantitative trading involves significant risk. This framework is for research purposes only.*
