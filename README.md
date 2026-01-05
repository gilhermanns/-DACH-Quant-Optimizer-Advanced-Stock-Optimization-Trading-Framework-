# 📈 Institutional DACH Quant Optimizer

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Quant-Finance](https://img.shields.io/badge/Focus-Quant%20Finance-orange.svg)]()

An institutional-grade quantitative trading and portfolio optimization framework specifically engineered for the **DACH region (Germany, Austria, Switzerland)**. This system integrates advanced alpha-generating strategies with robust risk management and topological data analysis.

---

## 💡 Core Value Proposition

The DACH markets (XETRA, SIX) present unique liquidity and regime dynamics. This framework provides a professional research environment to:

| Feature | Benefit |
| :--- | :--- |
| **Multi-Strategy Alpha** | Includes 7 optimized strategies from ML-based regime detection to Stat-Arb. |
| **HRP Optimization** | Uses Hierarchical Risk Parity for superior portfolio stability vs. Mean-Variance. |
| **TDA Crash Detection** | Employs Topological Data Analysis to identify market instabilities before crashes. |
| **Institutional Backtesting** | Accounts for European commission models and adaptive slippage. |

---

## 🛠 Technical Architecture

### 1. Strategy Engine (`strategies/`)
A modular library of 7 high-conviction strategies:
*   **Swiss Regime SVM**: ML-based regime classification for the SMI.
*   **Deep Alpha (LSTM)**: Neural network for short-term price prediction.
*   **TDA Crash Signal**: Topological monitoring of market manifolds.
*   **Cross-Asset Stat-Arb**: Statistical arbitrage between DAX and international peers.

### 2. Portfolio Management (`portfolio.py`)
Implements **Hierarchical Risk Parity (HRP)**, which uses graph theory to cluster assets by correlation, ensuring true diversification even during high-correlation market events.

### 3. Data & Execution (`data_fetcher.py` & `backtester.py`)
Handles automated ticker mapping for XETRA/SIX and provides a high-fidelity backtesting environment with realistic execution constraints.

---

## 📊 Project Structure

```text
/DACH-Quant-Optimizer
├── README.md               # Comprehensive project documentation
├── LIMITATIONS.md          # Mature disclosure of system boundaries
├── requirements.txt        # Python dependencies
├── run_all.py              # Global strategy dashboard (Demo)
├── main.py                 # Strategy execution CLI
├── backtester.py           # Institutional backtesting engine
├── data_fetcher.py         # Automated DACH data ingestion
├── portfolio.py            # HRP portfolio optimizer
└── strategies/             # Modular strategy library
    ├── swiss_regime_svm.py
    ├── lstm_deep_alpha.py
    └── tda_crash_detection.py
```

---

## 🚦 Getting Started

### Prerequisites
*   Python 3.9+
*   `pandas`, `numpy`, `scikit-learn`, `backtrader`

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/gilhermanns/-DACH-Quant-Optimizer-Advanced-Stock-Optimization-Trading-Framework-.git
   cd -DACH-Quant-Optimizer-Advanced-Stock-Optimization-Trading-Framework-
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Dashboard
Execute the global comparison dashboard to see simulated performance across all DACH strategies:
```bash
python3 run_all.py
```

---

## 🛡 License & Disclaimer

This project is licensed under the MIT License. Quantitative trading involves significant risk. This framework is for research and educational purposes only.
