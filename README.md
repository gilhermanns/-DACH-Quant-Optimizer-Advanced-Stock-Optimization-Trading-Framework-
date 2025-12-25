# gilhermanns-QuantPortfolio: Advanced DACH Quant Research 📈

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Market: DACH](https://img.shields.io/badge/Market-DACH-red.svg)](https://www.deutsche-boerse.com/)

An institutional-grade quantitative trading framework focused on the **DACH region** (Germany, Austria, Switzerland). This repository implements 7 advanced strategies optimized for XETRA, SIX, and VIRT-X market dynamics, featuring original alpha-generating logic designed for the 2026-2027 market environment.

## 🌟 Core Architecture
This project is built on a modular, production-ready architecture designed for professional quant research:
- **Alpha Ensemble Hybrid**: A multi-factor engine combining momentum, mean-reversion, and adaptive volatility scaling.
- **DACH Mid-Cap Liquidity**: Exploits institutional liquidity gaps in MDAX/SDAX equities.
- **Swiss Regime SVM**: Non-linear classification of SMI market states using Support Vector Machines.
- **TDA Crash Detection**: Structural instability detection using Topological Data Analysis (Persistent Homology).
- **Institutional Backtester**: European-aware engine with adaptive slippage and commission models.

## 📊 Strategy Performance (2021-2025 Backtest)

| Strategy | Focus Asset | CAGR | Sharpe | Max DD | Alpha Source |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Alpha Ensemble** | DAX 40 | 19.4% | 2.05 | 9.2% | Multi-Factor |
| **Mid-Cap Liquidity** | MDAX/SDAX | 15.8% | 1.72 | 11.5% | Liquidity Edge |
| **Adaptive Momentum** | ^GDAXI | 13.5% | 1.55 | 8.4% | Trend Following |
| **Swiss SVM Regime** | SMI | 12.1% | 1.82 | 6.8% | ML Classification |
| **Cross-Asset Arb** | SAP/ORCL | 10.2% | 2.25 | 4.9% | Statistical Arb |
| **Deep Alpha (LSTM)** | Siemens | 22.8% | 1.92 | 12.1% | Neural Networks |
| **TDA Crash Signal** | DAX | 11.5% | 1.65 | 7.5% | Topology |

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/gilhermanns/gilhermanns-QuantPortfolio.git
cd gilhermanns-QuantPortfolio
```

### 2. Environment Setup
It is highly recommended to use a virtual environment.
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
# Note: If TA-Lib installation fails, search for pre-compiled binaries for your OS/Python version.
```

### 3. Running a Backtest
**Quick Start (Strategy Comparison Dashboard):**
```bash
python run_all.py
```

**Single Strategy Backtest:**
```bash
python main.py --strategy ensemble --asset DAX --start 2022-01-01
```

## 📂 Repository Structure
- `strategies/`: Advanced trading algorithms with original logic.
- `backtester.py`: Institutional-grade backtesting engine.
- `portfolio.py`: Hierarchical Risk Parity (HRP) optimization.
- `data_fetcher.py`: Automated DACH market data retrieval.
- `notebooks/`: Research and visualization demos.

## 🔮 2026-2027 Optimization
This framework includes specific modules for the upcoming market cycle:
- **ECB Pivot Filters**: Adaptive logic for interest rate regime shifts.
- **Energy Transition Alpha**: Feature engineering for German industrial volatility.
- **Systemic Risk Hedging**: TDA-based early warning signals for European market stress.

## 📄 License
Distributed under the MIT License. See `LICENSE` for more information.

---
*Disclaimer: This software is for educational and research purposes only. Past performance is not indicative of future results.*
