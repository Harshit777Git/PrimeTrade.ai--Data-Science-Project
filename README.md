# PrimeTrade.ai--Data-Science-Project
# 📈 Hyperliquid Trader Behavior & Sentiment Analysis

An end-to-end data analysis and machine learning project exploring the relationship between the **Crypto Fear & Greed Index** and trader performance on the **Hyperliquid** perpetual exchange.

## 🚀 Project Overview
This project investigates whether market sentiment (Fear & Greed) drives distinct behavioral regimes among traders. By aligning historical trade executions with daily sentiment scores, we identify profitable trader segments and propose actionable automated trading strategies.

### Core Objectives:
* **Data Pipeline:** Align and clean disparate datasets (Trade History + Sentiment Index).
* **Behavioral Analysis:** Quantify the impact of sentiment on Daily PnL, Win Rate, and Trade Frequency.
* **Trader Segmentation:** Cluster users into archetypes using Machine Learning.
* **Strategy Formulation:** Propose data-backed "Rules of Thumb" for algorithmic trading.

---

## 📊 Key Findings & Insights
1. **The Panic Paradox:** Trade frequency spikes by **~37%** during "Extreme Fear" days. While retail sentiment is panicked, professional segments show the highest PnL during these periods by providing liquidity.
2. **Contrarian Bias:** Top-performing traders act as contrarians—shifting to a Short-bias during "Extreme Greed" and a Long-bias during "Fear" windows.
3. **Regime Impact:** Sentiment extremes (Fear/Greed) create **32.3% higher PnL variance** compared to neutral markets, confirming that "Sentiment Regimes" are a valid signal for risk management.

---

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Data Science:** Pandas, NumPy, Scikit-Learn
* **Visualization:** Plotly (Interactive), Seaborn, Matplotlib
* **Deployment:** Streamlit (Dashboard), Ngrok/Localtunnel

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone [https://github.com/yourusername/hyperliquid-sentiment-analysis.git](https://github.com/yourusername/hyperliquid-sentiment-analysis.git)
cd hyperliquid-sentiment-analysis

##2. Install Dependencies

pip install -r requirements.txt

3. Run the Dashboard (Streamlit)
If running locally:

streamlit run app.py
📂 Project Structure

├── data/
│   ├── historical_data.csv        # Raw Hyperliquid trade history
│   ├── fear_greed_index.csv       # Daily Sentiment values
│   ├── trader_daily_metrics.csv   # Processed account-level daily data
│   └── trader_archetypes.csv      # ML-generated behavioral clusters
├── app.py                         # Streamlit Dashboard source code
├── Analysis_Notebook.ipynb        # Full methodology & ML Pipeline
├── requirements.txt               # Required Python packages
└── README.md                      # Project documentation

💡 Proposed Strategies (Part C)
Rule 1: The "Fear" Liquidity Provision Rule
Finding: High volume and high frequency are detected during market panics.

Action: During Fear days, increase trade frequency by 15% but cap Implied Leverage at 5x to capture volatility without liquidation risk.

Rule 2: The "Greed" De-Risking Protocol
Finding: Win rates peak but Long/Short bias becomes dangerously skewed.

Action: When sentiment index > 75 (Extreme Greed), implement a mandatory 30% reduction in position size for high-leverage segments.

🧪 Methodology
Data Alignment: Cleaned and normalized trade data to a daily grain; merged with sentiment data via date_key.

Feature Engineering: Created metrics for Implied Leverage, Win Rate, and Long/Short Ratio.

Clustering: Used K-Means to identify three archetypes: High-Stakes Scalpers, Professionals, and Retail Explorers.

Validation: Used if-else logical loops to determine if performance gaps between segments were statistically actionable.
