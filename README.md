# 📈 Stock Market Prediction and Analysis System

A machine learning-powered web application that fetches **real stock market data**, performs **technical analysis**, trains an **ML prediction model**, and displays everything in an **interactive Streamlit dashboard**.

> **MCA Final Year Project** — Built with Python, Streamlit, and Scikit-learn

---

## 🎯 Project Overview

This project demonstrates the application of machine learning to financial data analysis. It downloads real-time stock data from Yahoo Finance, calculates popular technical indicators (SMA, RSI, MACD), trains a Random Forest Regressor to predict future stock prices, and provides actionable Buy/Sell/Hold suggestions — all wrapped in a modern, user-friendly web interface.

---

## ✨ Features

| Feature | Description |
|---|---|
| **📡 Real-Time Data** | Fetches live stock data via Yahoo Finance (yfinance) |
| **📈 Interactive Charts** | Candlestick, OHLC, Volume charts using Plotly |
| **📊 Technical Indicators** | SMA-10, SMA-20, RSI (14), MACD with signal line |
| **🤖 ML Prediction** | Random Forest Regressor for next-day price prediction |
| **📏 Model Metrics** | MAE, RMSE, R² Score for model evaluation |
| **🔮 Future Forecast** | 7-day, 14-day, or 30-day price forecast |
| **💡 Trade Suggestion** | Buy / Sell / Hold recommendation based on indicators + ML |
| **🏢 Stock Info** | Company name, sector, market cap, P/E ratio |
| **🌍 Global Support** | Works with NSE (India), NYSE, NASDAQ, and more |
| **🎨 Modern UI** | Dark theme, responsive layout, gradient accents |

---

## 🛠️ Tech Stack

- **Language:** Python 3.9+
- **Frontend:** Streamlit
- **Data Source:** Yahoo Finance (via `yfinance`)
- **ML Model:** Random Forest Regressor (scikit-learn)
- **Charts:** Plotly
- **Libraries:** pandas, numpy, matplotlib

---

## 📁 Project Structure

```
stock-market-prediction/
├── app.py              # Main Streamlit application (entry point)
├── data_loader.py      # Fetches stock data from Yahoo Finance
├── indicators.py       # Calculates SMA, RSI, MACD indicators
├── model.py            # ML model training, evaluation & forecasting
├── utils.py            # Plotly chart builders & formatting utilities
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation (this file)
└── .streamlit/
    └── config.toml     # Streamlit theme configuration
```

### Module Descriptions

| Module | Responsibility |
|---|---|
| `app.py` | Orchestrates the entire dashboard — sidebar inputs, data flow, chart rendering |
| `data_loader.py` | Downloads OHLCV data and stock info from Yahoo Finance |
| `indicators.py` | Computes SMA (10 & 20 day), RSI (14 day), and MACD |
| `model.py` | Feature engineering, Random Forest training, evaluation, and future forecasting |
| `utils.py` | All Plotly chart functions and number formatting helpers |

---

## 🚀 Installation & Setup

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/stock-market-prediction.git
cd stock-market-prediction
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Application

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`.

---

## 📖 How to Use

1. **Enter a ticker symbol** in the sidebar (e.g., `RELIANCE.NS`, `TCS.NS`, `AAPL`, `MSFT`).
2. **Select a date range** — at least 6 months of data is recommended for good ML results.
3. **Choose forecast period** — 7, 14, or 30 days.
4. **Click "🚀 Run Analysis"** to start the analysis.
5. **Explore the tabs** — Price Charts, Technical Indicators, Raw Data.
6. **Review ML predictions** — Model metrics, actual vs predicted chart, feature importance.
7. **Check the forecast** — Future price trend with detailed table.
8. **Read the suggestion** — Buy / Sell / Hold recommendation with supporting reasons.

### 🇮🇳 Indian Stock Examples (NSE)

| Company | Ticker |
|---|---|
| Reliance Industries | `RELIANCE.NS` |
| Tata Consultancy Services | `TCS.NS` |
| Infosys | `INFY.NS` |
| HDFC Bank | `HDFCBANK.NS` |
| Wipro | `WIPRO.NS` |

### 🇺🇸 US Stock Examples

| Company | Ticker |
|---|---|
| Apple | `AAPL` |
| Microsoft | `MSFT` |
| Google | `GOOGL` |
| Amazon | `AMZN` |
| Tesla | `TSLA` |

---

## 📸 Screenshots

> *Add screenshots of your running application here for the project report.*

| Screenshot | Description |
|---|---|
| `screenshots/landing.png` | Landing page with ticker guide |
| `screenshots/dashboard.png` | Main analysis dashboard |
| `screenshots/indicators.png` | Technical indicators tab |
| `screenshots/prediction.png` | ML prediction results |
| `screenshots/forecast.png` | Future price forecast |
| `screenshots/suggestion.png` | Buy/Sell/Hold suggestion |

---

## 🧠 How the ML Model Works

### Feature Engineering

The model uses the following features derived from raw stock data:

- **Lag Features:** Closing prices from 1, 2, 3, 5, and 10 days ago
- **Daily Return:** Percentage change from the previous day
- **Volatility:** 10-day rolling standard deviation of returns
- **Momentum:** Price change over 5 days
- **High-Low Spread:** Intraday trading range
- **Volume Change:** Day-over-day volume change
- **Calendar Features:** Day of week and month
- **Technical Indicators:** SMA, RSI, MACD values

### Model Choice: Random Forest Regressor

- Handles non-linear patterns effectively
- Robust to noisy financial data
- Does not require feature scaling
- Provides feature importance rankings

### Evaluation Metrics

| Metric | What It Measures |
|---|---|
| **MAE** | Average absolute prediction error |
| **RMSE** | Root mean squared error (penalizes large errors) |
| **R² Score** | Proportion of variance explained (1.0 = perfect) |

---

## 🔮 Future Improvements

- [ ] Add more ML models (LSTM, XGBoost, GRU) for comparison
- [ ] Implement portfolio tracking for multiple stocks
- [ ] Add sentiment analysis from financial news (NLP)
- [ ] Include Bollinger Bands, Fibonacci retracement levels
- [ ] Add real-time price alerts and notifications
- [ ] Deploy on cloud (Streamlit Cloud, AWS, or Heroku)
- [ ] Add user authentication and saved watchlists
- [ ] Support for cryptocurrency analysis
- [ ] Backtesting engine for strategy validation
- [ ] Export reports as PDF

---

## ⚠️ Disclaimer

This project is built **for educational and academic purposes only**. The predictions and buy/sell/hold suggestions generated by this application are based on simplified machine learning models and should **NOT** be used for actual financial trading or investment decisions. Always consult a certified financial advisor before making investment decisions.

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

- [Yahoo Finance](https://finance.yahoo.com/) — Real-time stock data
- [Streamlit](https://streamlit.io/) — Web application framework
- [scikit-learn](https://scikit-learn.org/) — Machine learning library
- [Plotly](https://plotly.com/) — Interactive charting library
- [yfinance](https://github.com/ranaroussi/yfinance) — Yahoo Finance API wrapper

---

<p align="center">
  Made with ❤️ for MCA Final Year Project
</p>
