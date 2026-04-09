"""
app.py — Stock Market Prediction and Analysis System
=====================================================
Main Streamlit application that ties all modules together.

Run with:
    streamlit run app.py

Modules used:
    data_loader.py  → fetches stock data from Yahoo Finance
    indicators.py   → calculates SMA, RSI, MACD
    model.py        → ML model training, evaluation, and forecasting
    utils.py        → Plotly charts and formatting helpers
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# Import project modules
from data_loader import fetch_stock_data, get_stock_info
from indicators import add_all_indicators
from model import prepare_features, train_model, get_feature_importance, forecast_future, generate_suggestion
from utils import (
    plot_candlestick,
    plot_ohlc_lines,
    plot_moving_averages,
    plot_rsi,
    plot_macd,
    plot_predictions,
    plot_forecast,
    plot_feature_importance,
    format_large_number,
    COLORS,
)


# ──────────────────────────────────────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Stock Market Prediction & Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS for a polished, modern look
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* ── Import Google Font ─────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Global ─────────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Header area ────────────────────────────────── */
    .main-header {
        background: linear-gradient(135deg, #6C63FF 0%, #FF6584 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0;
    }
    .sub-header {
        text-align: center;
        color: #888;
        font-size: 1rem;
        margin-top: -10px;
        margin-bottom: 30px;
    }

    /* ── Metric cards ───────────────────────────────── */
    .metric-card {
        background: linear-gradient(135deg, #1A1D29 0%, #252836 100%);
        border: 1px solid #2A2D3A;
        border-radius: 16px;
        padding: 20px 24px;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 30px rgba(108, 99, 255, 0.15);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #FAFAFA;
        margin-bottom: 4px;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* ── Suggestion card ────────────────────────────── */
    .suggestion-card {
        background: linear-gradient(135deg, #1A1D29 0%, #252836 100%);
        border: 1px solid #2A2D3A;
        border-radius: 16px;
        padding: 28px;
        margin-top: 10px;
    }
    .suggestion-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .suggestion-confidence {
        font-size: 0.9rem;
        color: #888;
        margin-bottom: 16px;
    }
    .reason-item {
        padding: 8px 14px;
        margin: 6px 0;
        background: rgba(108, 99, 255, 0.08);
        border-left: 3px solid #6C63FF;
        border-radius: 0 8px 8px 0;
        color: #CCC;
        font-size: 0.9rem;
    }

    /* ── Section divider ────────────────────────────── */
    .section-divider {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #6C63FF, transparent);
        margin: 40px 0;
    }

    /* ── Sidebar styling ────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: #13151D;
    }

    /* ── Dataframe styling ──────────────────────────── */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }

    /* ── Tab styling ────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0;
        padding: 10px 24px;
    }
    
    /* ── Primary run analysis button styling ────────── */
    div[data-testid="stSidebar"] div.stButton > button {
        font-weight: 600;
    }
    
    /* ── Landing page buttons styling ───────────────── */
    .landing-btn-container div.stButton > button {
        background: #1A1D29;
        border: 1px solid #2A2D3A;
        border-radius: 12px;
        padding: 24px 20px;
        color: #FAFAFA;
        font-size: 1.1rem;
        font-weight: 500;
        height: auto;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 12px;
        transition: all 0.2s ease;
    }
    .landing-btn-container div.stButton > button:hover {
        border-color: #6C63FF;
        background: #252836;
        box-shadow: 0 8px 30px rgba(108, 99, 255, 0.15);
        color: #fff;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────────

st.markdown('<h1 class="main-header">📈 Stock Market Prediction & Analysis</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-powered stock analysis with real-time data, technical indicators, and ML predictions</p>', unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar — User Inputs
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")

    # Load preset ticker suggestions
    try:
        with open("comprehensive_tickers.json", "r") as f:
            preset_tickers = json.load(f)
    except Exception:
        # Fallback
        preset_tickers = {
            "NIFTY 50 (Index)": "^NSEI",
            "BSE SENSEX (Index)": "^BSESN",
            "Reliance Industries": "RELIANCE.NS",
            "TCS": "TCS.NS",
            "Gold (COMEX)": "GC=F",
            "Apple": "AAPL",
            "Crude Oil": "CL=F"
        }

    # Quick-select from presets
    st.markdown("**Quick Select**")
    selected_preset = st.selectbox(
        "Choose a popular stock",
        options=["Custom"] + list(preset_tickers.keys()),
        index=0,
        help="Select a popular stock or choose 'Custom' to enter your own ticker.",
    )

    # Ticker input
    if selected_preset != "Custom":
        ticker_input = st.text_input(
            "📌 Stock Ticker Symbol",
            value=preset_tickers[selected_preset],
            help="Yahoo Finance ticker (e.g., RELIANCE.NS for NSE, AAPL for NASDAQ)",
        )
    else:
        ticker_input = st.text_input(
            "📌 Stock Ticker Symbol",
            value="RELIANCE.NS",
            help="Yahoo Finance ticker (e.g., RELIANCE.NS for NSE, AAPL for NASDAQ)",
        )

    st.markdown("---")

    # Date range selection
    st.markdown("**📅 Date Range**")
    col_start, col_end = st.columns(2)
    with col_start:
        start_date = st.date_input(
            "Start",
            value=datetime.now() - timedelta(days=365 * 2),  # 2 years ago
            max_value=datetime.now(),
        )
    with col_end:
        end_date = st.date_input(
            "End",
            value=datetime.now(),
            max_value=datetime.now(),
        )

    st.markdown("---")

    # Forecast configuration
    st.markdown("**🔮 Forecast Settings**")
    forecast_days = st.selectbox(
        "Forecast Period",
        options=[7, 14, 30],
        index=0,
        help="Number of future trading days to predict.",
    )

    st.markdown("---")

    # Run button
    if st.button(
        "🚀 Run Analysis",
        type="primary",
        use_container_width=True,
    ):
        st.session_state.run_dashboard = True

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#666; font-size:0.75rem;'>"
        "Built with ❤️ using Python, Streamlit & ML<br>"
        "MCA Final Year Project"
        "</div>",
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main Analysis Flow
# ──────────────────────────────────────────────────────────────────────────────

if st.session_state.get('run_dashboard', False):
    ticker = ticker_input.strip().upper()

    if not ticker:
        st.error("⚠️ Please enter a valid stock ticker symbol.")
        st.stop()

    if start_date >= end_date:
        st.error("⚠️ Start date must be before end date.")
        st.stop()

    # ── Step 1: Fetch Data ──────────────────────────────────────────────────
    with st.spinner("📡 Fetching stock data from Yahoo Finance..."):
        try:
            df = fetch_stock_data(ticker, str(start_date), str(end_date))
            stock_info = get_stock_info(ticker)
        except ValueError as e:
            st.error(f"❌ {str(e)}")
            st.stop()

    # ── Stock Info Cards ────────────────────────────────────────────────────
    st.markdown(f"### 🏢 {stock_info.get('name', ticker)}")

    info_cols = st.columns(5)
    info_items = [
        ("Sector", stock_info.get("sector", "N/A")),
        ("Industry", stock_info.get("industry", "N/A")),
        ("Currency", stock_info.get("currency", "N/A")),
        ("Market Cap", format_large_number(stock_info.get("market_cap", "N/A"))),
        ("P/E Ratio", f"{stock_info.get('pe_ratio', 'N/A')}"),
    ]
    for col, (label, value) in zip(info_cols, info_items):
        with col:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value">{value}</div>'
                f'<div class="metric-label">{label}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── Price Summary Metrics ───────────────────────────────────────────────
    st.markdown("")

    # Find the last row with a valid (non-NaN, non-zero) closing price
    valid_close = df[df["Close"].notna() & (df["Close"] > 0)]
    if len(valid_close) >= 2:
        latest = valid_close.iloc[-1]
        prev = valid_close.iloc[-2]
    elif len(valid_close) == 1:
        latest = valid_close.iloc[-1]
        prev = latest
    else:
        latest = df.iloc[-1]
        prev = latest

    # Safely compute price change
    close_val = float(latest["Close"]) if pd.notna(latest["Close"]) else 0
    prev_val = float(prev["Close"]) if pd.notna(prev["Close"]) else close_val
    high_val = float(latest["High"]) if pd.notna(latest.get("High")) else close_val
    low_val = float(latest["Low"]) if pd.notna(latest.get("Low")) else close_val
    price_change = close_val - prev_val
    price_change_pct = (price_change / prev_val * 100) if prev_val != 0 else 0

    price_cols = st.columns(4)
    price_items = [
        ("Current Price", f"{close_val:.2f}", ""),
        ("Day Change", f"{price_change:+.2f}", f"({price_change_pct:+.2f}%)"),
        ("Day High", f"{high_val:.2f}", ""),
        ("Day Low", f"{low_val:.2f}", ""),
    ]
    for col, (label, value, extra) in zip(price_cols, price_items):
        with col:
            try:
                is_positive = "+" in value or float(value.replace("+", "").replace(",", "")) >= 0
            except ValueError:
                is_positive = True
            color = "#00C9A7" if is_positive else "#FF6B6B"
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value" style="color:{color if label == "Day Change" else "#FAFAFA"}">{value} <span style="font-size:0.9rem;">{extra}</span></div>'
                f'<div class="metric-label">{label}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Step 2: Calculate Technical Indicators ──────────────────────────────
    with st.spinner("📊 Calculating technical indicators..."):
        df_with_indicators = add_all_indicators(df)

    # ── Step 3: Tabs for Charts ─────────────────────────────────────────────
    tab_price, tab_indicators, tab_data = st.tabs([
        "📈 Price Charts",
        "📊 Technical Indicators",
        "📋 Raw Data",
    ])

    # --- Tab: Price Charts ---
    with tab_price:
        # Candlestick chart
        st.plotly_chart(
            plot_candlestick(df, f"{ticker} — Candlestick Chart"),
            use_container_width=True,
        )

        # OHLC line chart
        st.plotly_chart(
            plot_ohlc_lines(df),
            use_container_width=True,
        )

    # --- Tab: Technical Indicators ---
    with tab_indicators:
        st.plotly_chart(
            plot_moving_averages(df_with_indicators),
            use_container_width=True,
        )

        ind_col1, ind_col2 = st.columns(2)
        with ind_col1:
            st.plotly_chart(
                plot_rsi(df_with_indicators),
                use_container_width=True,
            )
        with ind_col2:
            st.plotly_chart(
                plot_macd(df_with_indicators),
                use_container_width=True,
            )

        # Latest indicator values
        st.markdown("#### 📐 Latest Indicator Values")
        last_row = df_with_indicators.iloc[-1]
        ind_metric_cols = st.columns(5)
        ind_metrics = [
            ("SMA-10", f"{last_row.get('SMA_10', 'N/A'):.2f}" if pd.notna(last_row.get("SMA_10")) else "N/A"),
            ("SMA-20", f"{last_row.get('SMA_20', 'N/A'):.2f}" if pd.notna(last_row.get("SMA_20")) else "N/A"),
            ("RSI (14)", f"{last_row.get('RSI', 'N/A'):.2f}" if pd.notna(last_row.get("RSI")) else "N/A"),
            ("MACD", f"{last_row.get('MACD', 'N/A'):.4f}" if pd.notna(last_row.get("MACD")) else "N/A"),
            ("MACD Signal", f"{last_row.get('MACD_Signal', 'N/A'):.4f}" if pd.notna(last_row.get("MACD_Signal")) else "N/A"),
        ]
        for col, (label, value) in zip(ind_metric_cols, ind_metrics):
            with col:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-value" style="font-size:1.3rem;">{value}</div>'
                    f'<div class="metric-label">{label}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # --- Tab: Raw Data ---
    with tab_data:
        st.markdown("#### 📋 Historical Stock Data")
        st.dataframe(
            df_with_indicators.style.format("{:.2f}", subset=[
                c for c in df_with_indicators.columns if c != "Volume"
            ]).format("{:,.0f}", subset=["Volume"] if "Volume" in df_with_indicators.columns else []),
            use_container_width=True,
            height=400,
        )
        st.markdown(f"*Showing **{len(df_with_indicators)}** trading days of data.*")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Step 4: ML Model Training ───────────────────────────────────────────
    st.markdown("## 🤖 Machine Learning Prediction")

    # Check we have enough data for the model
    min_required = 60  # Need at least 60 rows after feature engineering
    if len(df_with_indicators) < min_required:
        st.warning(
            f"⚠️ Insufficient data for ML modeling. Need at least {min_required} trading days, "
            f"but only have {len(df_with_indicators)}. Please select a wider date range."
        )
        st.stop()

    with st.spinner("🧠 Preparing features and training the ML model..."):
        # Prepare features from indicator-enriched data
        df_features = prepare_features(df_with_indicators)

        # Train the Random Forest model
        model_result = train_model(df_features)

    # ── Model Evaluation Metrics ────────────────────────────────────────────
    st.markdown("### 📏 Model Evaluation Metrics")
    eval_cols = st.columns(3)

    eval_metrics = [
        ("MAE (Mean Absolute Error)", f"{model_result['mae']:.4f}", "Lower is better — average prediction error"),
        ("RMSE (Root Mean Squared Error)", f"{model_result['rmse']:.4f}", "Lower is better — penalizes large errors"),
        ("R² Score", f"{model_result['r2']:.4f}", "Closer to 1.0 is better — explained variance"),
    ]

    for col, (label, value, desc) in zip(eval_cols, eval_metrics):
        with col:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value" style="color:#6C63FF;">{value}</div>'
                f'<div class="metric-label">{label}</div>'
                f'<div style="font-size:0.75rem; color:#666; margin-top:8px;">{desc}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── Actual vs Predicted Chart ───────────────────────────────────────────
    st.markdown("### 🎯 Actual vs Predicted Prices (Test Set)")

    # Get the dates for the test set
    test_size = len(model_result["y_test"])
    test_dates = df_features.index[-test_size:]

    st.plotly_chart(
        plot_predictions(model_result["y_test"], model_result["y_pred"], test_dates),
        use_container_width=True,
    )

    # ── Feature Importance ──────────────────────────────────────────────────
    importance_df = get_feature_importance(model_result)

    with st.expander("🔍 Feature Importance — What drives the predictions?", expanded=False):
        st.plotly_chart(
            plot_feature_importance(importance_df),
            use_container_width=True,
        )

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Step 5: Future Forecast ─────────────────────────────────────────────
    st.markdown(f"## 🔮 {forecast_days}-Day Price Forecast")

    with st.spinner(f"Generating {forecast_days}-day forecast..."):
        forecast_df = forecast_future(df_features, model_result, days=forecast_days)

    st.plotly_chart(
        plot_forecast(df, forecast_df, f"{forecast_days}-Day"),
        use_container_width=True,
    )

    # Forecast table
    with st.expander("📋 Forecast Details", expanded=True):
        display_forecast = forecast_df.copy()
        display_forecast.index = display_forecast.index.strftime("%Y-%m-%d (%A)")
        display_forecast.columns = ["Predicted Close Price"]
        st.dataframe(
            display_forecast.style.format("{:.2f}"),
            use_container_width=True,
        )

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Step 6: Buy / Sell / Hold Suggestion ────────────────────────────────
    st.markdown("## 💡 Buy / Sell / Hold Suggestion")

    suggestion = generate_suggestion(df_with_indicators, forecast_df)

    st.markdown(
        f'<div class="suggestion-card">'
        f'<div class="suggestion-title">{suggestion["suggestion"]}</div>'
        f'<div class="suggestion-confidence">Confidence: {suggestion["confidence"]:.0f}%</div>'
        + "".join(
            f'<div class="reason-item">• {reason}</div>' for reason in suggestion["reasons"]
        )
        + "</div>",
        unsafe_allow_html=True,
    )

    # ── Disclaimer ──────────────────────────────────────────────────────────
    st.markdown("")
    st.info(
        "⚠️ **Disclaimer:** This is a student project for educational purposes only. "
        "The predictions and suggestions are based on a simple ML model and should NOT "
        "be used for actual financial decisions. Always consult a certified financial "
        "advisor before making investment decisions."
    )

else:
    # ── Landing state (before user clicks "Run Analysis") ───────────────────
    st.markdown("")
    st.markdown("")

    # Show a welcoming landing section
    landing_cols = st.columns([1, 2, 1])
    with landing_cols[1]:
        st.markdown(
            """
            <div style="text-align: center; padding: 60px 20px;">
                <div style="font-size: 4rem; margin-bottom: 16px;">📊</div>
                <h3 style="color: #FAFAFA; margin-bottom: 12px;">Welcome to the Stock Analysis Dashboard</h3>
                <p style="color: #888; font-size: 1rem; max-width: 500px; margin: 0 auto 30px auto;">
                    Select a stock (or index/commodity) in the sidebar, choose your dates,
                    and click <strong style="color: #6C63FF;">Run Analysis</strong> to get started. You can also click below to begin with the defaults.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="landing-btn-container">', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            btn1 = st.button("📈\nPrice Charts", use_container_width=True)
        with col2:
            btn2 = st.button("📊\nIndicators", use_container_width=True)
        with col3:
            btn3 = st.button("🤖\nML Prediction", use_container_width=True)
        with col4:
            btn4 = st.button("🔮\nForecast", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if btn1 or btn2 or btn3 or btn4:
            st.session_state.run_dashboard = True
            st.rerun()

    # Popular tickers guide
    st.markdown("---")
    st.markdown("#### 🌍 Popular Tickers to Try")
    guide_cols = st.columns(4)
    guides = [
        ("🇮🇳 Indian (NSE)", ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"]),
        ("🇺🇸 US (NYSE/NASDAQ)", ["AAPL", "MSFT", "GOOGL", "AMZN"]),
        ("🇪🇺 European", ["ASML.AS", "SAP.DE", "NESN.SW"]),
        ("🪙 Crypto (via Yahoo)", ["BTC-USD", "ETH-USD", "SOL-USD"]),
    ]
    for col, (region, tickers) in zip(guide_cols, guides):
        with col:
            st.markdown(f"**{region}**")
            for t in tickers:
                st.code(t, language=None)
