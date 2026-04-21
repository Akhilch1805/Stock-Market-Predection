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
from data_loader import fetch_stock_data, get_stock_info, get_exchange_rate, get_fuel_prices
from indicators import add_all_indicators
from model import prepare_features, train_model, get_feature_importance, forecast_future, generate_suggestion
from streamlit_autorefresh import st_autorefresh
import time
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
    CURRENCY_SYMBOLS,
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
        background: rgba(26, 29, 41, 0.6);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 24px;
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .metric-card:hover {
        transform: translateY(-8px) scale(1.02);
        background: rgba(37, 40, 54, 0.8);
        box-shadow: 0 12px 40px rgba(108, 99, 255, 0.25);
        border-color: rgba(108, 99, 255, 0.4);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FAFAFA 0%, #A0A0A0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    .metric-label {
        font-size: 0.9rem;
        font-weight: 600;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1.5px;
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

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar — User Inputs
# ──────────────────────────────────────────────────────────────────────────────

if 'run_dashboard' not in st.session_state:
    st.session_state.run_dashboard = False

if 'current_city' not in st.session_state:
    st.session_state.current_city = "Mumbai"

with st.sidebar:
    st.markdown("### 🧭 Navigation")
    if st.button("🏠 Home Dashboard", type="secondary" if st.session_state.run_dashboard else "primary", use_container_width=True):
        st.session_state.run_dashboard = False
        st.rerun()

    st.markdown("---")
    st.markdown("### ⚙️ Global Settings")
    

    
    # Currency Selection (Global)
    target_currency = st.selectbox(
        "💱 Display Currency",
        options=["INR", "USD", "EUR", "GBP", "JPY"],
        index=0,  # Default to INR
        help="All prices across the app will be converted to this currency."
    )
    
    # City Selection for Fuel (Only relevant for Home)
    if not st.session_state.run_dashboard:
        st.session_state.current_city = st.selectbox(
            "📍 Fuel Price City",
            options=["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad", "Kolkata", "Pune", "Ahmedabad", "Jaipur", "Lucknow"],
            index=0
        )

    # Live refresh toggle
    live_refresh = st.toggle("🔄 Auto-refresh Home", value=True, help="Automatically refresh market data every 60 seconds.")

    st.markdown("---")
    st.markdown("### 🔍 Stock Analysis")

    # Load preset ticker suggestions
    try:
        with open("comprehensive_tickers.json", "r") as f:
            preset_tickers = json.load(f)
    except Exception:
        preset_tickers = {
            "NIFTY 50 (Index)": "^NSEI",
            "BSE SENSEX (Index)": "^BSESN",
            "Reliance Industries": "RELIANCE.NS",
            "TCS": "TCS.NS",
            "Gold (COMEX)": "GC=F",
            "Apple": "AAPL",
            "Crude Oil": "CL=F"
        }

    # Unified Search Bar
    selected_preset = st.selectbox(
        "🔎 Search for a Stock",
        options=["Enter custom ticker..."] + list(preset_tickers.keys()),
        index=1,
        help="Type a company name to instantly search from 2000+ stocks"
    )

    # Ticker input resolution
    if selected_preset == "Enter custom ticker...":
        ticker_input = st.text_input(
            "📌 Custom Yahoo Finance Ticker Symbol",
            value="RELIANCE.NS",
            help="E.g., TCS.NS for NSE, AAPL for US"
        )
    else:
        ticker_input = preset_tickers[selected_preset]

    # Date range selection
    st.markdown("**📅 Date Range**")
    col_start, col_end = st.columns(2)
    with col_start:
        start_date = st.date_input("Start", value=datetime.now() - timedelta(days=365 * 2))
    with col_end:
        end_date = st.date_input("End", value=datetime.now())

    # Forecast configuration
    forecast_days = st.selectbox("🔮 Forecast days", options=[7, 14, 30], index=0)

    # Run button
    if st.button("🚀 Run Deep Analysis", type="primary", use_container_width=True):
        st.session_state.run_dashboard = True

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#666; font-size:0.75rem;'>"
        "Built with ❤️ using Python, Streamlit & ML<br>"
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
    with st.spinner(f"📡 Fetching data for {ticker}..."):
        try:
            df = fetch_stock_data(ticker, str(start_date), str(end_date))
            stock_info = get_stock_info(ticker)
        except ValueError as e:
            st.error(f"❌ {str(e)}")
            st.stop()

    # ── Currency Conversion ────────────────────────────────────────────────
    base_currency = stock_info.get("currency", "USD")
    conv_rate = get_exchange_rate(base_currency, target_currency)
    curr_sym = CURRENCY_SYMBOLS.get(target_currency, "")

    if conv_rate != 1.0:
        df["Open"] *= conv_rate
        df["High"] *= conv_rate
        df["Low"] *= conv_rate
        df["Close"] *= conv_rate

    # ── Stock Info Header ──────────────────────────────────────────────────
    col_title, col_home = st.columns([4, 1])
    with col_title:
        st.markdown(f"### 🏢 {stock_info.get('name', ticker)}")
    with col_home:
        if st.button("⬅️ Back to Home", key="back_home_btn", use_container_width=True):
            st.session_state.run_dashboard = False
            st.rerun()

    info_cols = st.columns(5)
    info_items = [
        ("Sector", stock_info.get("sector", "N/A")),
        ("Industry", stock_info.get("industry", "N/A")),
        ("Currency", stock_info.get("currency", "N/A")),
        ("Market Cap", format_large_number(stock_info.get("market_cap", "N/A") * (conv_rate if stock_info.get("market_cap") != "N/A" else 1), curr_sym)),
        ("P/E Ratio", f"{stock_info.get('pe_ratio', 'N/A')}"),
    ]
    for col, (label, value) in zip(info_cols, info_items):
        with col:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value" style="font-size:1.4rem;">{value}</div>'
                f'<div class="metric-label">{label}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── Price Summary Metrics ───────────────────────────────────────────────
    valid_close = df[df["Close"].notna() & (df["Close"] > 0)]
    if len(valid_close) >= 2:
        latest = valid_close.iloc[-1]
        prev = valid_close.iloc[-2]
    else:
        latest = df.iloc[-1]
        prev = latest

    close_val = float(latest["Close"])
    prev_val = float(prev["Close"])
    price_change = close_val - prev_val
    price_change_pct = (price_change / prev_val * 100) if prev_val != 0 else 0

    price_cols = st.columns(4)
    price_items = [
        ("Current Price", f"{curr_sym}{close_val:,.2f}", "", None),
        ("Day Change", f"{price_change:+,.2f}", f"({price_change_pct:+.2f}%)", price_change),
        ("Day High", f"{curr_sym}{latest['High']:,.2f}", "", None),
        ("Day Low", f"{curr_sym}{latest['Low']:,.2f}", "", None),
    ]
    for col, (label, value, extra, raw_val) in zip(price_cols, price_items):
        with col:
            if label == "Day Change":
                color = "#00C9A7" if raw_val >= 0 else "#FF6B6B"
            else:
                color = "#FAFAFA"
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value" style="color:{color}">{value} <span style="font-size:0.9rem;">{extra}</span></div>'
                f'<div class="metric-label">{label}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Step 2: Indicators & Charts ──────────────────────────────
    df_with_indicators = add_all_indicators(df)
    tab_price, tab_indicators, tab_data = st.tabs(["📈 Price Chartsbolt", "📊 technical Indicators", "📋 Raw Data"])

    with tab_price:
        st.plotly_chart(plot_candlestick(df, f"{ticker} — Price Movement"), use_container_width=True)
    with tab_indicators:
        st.plotly_chart(plot_moving_averages(df_with_indicators), use_container_width=True)
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(plot_rsi(df_with_indicators), use_container_width=True)
        with c2: st.plotly_chart(plot_macd(df_with_indicators), use_container_width=True)
    with tab_data:
        st.dataframe(df_with_indicators, use_container_width=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Step 3: ML Prediction ───────────────────────────────────────────
    st.markdown("## 🤖 Machine Learning Forecast")
    df_features = prepare_features(df_with_indicators)
    model_result = train_model(df_features)

    forecast_df = forecast_future(df_features, model_result, days=forecast_days)
    st.plotly_chart(plot_forecast(df, forecast_df, f"{forecast_days}-Day"), use_container_width=True)

    suggestion = generate_suggestion(df_with_indicators, forecast_df)
    
    st.markdown(f'<div class="suggestion-card"><div class="suggestion-title">{suggestion["suggestion"]}</div>'
                f'<div class="suggestion-confidence">Confidence: {suggestion["confidence"]:.0f}%</div>'
                + "".join(f'<div class="reason-item">• {reason}</div>' for reason in suggestion["reasons"]) + "</div>", unsafe_allow_html=True)

else:
    # ── HOME PAGE: LIVE MARKET OVERVIEW ───────────────────────────────
    
    if live_refresh:
        st_autorefresh(interval=60000, key="market_refresh")

    st.markdown('<h2 style="text-align:center; color:#6C63FF; margin-bottom:30px;">🌍 Live Market Overview</h2>', unsafe_allow_html=True)
    
    with st.spinner("📡 Syncing with global markets..."):
        fuel_data = get_fuel_prices(st.session_state.current_city)
        usdinr_rate = get_exchange_rate("USD", "INR")
        curr_sym = CURRENCY_SYMBOLS.get(target_currency, "₹")
        target_rate = get_exchange_rate("INR", target_currency) if target_currency != "INR" else 1.0

        # Markers layout
        m_col1, m_col2, m_col3 = st.columns(3)
        
        # 1. INDICES Section
        with m_col1:
            st.markdown("#### 📈 Major Indices")
            indices = {"NIFTY 50": "^NSEI", "SENSEX": "^BSESN"}
            for name, code in indices.items():
                try:
                    m_data = fetch_stock_data(code, (datetime.now() - timedelta(days=20)).strftime('%Y-%m-%d'), datetime.now().strftime('%Y-%m-%d'))
                    val = m_data["Close"].iloc[-1] * target_rate
                    delta = ((m_data["Close"].iloc[-1] - m_data["Close"].iloc[-2]) / m_data["Close"].iloc[-2]) * 100
                    color = "#00C9A7" if delta >= 0 else "#FF6B6B"
                    
                    st.markdown(f"""
                        <div class="metric-card" style="border-top: 4px solid {color}; padding: 15px; margin-bottom: 15px;">
                            <div class="metric-label" style="font-size:0.8rem;">{name}</div>
                            <div class="metric-value" style="font-size:1.5rem;">{curr_sym}{val:,.2f}</div>
                            <div style="color:{color}; font-weight:600; font-size:0.8rem;">
                                {'▲' if delta >= 0 else '▼'} {abs(delta):.2f}%
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    from utils import plot_mini_sparkline
                    st.plotly_chart(plot_mini_sparkline(m_data["Close"].tail(15), color), use_container_width=True, config={'displayModeBar': False})
                except: st.error(f"Failed: {name}")

        # 2. COMMODITIES Section
        with m_col2:
            st.markdown("#### 🪙 Commodities")
            commodities = {"GOLD (10g)": "GC=F", "SILVER (1kg)": "SI=F"}
            for name, code in commodities.items():
                try:
                    m_data = fetch_stock_data(code, (datetime.now() - timedelta(days=20)).strftime('%Y-%m-%d'), datetime.now().strftime('%Y-%m-%d'))
                    raw_val = m_data["Close"].iloc[-1]
                    raw_prev = m_data["Close"].iloc[-2]
                    
                    if "GOLD" in name:
                        val = (raw_val / 31.1035) * 10 * usdinr_rate * target_rate
                        p_val = (raw_prev / 31.1035) * 10 * usdinr_rate * target_rate
                    else:
                        val = (raw_val / 31.1035) * 1000 * usdinr_rate * target_rate
                        p_val = (raw_prev / 31.1035) * 1000 * usdinr_rate * target_rate

                    delta = ((val - p_val) / p_val) * 100
                    color = "#00C9A7" if delta >= 0 else "#FF6B6B"
                    
                    st.markdown(f"""
                        <div class="metric-card" style="border-top: 4px solid {color}; padding: 15px; margin-bottom: 15px;">
                            <div class="metric-label" style="font-size:0.8rem;">{name}</div>
                            <div class="metric-value" style="font-size:1.5rem;">{curr_sym}{val:,.2f}</div>
                            <div style="color:{color}; font-weight:600; font-size:0.8rem;">
                                {'▲' if delta >= 0 else '▼'} {abs(delta):.2f}%
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    st.plotly_chart(plot_mini_sparkline(m_data["Close"].tail(15), color), use_container_width=True, config={'displayModeBar': False})
                except: st.error(f"Failed: {name}")

        # 3. FUEL & FOREX Section
        with m_col3:
            st.markdown("#### ⛽ Fuel & Forex")
            # Petrol
            p_val = float(fuel_data["petrol"]) * target_rate
            st.markdown(f"""
                <div class="metric-card" style="border-top: 4px solid #6C63FF; padding: 15px; margin-bottom: 10px;">
                    <div class="metric-label" style="font-size:0.8rem;">PETROL ({st.session_state.current_city})</div>
                    <div class="metric-value" style="font-size:1.5rem;">{curr_sym}{p_val:.2f}</div>
                </div>
            """, unsafe_allow_html=True)
            # Diesel
            d_val = float(fuel_data["diesel"]) * target_rate
            st.markdown(f"""
                <div class="metric-card" style="border-top: 4px solid #FF6584; padding: 15px; margin-bottom: 10px;">
                    <div class="metric-label" style="font-size:0.8rem;">DIESEL ({st.session_state.current_city})</div>
                    <div class="metric-value" style="font-size:1.5rem;">{curr_sym}{d_val:.2f}</div>
                </div>
            """, unsafe_allow_html=True)
            
            # USD/INR Rate (For context)
            st.markdown(f"""
                <div class="metric-card" style="border-top: 4px solid #FFD93D; padding: 15px;">
                    <div class="metric-label" style="font-size:0.8rem;">USD / INR</div>
                    <div class="metric-value" style="font-size:1.5rem;">₹{usdinr_rate:.2f}</div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("""
        <div style="text-align: center; background: rgba(108, 99, 255, 0.05); padding: 30px; border-radius: 20px;">
            <h3 style="color: #6C63FF; margin-bottom: 10px;">Ready for Deep Analysis?</h3>
            <p style="color: #888; max-width: 600px; margin: 0 auto 20px auto;">
                Select a stock from the sidebar or enter any Yahoo Finance ticker to get started with 
                our AI-powered prediction model and technical analysis tools.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Info grid
    g1, g2, g3, g4 = st.columns(4)
    with g1: st.info("**NSE Stocks**  \nAdd `.NS` suffix  \nEx: `TCS.NS`", icon="🇮🇳")
    with g2: st.info("**BSE Stocks**  \nAdd `.BO` suffix  \nEx: `RELIANCE.BO`", icon="🏛️")
    with g3: st.info("**US Stocks**  \nDirect ticker  \nEx: `NVDA`", icon="🇺🇸")
    with g4: st.info("**Forex**  \nUse `=X` suffix  \nEx: `EURUSD=X`", icon="💱")
