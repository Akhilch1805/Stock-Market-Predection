"""
utils.py
--------
Utility functions for chart creation and formatting.

This module provides:
  - Plotly chart builders for stock prices, indicators, and predictions
  - Number formatting helpers
  - Color scheme constants for consistent styling
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Color Palette (consistent across all charts)
# ---------------------------------------------------------------------------

COLORS = {
    "primary": "#6C63FF",       # Purple accent
    "secondary": "#FF6584",     # Pink accent
    "success": "#00C9A7",       # Green
    "warning": "#FFD93D",       # Yellow
    "danger": "#FF6B6B",        # Red
    "info": "#4ECDC4",          # Teal
    "bg": "#0E1117",            # Dark background
    "card_bg": "#1A1D29",       # Card background
    "text": "#FAFAFA",          # Light text
    "grid": "#2A2D3A",          # Grid lines
    "open": "#FFD93D",          # Open price
    "high": "#00C9A7",          # High price
    "low": "#FF6B6B",           # Low price
    "close": "#6C63FF",         # Close price
    "volume": "#4ECDC4",        # Volume bars
    "sma_10": "#FF6584",        # SMA 10
    "sma_20": "#FFD93D",        # SMA 20
    "rsi": "#6C63FF",           # RSI line
    "macd": "#00C9A7",          # MACD line
    "macd_signal": "#FF6584",   # MACD Signal
    "forecast": "#FFD93D",      # Forecast line
}

CURRENCY_SYMBOLS = {
    "INR": "₹",
    "USD": "$",
    "EUR": "€",
    "GBP": "£",
    "JPY": "¥"
}

# ---------------------------------------------------------------------------
# Chart Layout Template
# ---------------------------------------------------------------------------

def _base_layout(title: str, yaxis_title: str = "", height: int = 500) -> dict:
    """Return a consistent Plotly layout dictionary."""
    return dict(
        title=dict(text=title, font=dict(size=18, color=COLORS["text"])),
        plot_bgcolor=COLORS["bg"],
        paper_bgcolor=COLORS["bg"],
        font=dict(color=COLORS["text"], family="Inter, sans-serif"),
        xaxis=dict(
            gridcolor=COLORS["grid"],
            showgrid=True,
            zeroline=False,
        ),
        yaxis=dict(
            title=yaxis_title,
            gridcolor=COLORS["grid"],
            showgrid=True,
            zeroline=False,
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=COLORS["text"]),
        ),
        height=height,
        margin=dict(l=40, r=20, t=50, b=40),
        hovermode="x unified",
    )


# ---------------------------------------------------------------------------
# Candlestick Chart
# ---------------------------------------------------------------------------

def plot_candlestick(df: pd.DataFrame, title: str = "Stock Price") -> go.Figure:
    """
    Create an interactive candlestick chart with volume bars.

    Parameters
    ----------
    df : pd.DataFrame
        Stock data with Open, High, Low, Close, Volume columns.
    title : str
        Chart title.

    Returns
    -------
    go.Figure
        Plotly figure object.
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.75, 0.25],
        subplot_titles=("Price", "Volume"),
    )

    # Candlestick trace
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="OHLC",
            increasing_line_color=COLORS["success"],
            decreasing_line_color=COLORS["danger"],
        ),
        row=1, col=1,
    )

    # Volume bar trace
    # Color volume bars green/red based on close vs open
    volume_colors = [
        COLORS["success"] if row["Close"] >= row["Open"] else COLORS["danger"]
        for _, row in df.iterrows()
    ]

    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["Volume"],
            name="Volume",
            marker_color=volume_colors,
            opacity=0.6,
        ),
        row=2, col=1,
    )

    layout = _base_layout(title, height=600)
    layout["xaxis_rangeslider_visible"] = False
    fig.update_layout(**layout)
    fig.update_yaxes(title_text="Price", row=1, col=1, gridcolor=COLORS["grid"])
    fig.update_yaxes(title_text="Volume", row=2, col=1, gridcolor=COLORS["grid"])

    return fig


# ---------------------------------------------------------------------------
# OHLC Line Chart
# ---------------------------------------------------------------------------

def plot_ohlc_lines(df: pd.DataFrame) -> go.Figure:
    """
    Create individual line traces for Open, High, Low, Close prices.

    Parameters
    ----------
    df : pd.DataFrame
        Stock data.

    Returns
    -------
    go.Figure
    """
    fig = go.Figure()

    for col, color, dash in [
        ("Open", COLORS["open"], "dot"),
        ("High", COLORS["high"], "dash"),
        ("Low", COLORS["low"], "dash"),
        ("Close", COLORS["close"], "solid"),
    ]:
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col],
                name=col, mode="lines",
                line=dict(color=color, width=2, dash=dash),
            ))

    fig.update_layout(**_base_layout("Open / High / Low / Close", "Price"))
    return fig


# ---------------------------------------------------------------------------
# Moving Average Chart
# ---------------------------------------------------------------------------

def plot_moving_averages(df: pd.DataFrame) -> go.Figure:
    """
    Plot closing price with SMA-10 and SMA-20 overlays.

    Parameters
    ----------
    df : pd.DataFrame
        Stock data with SMA columns.

    Returns
    -------
    go.Figure
    """
    fig = go.Figure()

    # Close price
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"],
        name="Close", mode="lines",
        line=dict(color=COLORS["close"], width=2),
    ))

    # SMA-10
    if "SMA_10" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["SMA_10"],
            name="SMA 10", mode="lines",
            line=dict(color=COLORS["sma_10"], width=1.5, dash="dash"),
        ))

    # SMA-20
    if "SMA_20" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["SMA_20"],
            name="SMA 20", mode="lines",
            line=dict(color=COLORS["sma_20"], width=1.5, dash="dot"),
        ))

    fig.update_layout(**_base_layout("Moving Averages (SMA 10 & SMA 20)", "Price"))
    return fig


# ---------------------------------------------------------------------------
# RSI Chart
# ---------------------------------------------------------------------------

def plot_rsi(df: pd.DataFrame) -> go.Figure:
    """
    Plot RSI with overbought (70) and oversold (30) reference lines.

    Parameters
    ----------
    df : pd.DataFrame
        Stock data with 'RSI' column.

    Returns
    -------
    go.Figure
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index, y=df["RSI"],
        name="RSI", mode="lines",
        line=dict(color=COLORS["rsi"], width=2),
    ))

    # Overbought line (70)
    fig.add_hline(y=70, line_dash="dash", line_color=COLORS["danger"],
                  annotation_text="Overbought (70)", annotation_position="top left")

    # Oversold line (30)
    fig.add_hline(y=30, line_dash="dash", line_color=COLORS["success"],
                  annotation_text="Oversold (30)", annotation_position="bottom left")

    # Shade overbought & oversold zones
    fig.add_hrect(y0=70, y1=100, fillcolor=COLORS["danger"], opacity=0.08, line_width=0)
    fig.add_hrect(y0=0, y1=30, fillcolor=COLORS["success"], opacity=0.08, line_width=0)

    layout = _base_layout("Relative Strength Index (RSI)", "RSI")
    layout["yaxis"]["range"] = [0, 100]
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# MACD Chart
# ---------------------------------------------------------------------------

def plot_macd(df: pd.DataFrame) -> go.Figure:
    """
    Plot MACD line, Signal line, and Histogram.

    Parameters
    ----------
    df : pd.DataFrame
        Stock data with MACD columns.

    Returns
    -------
    go.Figure
    """
    fig = go.Figure()

    # MACD Line
    fig.add_trace(go.Scatter(
        x=df.index, y=df["MACD"],
        name="MACD", mode="lines",
        line=dict(color=COLORS["macd"], width=2),
    ))

    # Signal Line
    fig.add_trace(go.Scatter(
        x=df.index, y=df["MACD_Signal"],
        name="Signal", mode="lines",
        line=dict(color=COLORS["macd_signal"], width=2, dash="dash"),
    ))

    # Histogram (bar chart showing MACD - Signal)
    histogram_colors = [
        COLORS["success"] if val >= 0 else COLORS["danger"]
        for val in df["MACD_Histogram"]
    ]
    fig.add_trace(go.Bar(
        x=df.index, y=df["MACD_Histogram"],
        name="Histogram",
        marker_color=histogram_colors,
        opacity=0.5,
    ))

    fig.update_layout(**_base_layout("MACD (Moving Average Convergence Divergence)", "Value"))
    return fig


# ---------------------------------------------------------------------------
# Prediction Results Chart
# ---------------------------------------------------------------------------

def plot_predictions(y_test, y_pred, dates=None) -> go.Figure:
    """
    Plot actual vs predicted prices on the test set.

    Parameters
    ----------
    y_test : array-like
        Actual closing prices.
    y_pred : array-like
        Predicted closing prices.
    dates : array-like, optional
        Date index for x-axis.

    Returns
    -------
    go.Figure
    """
    fig = go.Figure()

    x_axis = dates if dates is not None else list(range(len(y_test)))

    fig.add_trace(go.Scatter(
        x=x_axis, y=y_test,
        name="Actual Price", mode="lines",
        line=dict(color=COLORS["close"], width=2),
    ))

    fig.add_trace(go.Scatter(
        x=x_axis, y=y_pred,
        name="Predicted Price", mode="lines",
        line=dict(color=COLORS["secondary"], width=2, dash="dash"),
    ))

    fig.update_layout(**_base_layout("Actual vs Predicted Price", "Price"))
    return fig


# ---------------------------------------------------------------------------
# Forecast Chart
# ---------------------------------------------------------------------------

def plot_forecast(df: pd.DataFrame, forecast_df: pd.DataFrame, days_label: str = "7-Day") -> go.Figure:
    """
    Plot historical closing prices and future forecast together.

    Parameters
    ----------
    df : pd.DataFrame
        Historical stock data with 'Close' column.
    forecast_df : pd.DataFrame
        Forecast data with 'Predicted_Close' column.
    days_label : str
        Label for the forecast (e.g., '7-Day', '30-Day').

    Returns
    -------
    go.Figure
    """
    fig = go.Figure()

    # Show only the last 60 trading days for context
    recent = df.tail(60)

    # Historical prices
    fig.add_trace(go.Scatter(
        x=recent.index, y=recent["Close"],
        name="Historical Close", mode="lines",
        line=dict(color=COLORS["close"], width=2),
    ))

    # Forecast prices
    fig.add_trace(go.Scatter(
        x=forecast_df.index, y=forecast_df["Predicted_Close"],
        name=f"{days_label} Forecast", mode="lines+markers",
        line=dict(color=COLORS["forecast"], width=2.5, dash="dot"),
        marker=dict(size=7, color=COLORS["forecast"]),
    ))

    # Vertical line showing where the forecast starts
    # Use add_shape + add_annotation to avoid Plotly's shapeannotation bug
    last_date = recent.index[-1]
    fig.add_shape(
        type="line",
        x0=last_date, x1=last_date,
        y0=0, y1=1,
        yref="paper",
        line=dict(color=COLORS["text"], width=1, dash="dash"),
        opacity=0.4,
    )
    fig.add_annotation(
        x=last_date, y=1, yref="paper",
        text="Forecast Start",
        showarrow=False,
        font=dict(color=COLORS["text"], size=11),
        yshift=10,
    )

    fig.update_layout(**_base_layout(f"{days_label} Price Forecast", "Price"))
    return fig


# ---------------------------------------------------------------------------
# Feature Importance Chart
# ---------------------------------------------------------------------------

def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """
    Horizontal bar chart of top-N most important features.

    Parameters
    ----------
    importance_df : pd.DataFrame
        DataFrame with 'Feature' and 'Importance' columns.
    top_n : int
        Number of top features to display.

    Returns
    -------
    go.Figure
    """
    top = importance_df.head(top_n).sort_values("Importance")

    fig = go.Figure(go.Bar(
        x=top["Importance"],
        y=top["Feature"],
        orientation="h",
        marker_color=COLORS["primary"],
        marker_line_color=COLORS["primary"],
        opacity=0.85,
    ))

    fig.update_layout(**_base_layout(f"Top {top_n} Feature Importances", "Importance"))
    fig.update_layout(yaxis=dict(title="", gridcolor=COLORS["grid"]))
    return fig


# ---------------------------------------------------------------------------
# Sparkline (for Dashboard Cards)
# ---------------------------------------------------------------------------

def _hex_to_rgba(hex_color: str, alpha: float = 0.1) -> str:
    """Convert a hex color string to rgba format with given alpha."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 6:
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"
    return f"rgba(108,99,255,{alpha})"  # fallback


def plot_mini_sparkline(data: pd.Series, color: str = COLORS["primary"]) -> go.Figure:
    """
    Create a minimal sparkline for display in small cards.
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data.values,
        mode="lines",
        line=dict(color=color, width=2),
        fill='tozeroy',
        fillcolor=_hex_to_rgba(color, 0.1),
        hoverinfo='none'
    ))
    
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=40,
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        hovermode=False
    )
    
    return fig


# ---------------------------------------------------------------------------
# Formatting Helpers
# ---------------------------------------------------------------------------

def format_large_number(num, symbol="") -> str:
    """
    Format large numbers into human-readable strings.
    Examples: 1,500,000 → '1.50M', 2,300,000,000 → '2.30B'
    """
    if num is None or num == "N/A":
        return "N/A"
    try:
        num = float(num)
    except (ValueError, TypeError):
        return str(num)

    # Use Indian numbering system if symbol is ₹
    is_inr = symbol == "₹"

    if abs(num) >= 1e12:
        return f"{symbol}{num / 1e12:.2f}T"
    elif abs(num) >= 1e9:
        return f"{symbol}{num / 1e9:.2f}B"
    elif abs(num) >= 1e7 and is_inr:
        return f"{symbol}{num / 1e7:.2f}Cr"
    elif abs(num) >= 1e6:
        return f"{symbol}{num / 1e6:.2f}M"
    elif abs(num) >= 1e5 and is_inr:
        return f"{symbol}{num / 1e5:.2f}L"
    elif abs(num) >= 1e3:
        return f"{symbol}{num / 1e3:.2f}K"
    else:
        return f"{symbol}{num:.2f}"
