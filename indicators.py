"""
indicators.py
-------------
Calculates common technical indicators used in stock market analysis.

Technical indicators help traders and analysts understand the momentum,
trend, and volatility of a stock. This module implements:
  - Simple Moving Average (SMA)
  - Relative Strength Index (RSI)
  - Moving Average Convergence Divergence (MACD)
"""

import pandas as pd
import numpy as np


def add_moving_averages(df: pd.DataFrame, short_window: int = 10, long_window: int = 20) -> pd.DataFrame:
    """
    Add Simple Moving Average (SMA) columns to the DataFrame.

    A Moving Average smooths out price data by creating a constantly
    updated average price over a specific number of periods.

    Parameters
    ----------
    df : pd.DataFrame
        Stock data with a 'Close' column.
    short_window : int
        Period for the short-term moving average (default: 10 days).
    long_window : int
        Period for the long-term moving average (default: 20 days).

    Returns
    -------
    pd.DataFrame
        Original DataFrame with added SMA columns.
    """
    df = df.copy()
    df[f"SMA_{short_window}"] = df["Close"].rolling(window=short_window).mean()
    df[f"SMA_{long_window}"] = df["Close"].rolling(window=long_window).mean()
    return df


def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calculate the Relative Strength Index (RSI).

    RSI measures the speed and magnitude of recent price changes.
    - RSI > 70  → Stock may be overbought (potential sell signal)
    - RSI < 30  → Stock may be oversold (potential buy signal)

    Formula:
        RSI = 100 - (100 / (1 + RS))
        RS  = Average Gain / Average Loss  (over 'period' days)

    Parameters
    ----------
    df : pd.DataFrame
        Stock data with a 'Close' column.
    period : int
        Look-back period for RSI calculation (default: 14 days).

    Returns
    -------
    pd.DataFrame
        Original DataFrame with an added 'RSI' column.
    """
    df = df.copy()

    # Calculate daily price changes
    delta = df["Close"].diff()

    # Separate gains and losses
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    # Calculate the exponential moving average of gains and losses
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    # Calculate Relative Strength (RS) and RSI
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    return df


def calculate_macd(
    df: pd.DataFrame,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> pd.DataFrame:
    """
    Calculate Moving Average Convergence Divergence (MACD).

    MACD shows the relationship between two EMAs of a stock's price.
    - MACD Line   = EMA(fast) - EMA(slow)
    - Signal Line = EMA of MACD Line
    - Histogram   = MACD Line - Signal Line

    Buy signal:  MACD crosses above Signal line
    Sell signal: MACD crosses below Signal line

    Parameters
    ----------
    df : pd.DataFrame
        Stock data with a 'Close' column.
    fast_period : int
        Period for the fast EMA (default: 12).
    slow_period : int
        Period for the slow EMA (default: 26).
    signal_period : int
        Period for the signal line EMA (default: 9).

    Returns
    -------
    pd.DataFrame
        Original DataFrame with MACD, MACD_Signal, and MACD_Histogram columns.
    """
    df = df.copy()

    # Calculate fast and slow Exponential Moving Averages
    ema_fast = df["Close"].ewm(span=fast_period, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow_period, adjust=False).mean()

    # MACD Line = Fast EMA - Slow EMA
    df["MACD"] = ema_fast - ema_slow

    # Signal Line = EMA of MACD
    df["MACD_Signal"] = df["MACD"].ewm(span=signal_period, adjust=False).mean()

    # Histogram = MACD - Signal
    df["MACD_Histogram"] = df["MACD"] - df["MACD_Signal"]

    return df


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to add all technical indicators at once.

    Parameters
    ----------
    df : pd.DataFrame
        Raw stock data with at least 'Open', 'High', 'Low', 'Close', 'Volume'.

    Returns
    -------
    pd.DataFrame
        DataFrame enriched with SMA, RSI, and MACD indicators.
    """
    df = add_moving_averages(df)
    df = calculate_rsi(df)
    df = calculate_macd(df)
    return df
