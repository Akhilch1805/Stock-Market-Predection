"""
data_loader.py
--------------
Responsible for downloading stock market data from Yahoo Finance
using the yfinance library.

This module handles:
  - Fetching historical OHLCV data for a given ticker and date range
  - Basic validation of the downloaded data
  - Returning a clean pandas DataFrame
"""

import yfinance as yf
import pandas as pd
from datetime import datetime


def fetch_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download historical stock data from Yahoo Finance.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g., 'AAPL', 'RELIANCE.NS', 'TCS.NS').
    start_date : str
        Start date in 'YYYY-MM-DD' format.
    end_date : str
        End date in 'YYYY-MM-DD' format.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Open, High, Low, Close, Adj Close, Volume.

    Raises
    ------
    ValueError
        If the ticker is invalid or no data is available for the given range.
    """
    try:
        # Create a yfinance Ticker object
        stock = yf.Ticker(ticker)

        # Download historical data
        df = stock.history(start=start_date, end=end_date)

        # Check if data is empty (invalid ticker or no data for date range)
        if df.empty:
            raise ValueError(
                f"No data found for ticker '{ticker}' between {start_date} and {end_date}. "
                "Please check the ticker symbol and date range."
            )

        # Drop any rows with all NaN values
        df.dropna(how="all", inplace=True)

        # Keep only the columns we need
        columns_to_keep = ["Open", "High", "Low", "Close", "Volume"]
        available_cols = [col for col in columns_to_keep if col in df.columns]
        df = df[available_cols]

        # Remove rows where Close price is 0 or NaN (incomplete data)
        if "Close" in df.columns:
            df = df[df["Close"].notna() & (df["Close"] > 0)]

        return df

    except Exception as e:
        # Re-raise ValueError as-is; wrap other exceptions
        if isinstance(e, ValueError):
            raise
        raise ValueError(
            f"Error fetching data for '{ticker}': {str(e)}. "
            "Please verify the ticker symbol is correct."
        )


def get_stock_info(ticker: str) -> dict:
    """
    Fetch basic information about a stock (name, sector, etc.).

    Parameters
    ----------
    ticker : str
        Stock ticker symbol.

    Returns
    -------
    dict
        Dictionary containing stock information.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "name": info.get("longName", ticker),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "currency": info.get("currency", "N/A"),
            "exchange": info.get("exchange", "N/A"),
            "market_cap": info.get("marketCap", "N/A"),
            "pe_ratio": info.get("trailingPE", "N/A"),
            "52w_high": info.get("fiftyTwoWeekHigh", "N/A"),
            "52w_low": info.get("fiftyTwoWeekLow", "N/A"),
        }
    except Exception:
        # Return minimal info if fetching fails
        return {"name": ticker, "sector": "N/A", "industry": "N/A"}
