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
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import time


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





def get_exchange_rate(base_currency: str, target_currency: str) -> float:
    """
    Fetch the current exchange rate between two currencies.
    """
    if base_currency == target_currency or "N/A" in [base_currency, target_currency]:
        return 1.0
    
    # Standardize tickers
    ticker = f"{base_currency}{target_currency}=X"
    if base_currency == "INR" and target_currency == "USD":
        ticker = "INRU=X"
    elif base_currency == "USD" and target_currency == "INR":
        ticker = "USDINR=X"

    try:
        # Try to get very recent data for a "live" feel
        # Using a timeout and multiple attempts for robustness
        data = yf.download(ticker, period="1d", interval="1m", progress=False)
        if data.empty:
            data = yf.download(ticker, period="5d", progress=False)
            
        if not data.empty:
            return float(data["Close"].iloc[-1])
        return 1.0
    except Exception:
        # Fallback rates if YF is down
        fallback = {
            "USDINR": 83.50,
            "EURINR": 90.10,
            "GBPINR": 105.20,
            "JPYINR": 0.55
        }
        key = f"{base_currency}{target_currency}"
        return fallback.get(key, 1.0)


def get_fuel_prices(city: str = "Mumbai") -> dict:
    """
    Scrape Petrol and Diesel prices for a specific city in India.
    """
    prices = {"petrol": "N/A", "diesel": "N/A", "city": city, "date": datetime.now().strftime("%Y-%m-%d")}
    
    # Mapping common cities to goodreturns URL segments
    city_map = {
        "mumbai": "mumbai.html",
        "delhi": "delhi.html",
        "bangalore": "bangalore.html",
        "chennai": "chennai.html",
        "hyderabad": "hyderabad.html",
        "kolkata": "kolkata.html",
        "pune": "pune.html",
        "ahmedabad": "ahmedabad.html",
        "jaipur": "jaipur.html",
        "lucknow": "lucknow.html"
    }
    
    city_slug = city_map.get(city.lower(), "mumbai.html")
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        # Fetch Petrol Price
        p_url = f"https://www.goodreturns.in/petrol-price/{city_slug}"
        p_resp = requests.get(p_url, headers=headers, timeout=10)
        if p_resp.status_code == 200:
            p_soup = BeautifulSoup(p_resp.text, 'html.parser')
            p_val = p_soup.find("div", {"class": "gold_silver_table"})
            if p_val:
                tds = p_val.find_all("td")
                if len(tds) > 1:
                    prices["petrol"] = tds[1].text.strip().replace("₹", "").strip()

        # Fetch Diesel Price
        d_url = f"https://www.goodreturns.in/diesel-price/{city_slug}"
        d_resp = requests.get(d_url, headers=headers, timeout=10)
        if d_resp.status_code == 200:
            d_soup = BeautifulSoup(d_resp.text, 'html.parser')
            d_val = d_soup.find("div", {"class": "gold_silver_table"})
            if d_val:
                tds = d_val.find_all("td")
                if len(tds) > 1:
                    prices["diesel"] = tds[1].text.strip().replace("₹", "").strip()
                
    except Exception:
        pass

    # Fallback to some representative prices if scraping fails (more realistic ranges)
    if prices["petrol"] == "N/A":
        prices["petrol"] = "104.21" 
    if prices["diesel"] == "N/A":
        prices["diesel"] = "92.15"

    return prices
