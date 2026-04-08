"""
model.py
--------
Machine Learning module for stock price prediction.

This module handles:
  - Feature engineering (creating ML-ready features from stock data)
  - Training a Random Forest Regressor to predict the next-day closing price
  - Evaluating model performance with MAE, RMSE, and R² metrics
  - Generating future price forecasts (7-day and 30-day)
  - Producing Buy / Sell / Hold suggestions
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features for the ML model from the raw stock + indicator data.

    Features created:
      - Lagged closing prices (1, 2, 3, 5, 10 days ago)
      - Daily return (percentage change)
      - Rolling volatility (10-day std of returns)
      - Technical indicator values (SMA, RSI, MACD) if present
      - Day of week and month as cyclical features

    The target variable is 'Target' = next day's closing price.

    Parameters
    ----------
    df : pd.DataFrame
        Stock data with indicators already added.

    Returns
    -------
    pd.DataFrame
        DataFrame with feature columns and 'Target' column, NaNs dropped.
    """
    df = df.copy()

    # --- Price-based features ---
    # Lag features: what was the closing price N days ago?
    for lag in [1, 2, 3, 5, 10]:
        df[f"Close_Lag_{lag}"] = df["Close"].shift(lag)

    # Daily return (percentage change from previous day)
    df["Daily_Return"] = df["Close"].pct_change()

    # Rolling volatility: standard deviation of returns over 10 days
    df["Volatility_10"] = df["Daily_Return"].rolling(window=10).std()

    # Price momentum: difference between current close and close 5 days ago
    df["Momentum_5"] = df["Close"] - df["Close"].shift(5)

    # High-Low spread (intraday range)
    if "High" in df.columns and "Low" in df.columns:
        df["HL_Spread"] = df["High"] - df["Low"]

    # Volume change
    if "Volume" in df.columns:
        df["Volume_Change"] = df["Volume"].pct_change()

    # --- Calendar features ---
    df["Day_of_Week"] = df.index.dayofweek       # 0=Mon, 4=Fri
    df["Month"] = df.index.month

    # --- Target variable ---
    # We want to predict *tomorrow's* closing price
    df["Target"] = df["Close"].shift(-1)

    # Replace any infinity values with NaN, then drop all NaN rows
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Return the list of feature columns to be used by the ML model.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with features already prepared.

    Returns
    -------
    list
        Column names to use as features (everything except 'Target').
    """
    # Exclude the target and any non-numeric columns
    exclude = ["Target"]
    feature_cols = [
        col for col in df.select_dtypes(include=[np.number]).columns
        if col not in exclude
    ]
    return feature_cols


# ---------------------------------------------------------------------------
# Model Training & Evaluation
# ---------------------------------------------------------------------------

def train_model(df: pd.DataFrame, test_size: float = 0.2) -> dict:
    """
    Train a Random Forest Regressor to predict next-day closing price.

    Why Random Forest?
      - Handles non-linear relationships well
      - Robust to outliers and noisy data
      - Does not require feature scaling
      - Good balance of accuracy and interpretability

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with features and 'Target' column (from prepare_features).
    test_size : float
        Fraction of data to reserve for testing (default: 20%).

    Returns
    -------
    dict
        Contains: model, feature_columns, X_train, X_test, y_train, y_test,
        y_pred, and evaluation metrics.
    """
    feature_cols = get_feature_columns(df)
    X = df[feature_cols].values
    y = df["Target"].values

    # Split data into training and testing sets
    # We use shuffle=False to preserve time ordering (important for time series!)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    # Initialize and train the Random Forest Regressor
    model = RandomForestRegressor(
        n_estimators=200,       # Number of trees in the forest
        max_depth=15,           # Maximum depth of each tree
        min_samples_split=5,    # Minimum samples to split a node
        min_samples_leaf=2,     # Minimum samples at a leaf node
        random_state=42,        # For reproducibility
        n_jobs=-1,              # Use all CPU cores
    )
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return {
        "model": model,
        "feature_columns": feature_cols,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }


# ---------------------------------------------------------------------------
# Feature Importance
# ---------------------------------------------------------------------------

def get_feature_importance(model_result: dict) -> pd.DataFrame:
    """
    Extract and sort feature importances from the trained model.

    Parameters
    ----------
    model_result : dict
        Output from train_model().

    Returns
    -------
    pd.DataFrame
        Sorted DataFrame with 'Feature' and 'Importance' columns.
    """
    importances = model_result["model"].feature_importances_
    features = model_result["feature_columns"]

    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": importances,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    return importance_df


# ---------------------------------------------------------------------------
# Future Forecast
# ---------------------------------------------------------------------------

def forecast_future(df: pd.DataFrame, model_result: dict, days: int = 7) -> pd.DataFrame:
    """
    Generate future price predictions for the specified number of days.

    This works by iteratively predicting one day at a time and using
    each prediction as input for the next day's prediction.

    Parameters
    ----------
    df : pd.DataFrame
        Full feature DataFrame (from prepare_features).
    model_result : dict
        Output from train_model().
    days : int
        Number of future days to predict (default: 7).

    Returns
    -------
    pd.DataFrame
        DataFrame with 'Date' and 'Predicted_Close' columns.
    """
    model = model_result["model"]
    feature_cols = model_result["feature_columns"]

    # Start from the last available row
    last_row = df.iloc[-1:].copy()
    predictions = []
    current_date = df.index[-1]

    for i in range(days):
        # Predict the next day's closing price
        features = last_row[feature_cols].values
        pred = model.predict(features)[0]

        # Move to the next business day
        current_date = current_date + pd.tseries.offsets.BDay(1)
        predictions.append({"Date": current_date, "Predicted_Close": pred})

        # Update the row for the next iteration:
        # Shift lag features and update with the predicted price
        last_row = last_row.copy()

        # Update lag features
        for lag in [10, 5, 3, 2]:
            col_from = f"Close_Lag_{lag - 1}" if lag > 1 else "Close"
            col_to = f"Close_Lag_{lag}"
            if col_to in last_row.columns and col_from in last_row.columns:
                last_row[col_to] = last_row[col_from].values[0]

        if "Close_Lag_1" in last_row.columns:
            last_row["Close_Lag_1"] = last_row["Close"].values[0]

        # Update Close price to the prediction
        last_row["Close"] = pred

        # Update derived features
        if "Daily_Return" in last_row.columns and "Close_Lag_1" in last_row.columns:
            prev_close = last_row["Close_Lag_1"].values[0]
            if prev_close != 0:
                last_row["Daily_Return"] = (pred - prev_close) / prev_close

        if "Momentum_5" in last_row.columns and "Close_Lag_5" in last_row.columns:
            last_row["Momentum_5"] = pred - last_row["Close_Lag_5"].values[0]

    forecast_df = pd.DataFrame(predictions)
    forecast_df.set_index("Date", inplace=True)
    return forecast_df


# ---------------------------------------------------------------------------
# Buy / Sell / Hold Suggestion
# ---------------------------------------------------------------------------

def generate_suggestion(df: pd.DataFrame, forecast_df: pd.DataFrame) -> dict:
    """
    Generate a Buy / Sell / Hold suggestion based on multiple signals.

    Decision logic combines:
      1. RSI signal     (oversold → Buy, overbought → Sell)
      2. MACD signal    (MACD > Signal → Buy, MACD < Signal → Sell)
      3. SMA crossover  (Price > SMA_20 → Buy, Price < SMA_20 → Sell)
      4. Forecast trend (predicted price going up → Buy, down → Sell)

    Each signal votes, and the majority vote determines the final suggestion.

    Parameters
    ----------
    df : pd.DataFrame
        Stock data with technical indicators.
    forecast_df : pd.DataFrame
        Future predictions from forecast_future().

    Returns
    -------
    dict
        Contains 'suggestion' (Buy/Sell/Hold), 'confidence', and 'reasons'.
    """
    signals = []    # +1 for Buy, -1 for Sell, 0 for Hold
    reasons = []

    last = df.iloc[-1]

    # 1. RSI Signal
    if "RSI" in df.columns and not pd.isna(last.get("RSI")):
        rsi = last["RSI"]
        if rsi < 30:
            signals.append(1)
            reasons.append(f"RSI = {rsi:.1f} (oversold → bullish signal)")
        elif rsi > 70:
            signals.append(-1)
            reasons.append(f"RSI = {rsi:.1f} (overbought → bearish signal)")
        else:
            signals.append(0)
            reasons.append(f"RSI = {rsi:.1f} (neutral zone)")

    # 2. MACD Signal
    if "MACD" in df.columns and "MACD_Signal" in df.columns:
        macd_val = last.get("MACD", 0)
        macd_sig = last.get("MACD_Signal", 0)
        if not pd.isna(macd_val) and not pd.isna(macd_sig):
            if macd_val > macd_sig:
                signals.append(1)
                reasons.append("MACD is above Signal line (bullish crossover)")
            else:
                signals.append(-1)
                reasons.append("MACD is below Signal line (bearish crossover)")

    # 3. SMA Crossover
    if "SMA_20" in df.columns:
        sma_20 = last.get("SMA_20")
        # Use last valid close price for comparison
        valid_for_sma = df[df["Close"].notna() & (df["Close"] > 0)]
        close = valid_for_sma.iloc[-1]["Close"] if len(valid_for_sma) > 0 else last["Close"]
        if not pd.isna(sma_20) and pd.notna(close) and close > 0:
            if close > sma_20:
                signals.append(1)
                reasons.append(f"Price ({close:.2f}) is above SMA-20 ({sma_20:.2f})")
            else:
                signals.append(-1)
                reasons.append(f"Price ({close:.2f}) is below SMA-20 ({sma_20:.2f})")

    # 4. Forecast Trend
    if not forecast_df.empty:
        # Find the last valid close price (skip NaN / 0 at end of data)
        valid_closes = df[df["Close"].notna() & (df["Close"] > 0)]["Close"]
        current_price = valid_closes.iloc[-1] if len(valid_closes) > 0 else 0
        future_price = forecast_df["Predicted_Close"].iloc[-1]

        if current_price > 0:
            change_pct = ((future_price - current_price) / current_price) * 100
            if change_pct > 1.0:
                signals.append(1)
                reasons.append(f"Forecast shows +{change_pct:.1f}% increase (bullish)")
            elif change_pct < -1.0:
                signals.append(-1)
                reasons.append(f"Forecast shows {change_pct:.1f}% decrease (bearish)")
            else:
                signals.append(0)
                reasons.append(f"Forecast shows {change_pct:.1f}% change (sideways)")

    # Aggregate the signals
    if not signals:
        return {"suggestion": "Hold", "confidence": 0, "reasons": ["Insufficient data"]}

    avg_signal = np.mean(signals)
    confidence = abs(avg_signal) * 100

    if avg_signal > 0.25:
        suggestion = "🟢 BUY"
    elif avg_signal < -0.25:
        suggestion = "🔴 SELL"
    else:
        suggestion = "🟡 HOLD"

    return {
        "suggestion": suggestion,
        "confidence": min(confidence, 100),
        "reasons": reasons,
    }
