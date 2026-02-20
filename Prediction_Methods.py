import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ======================== Rolling Linear Regression =========================
# This function predicts the next point in a series using simple linear regression
# over a rolling window. It fits a straight line to the last `window` points and
# projects the next value.
def rolling_linear_regression_line(data, window=5):
    predictions = []

    # Loop through the entire dataset
    for i in range(len(data)):

        # Not enough points yet, append NaN
        if i + 1 < window:
            predictions.append(np.nan)
        else:
            # Select the last `window` points for regression
            y = np.array(data[i+1-window:i+1])
            x = np.arange(window)

            # Calculate slope (m) and intercept (b) using least squares formula
            N = window
            sum_x = np.sum(x)
            sum_y = np.sum(y)
            sum_x2 = np.sum(x**2)
            sum_xy = np.sum(x * y)
            m = (N * sum_xy - sum_x * sum_y) / (N * sum_x2 - sum_x**2)
            b = (sum_y - m * sum_x) / N

            # Predict the next value (t+1) based on linear trend
            predictions.append(m * (window-1) + b)

    return predictions

# ========================= Exponential Moving Average ========================
# Calculates a weighted moving average giving more importance to recent values.
def exponential_moving_average(data, window=5):
    ema = []
    alpha = 2 / (window + 1)  # smoothing factor

    # Loop through data and compute EMA
    for i, price in enumerate(data):
        if i == 0:
            ema.append(price)  # first value is itself
        else:
            ema_value = alpha * price + (1 - alpha) * ema[-1]
            ema.append(ema_value)

    return ema

# Simple wrapper to get the next EMA value (t+1)
def predict_next_ema(data, window=5):
    ema_values = exponential_moving_average(data, window)
    return ema_values[-1]

# ======================= Rolling Polynomial Regression ======================
# Fits a polynomial of a given degree over a rolling window to approximate curves
# and predict trends beyond the last point.
def rolling_polynomial_curve(data, window=5, degree=2):
    curve_segments = []

    for i in range(len(data)):
        if i + 1 < window:
            continue  # skip if not enough points yet

        y_window = np.array(data[i+1-window:i+1])       # windowed y values
        coeffs = np.polyfit(np.arange(window), y_window, degree)  # polynomial fit
        x_smooth = np.linspace(0, window-1, 50) + (i+1-window)  # fine x for plotting
        y_smooth = np.polyval(coeffs, np.linspace(0, window-1, 50))  # evaluated y

        curve_segments.append((x_smooth, y_smooth))

    return curve_segments

# ========================= Rolling Autoregressive (AR) =======================
# Predicts the next value based on an average of the last `order` points.
def rolling_autoregressive_predict(data, order=3):
    predictions = []
    data = np.array(data)

    for i in range(len(data)):
        if i < order:
            predictions.append(np.nan)
        else:
            X = data[i-order:i]               # last `order` points
            phi = np.ones(order) / order      # simple average weights
            y_pred = np.dot(phi, X)           # predicted next value
            predictions.append(y_pred)

    return predictions

# ======================== Rate-of-Change Extrapolation =======================
# Predicts the next value by calculating the average rate of change over the
# last `lookback` points.
def rate_of_change_predict_next(data, lookback=3):
    data = np.array(data)

    if len(data) < lookback + 1:
        raise ValueError("Not enough points for the given lookback")

    # Linear extrapolation: last value + average rate of change
    rate = (data[-1] - data[-lookback-1]) / lookback
    return data[-1] + rate
