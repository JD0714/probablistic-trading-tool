import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Prediction_Methods
from datetime import datetime

# ----------------------------- Fetch Stock Prices ---------------------------
# Safely fetches stock prices from Yahoo Finance and ensures consecutive 1-min intervals.
def fetch_stock_prices(ticker, num_points):

    data = yf.Ticker(ticker).history(period="5d", interval="1m")

    if data.empty:
        print(f"No data available for {ticker}.")
        return pd.DataFrame(columns=["Timestamp", "Price"])

    data = data.sort_index()                       # sort by timestamp
    timestamps = pd.to_datetime(data.index)        # ensure datetime format

    # Search for most recent consecutive block of `num_points`
    for start in range(len(timestamps) - num_points, -1, -1):
        block = timestamps[start:start + num_points]
        deltas = block.to_series().diff().dropna()

        if all(deltas == pd.Timedelta(minutes=1)):
            valid_block = data.iloc[start:start + num_points]
            price_table = pd.DataFrame({
                "Timestamp": valid_block.index,
                "Price": valid_block["Close"].values
            })
            return price_table

    print(f"Could not find {num_points} consecutive 1-minute intervals.")
    return pd.DataFrame(columns=["Timestamp", "Price"])

# =============================================================================
# ======================== Directional Probability Mode ======================
# =============================================================================
# Uses the last 100 points to evaluate the probability that each prediction
# method correctly predicts the direction of the next value.
def Directional_Probability_Mode(ticker, window):
    print(f"Evaluating directional probability for {ticker}...\n")

    # Always use 100 points
    price_table = fetch_stock_prices(ticker, 100)

    if price_table.empty or len(price_table) < 100:
        print("Not enough data to evaluate.")
        return

    data = price_table["Price"].tolist()
    groups = [data[i:i+10] for i in range(0, 100, 10)]  # 10 groups of 10
    results = {model:0 for model in ["Rolling Linear Regression",
                                     "Exponential Moving Average",
                                     "Polynomial Regression",
                                     "Autoregressive",
                                     "Rate of Change"]}
    valid_counts = {model:0 for model in results}

    # Loop through each 10-point group
    for group in groups:
        first9 = group[:9]
        actual10 = group[9]
        last9 = first9[-1]

        # Skip if 10th value exactly equals 9th
        if actual10 == last9:
            continue

        actual_direction = np.sign(actual10 - last9)

        # ----- Rolling Linear Regression -----
        pred_lr = rolling_linear_regression_line(first9, window)[-1]
        if not np.isnan(pred_lr):
            pred_direction = np.sign(pred_lr - last9)
            valid_counts["Rolling Linear Regression"] += 1
            if pred_direction == actual_direction:
                results["Rolling Linear Regression"] += 1

        # ----- EMA -----
        pred_ema = predict_next_ema(first9, window)
        pred_direction = np.sign(pred_ema - last9)
        valid_counts["Exponential Moving Average"] += 1
        if pred_direction == actual_direction:
            results["Exponential Moving Average"] += 1

        # ----- Polynomial Regression -----
        if len(first9) >= window:
            y_window = np.array(first9[-window:])
            coeffs = np.polyfit(np.arange(window), y_window, 2)
            pred_poly = np.polyval(coeffs, window-1)
            pred_direction = np.sign(pred_poly - last9)
            valid_counts["Polynomial Regression"] += 1
            if pred_direction == actual_direction:
                results["Polynomial Regression"] += 1

        # ----- Autoregressive -----
        if len(first9) >= 3:
            pred_ar = rolling_autoregressive_predict(first9, order=3)[-1]
            if not np.isnan(pred_ar):
                pred_direction = np.sign(pred_ar - last9)
                valid_counts["Autoregressive"] += 1
                if pred_direction == actual_direction:
                    results["Autoregressive"] += 1

        # ----- Rate-of-Change -----
        if len(first9) >= 4:
            pred_roc = rate_of_change_predict_next(first9, lookback=3)
            pred_direction = np.sign(pred_roc - last9)
            valid_counts["Rate of Change"] += 1
            if pred_direction == actual_direction:
                results["Rate of Change"] += 1

    # ----- Display results -----
    print(f"Predictions for stock {ticker}:\n")
    for model in results:
        if valid_counts[model] > 0:
            probability = (results[model] / valid_counts[model]) * 100
            print(f"{model} : {probability:.2f}%")
        else:
            print(f"{model} : Not enough valid data")

def Directional_Probability_Mode_AutoWindow(ticker):
    """
    Auto-window directional probability mode for a single ticker.
    Loops through window sizes 2–9, calculates directional probability for each
    method, and returns the best probability + window for each method.
    """
    print(f"\nEvaluating best directional probabilities for {ticker} across multiple windows...\n")
    
    price_table = fetch_stock_prices(ticker, 100)
    if price_table.empty or len(price_table) < 100:
        print("Not enough data to evaluate.")
        return

    data = price_table["Price"].tolist()
    groups = [data[i:i+10] for i in range(0, 100, 10)]  # 10 groups of 10

    methods = ["Rolling Linear Regression", "Exponential Moving Average",
               "Polynomial Regression", "Autoregressive", "Rate of Change"]

    best_results = {method: {"probability": 0, "window": None} for method in methods}

    # Loop through all possible windows
    for window in range(2, 10):
        results = {method: 0 for method in methods}
        valid_counts = {method: 0 for method in methods}

        for group in groups:
            first9 = group[:9]
            actual10 = group[9]
            last9 = first9[-1]

            if actual10 == last9:
                continue
            actual_direction = np.sign(actual10 - last9)

            # ----- Rolling Linear Regression -----
            if len(first9) >= window:
                pred_lr = rolling_linear_regression_line(first9, window)[-1]
                if not np.isnan(pred_lr):
                    pred_direction = np.sign(pred_lr - last9)
                    valid_counts["Rolling Linear Regression"] += 1
                    if pred_direction == actual_direction:
                        results["Rolling Linear Regression"] += 1

            # ----- EMA -----
            if len(first9) >= window:
                pred_ema = predict_next_ema(first9, window)
                pred_direction = np.sign(pred_ema - last9)
                valid_counts["Exponential Moving Average"] += 1
                if pred_direction == actual_direction:
                    results["Exponential Moving Average"] += 1

            # ----- Polynomial Regression -----
            if len(first9) >= window:
                y_window = np.array(first9[-window:])
                coeffs = np.polyfit(np.arange(window), y_window, 2)
                pred_poly = np.polyval(coeffs, window-1)
                pred_direction = np.sign(pred_poly - last9)
                valid_counts["Polynomial Regression"] += 1
                if pred_direction == actual_direction:
                    results["Polynomial Regression"] += 1

            # ----- Autoregressive -----
            if len(first9) >= 3:
                pred_ar = rolling_autoregressive_predict(first9, order=3)[-1]
                if not np.isnan(pred_ar):
                    pred_direction = np.sign(pred_ar - last9)
                    valid_counts["Autoregressive"] += 1
                    if pred_direction == actual_direction:
                        results["Autoregressive"] += 1

            # ----- Rate-of-Change -----
            if len(first9) >= 4:
                pred_roc = rate_of_change_predict_next(first9, lookback=3)
                pred_direction = np.sign(pred_roc - last9)
                valid_counts["Rate of Change"] += 1
                if pred_direction == actual_direction:
                    results["Rate of Change"] += 1

        # Update best result for each method
        for method in methods:
            if valid_counts[method] > 0:
                probability = (results[method] / valid_counts[method]) * 100
                if probability > best_results[method]["probability"]:
                    best_results[method]["probability"] = probability
                    best_results[method]["window"] = window

    # ----- Display final best results -----
    print(f"Best directional probabilities for {ticker}:\n")
    print(f"{'Method':35} {'Probability (%)':15} {'Best Window'}")
    print("-" * 65)
    for method in methods:
        prob = best_results[method]["probability"]
        win = best_results[method]["window"]
        if win is not None:
            print(f"{method:35} {prob:15.2f} {win}")
        else:
            print(f"{method:35} {'N/A':15} {'N/A'}")

def Directional_Probability_Mode_MultiTicker(num_tickers):
    """
    Multi-ticker auto-window directional probability mode using up to 3 days of intraday 1-min data.
    Collects up to 100 consecutive 10-minute groups per ticker (or fewer if not enough data).
    Sweeps windows 2-9 for all methods and returns top 10 probabilities across all tickers/windows/methods.
    """
    # Top 100 popular tickers
    top_100_tickers = [
    "AAPL","MSFT","GOOGL","AMZN","TSLA","NVDA","META","BRK-B","JPM","UNH",
    "V","HD","PG","MA","DIS","BAC","ADBE","PFE","NFLX","KO",
    "INTC","XOM","CSCO","PEP","T","VZ","CVX","MRK","ABT","CRM",
    "WMT","NKE","ORCL","ACN","ABBV","COST","MCD","DHR","MDT","LLY",
    "TXN","QCOM","HON","NEE","PM","UNP","IBM","AMGN","LIN","SCHW",
    "PYPL","BMY","RTX","SPGI","TMUS","ADP","NOW","LOW","PLD","COST",
    "SBUX","GE","BLK","DE","CAT","AXP","MMM","MDLZ","SYK","CVS",
    "GILD","INTU","AMAT","TMO","ANTM","ZTS","CI","DUK","SO","EXC",
    "ELV","SCHW","FIS","APD","BDX","ECL","WM","ICE","ITW","VRTX",
    "CSX","CCI","ROST","FISV","PNC","ADI","HUM","TJX","BKNG","HCA",
    "LRCX","KMB","BSX","GM","MCO","MAR","ODFL","EOG","MET","CL"
]

    if num_tickers < 1 or num_tickers > 100:
        print("Number of tickers must be between 1 and 100.")
        return

    selected_tickers = top_100_tickers[:num_tickers]
    all_results = []

    for ticker in selected_tickers:
        print(f"\nAnalyzing ticker {ticker} (up to 3 days of 1-min data)...")
        try:
            raw_data = yf.Ticker(ticker).history(period="3d", interval="1m")
        except:
            raw_data = pd.DataFrame()

        if raw_data.empty:
            print(f"Skipping {ticker}: no 1-min intraday data for 3 days.")
            continue

        raw_data = raw_data.sort_index()
        timestamps = pd.to_datetime(raw_data.index)
        prices = raw_data["Close"].tolist()

        # Collect consecutive 10-min groups, up to 100
        consecutive_blocks = []
        i = len(timestamps) - 10
        while i >= 0 and len(consecutive_blocks) < 100:
            block = timestamps[i:i+10]
            deltas = block.to_series().diff().dropna()
            if all(deltas == pd.Timedelta(minutes=1)):
                consecutive_blocks.insert(0, prices[i:i+10])
                i -= 10
            else:
                i -= 1

        if len(consecutive_blocks) == 0:
            print(f"Skipping {ticker}: no valid 10-min groups found.")
            continue
        else:
            print(f"Found {len(consecutive_blocks)} valid 10-min groups for {ticker}.")

        methods = ["Rolling Linear Regression", "Exponential Moving Average",
                   "Polynomial Regression", "Autoregressive", "Rate of Change"]

        # Sweep windows 2-9
        for window in range(2, 10):
            results = {method: 0 for method in methods}
            valid_counts = {method: 0 for method in methods}

            for group in consecutive_blocks:
                first9 = group[:9]
                actual10 = group[9]
                last9 = first9[-1]
                if actual10 == last9:
                    continue
                actual_direction = np.sign(actual10 - last9)

                # ----- Rolling Linear Regression -----
                if len(first9) >= window:
                    pred_lr = rolling_linear_regression_line(first9, window)[-1]
                    if not np.isnan(pred_lr):
                        pred_direction = np.sign(pred_lr - last9)
                        valid_counts["Rolling Linear Regression"] += 1
                        if pred_direction == actual_direction:
                            results["Rolling Linear Regression"] += 1

                # ----- EMA -----
                if len(first9) >= window:
                    pred_ema = predict_next_ema(first9, window)
                    pred_direction = np.sign(pred_ema - last9)
                    valid_counts["Exponential Moving Average"] += 1
                    if pred_direction == actual_direction:
                        results["Exponential Moving Average"] += 1

                # ----- Polynomial Regression -----
                if len(first9) >= window:
                    coeffs = np.polyfit(np.arange(window), np.array(first9[-window:]), 2)
                    pred_poly = np.polyval(coeffs, window-1)
                    pred_direction = np.sign(pred_poly - last9)
                    valid_counts["Polynomial Regression"] += 1
                    if pred_direction == actual_direction:
                        results["Polynomial Regression"] += 1

                # ----- Autoregressive -----
                pred_ar = rolling_autoregressive_predict(first9, order=3)[-1]
                if not np.isnan(pred_ar):
                    pred_direction = np.sign(pred_ar - last9)
                    valid_counts["Autoregressive"] += 1
                    if pred_direction == actual_direction:
                        results["Autoregressive"] += 1

                # ----- Rate-of-Change -----
                if len(first9) >= 4:
                    pred_roc = rate_of_change_predict_next(first9, lookback=3)
                    pred_direction = np.sign(pred_roc - last9)
                    valid_counts["Rate of Change"] += 1
                    if pred_direction == actual_direction:
                        results["Rate of Change"] += 1

            # Store results for this ticker/window/method
            for method in methods:
                if valid_counts[method] > 0:
                    probability = (results[method] / valid_counts[method]) * 100
                    all_results.append({
                        "Probability": probability,
                        "Ticker": ticker,
                        "Window": window,
                        "Method": method
                    })

    # Sort by probability descending and display top 10
    top_10 = sorted(all_results, key=lambda x: x["Probability"], reverse=True)[:10]

    print("\nTop 10 directional probabilities across tickers/windows/methods:\n")
    print(f"{'Probability (%)':15} {'Ticker':8} {'Window':6} {'Method'}")
    print("-" * 60)
    for res in top_10:
        print(f"{res['Probability']:15.2f} {res['Ticker']:8} {res['Window']:6} {res['Method']}")

# =============================================================================
# ==================== Weighted Directional Probability Mode ==================
# =============================================================================

def Weighted_Directional_Probability_Mode(num_tickers):
    """
    Weighted Directional Probability Mode with non-trivial move filter.
    Inputs: number of tickers (1-100)
    Uses up to 3 days of 1-min intraday data (~500 minutes)
    Computes weighted probabilities where bigger predicted moves count more
    Only displays results with >=5 non-trivial comparisons
    """
    top_100_tickers = [
        "AAPL","MSFT","GOOGL","AMZN","TSLA","NVDA","META","BRK-B","JPM","UNH",
        "V","HD","PG","MA","DIS","BAC","ADBE","PFE","NFLX","KO",
        "INTC","XOM","CSCO","PEP","T","VZ","CVX","MRK","ABT","CRM",
        "WMT","NKE","ORCL","ACN","ABBV","COST","MCD","DHR","MDT","LLY",
        "TXN","QCOM","HON","NEE","PM","UNP","IBM","AMGN","LIN","SCHW",
        "PYPL","BMY","RTX","SPGI","TMUS","ADP","NOW","LOW","PLD","COST",
        "SBUX","GE","BLK","DE","CAT","AXP","MMM","MDLZ","SYK","CVS",
        "GILD","INTU","AMAT","TMO","ANTM","ZTS","CI","DUK","SO","EXC",
        "ELV","SCHW","FIS","APD","BDX","ECL","WM","ICE","ITW","VRTX",
        "CSX","CCI","ROST","FISV","PNC","ADI","HUM","TJX","BKNG","HCA",
        "LRCX","KMB","BSX","GM","MCO","MAR","ODFL","EOG","MET","CL"
    ]

    if num_tickers < 1 or num_tickers > 100:
        print("Number of tickers must be between 1 and 100.")
        return

    selected_tickers = top_100_tickers[:num_tickers]
    all_results = []

    for ticker in selected_tickers:
        print(f"\nAnalyzing ticker {ticker} (up to 3 days of 1-min data)...")
        try:
            raw_data = yf.Ticker(ticker).history(period="3d", interval="1m")
        except:
            raw_data = pd.DataFrame()

        if raw_data.empty:
            print(f"Skipping {ticker}: no 1-min intraday data for 3 days.")
            continue

        raw_data = raw_data.sort_index()
        timestamps = pd.to_datetime(raw_data.index)
        prices = raw_data["Close"].tolist()

        # Collect consecutive 10-min groups, max 100 or as many as available
        consecutive_blocks = []
        i = len(timestamps) - 10
        while i >= 0 and len(consecutive_blocks) < 100:
            block = timestamps[i:i+10]
            deltas = block.to_series().diff().dropna()
            if all(deltas == pd.Timedelta(minutes=1)):
                consecutive_blocks.insert(0, prices[i:i+10])
                i -= 10
            else:
                i -= 1

        if len(consecutive_blocks) == 0:
            print(f"Skipping {ticker}: no valid 10-min groups found.")
            continue
        else:
            print(f"Found {len(consecutive_blocks)} valid 10-min groups for {ticker}.")

        methods = ["Rolling Linear Regression", "Exponential Moving Average",
                   "Polynomial Regression", "Autoregressive", "Rate of Change"]

        # Sweep windows 2-9
        for window in range(2, 10):
            weighted_scores = {method: 0 for method in methods}
            total_weight = {method: 0 for method in methods}
            non_trivial_counts = {method: 0 for method in methods}

            for group in consecutive_blocks:
                first9 = group[:9]
                actual10 = group[9]
                last9 = first9[-1]
                if actual10 == last9:
                    continue
                actual_direction = np.sign(actual10 - last9)

                # Helper function
                def process_prediction(pred, method):
                    change = pred - last9
                    if abs(change) >= 0.01:
                        total_weight[method] += abs(change)
                        non_trivial_counts[method] += 1
                        if np.sign(change) == actual_direction:
                            weighted_scores[method] += abs(change)

                # Rolling Linear Regression
                if len(first9) >= window:
                    pred_lr = rolling_linear_regression_line(first9, window)[-1]
                    if not np.isnan(pred_lr):
                        process_prediction(pred_lr, "Rolling Linear Regression")

                # EMA
                if len(first9) >= window:
                    pred_ema = predict_next_ema(first9, window)
                    process_prediction(pred_ema, "Exponential Moving Average")

                # Polynomial Regression
                if len(first9) >= window:
                    coeffs = np.polyfit(np.arange(window), np.array(first9[-window:]), 2)
                    pred_poly = np.polyval(coeffs, window-1)
                    process_prediction(pred_poly, "Polynomial Regression")

                # Autoregressive
                pred_ar = rolling_autoregressive_predict(first9, order=3)[-1]
                if not np.isnan(pred_ar):
                    process_prediction(pred_ar, "Autoregressive")

                # Rate-of-Change
                if len(first9) >= 4:
                    pred_roc = rate_of_change_predict_next(first9, lookback=3)
                    process_prediction(pred_roc, "Rate of Change")

            # Store weighted results only if >=5 non-trivial moves
            for method in methods:
                if non_trivial_counts[method] >= 5 and total_weight[method] > 0:
                    weighted_percent = (weighted_scores[method] / total_weight[method]) * 100
                    all_results.append({
                        "WeightedPercent": weighted_percent,
                        "Ticker": ticker,
                        "Window": window,
                        "Method": method,
                        "NonTrivial": non_trivial_counts[method]
                    })

    # Sort top 10 weighted probabilities
    top_10 = sorted(all_results, key=lambda x: x["WeightedPercent"], reverse=True)[:10]

    # Display results
    print("\nTop 10 weighted directional probabilities (≥5 non-trivial moves):\n")
    print(f"{'Weighted (%)':15} {'Ticker':8} {'Window':6} {'Method':30} {'#Non-Trivial'}")
    print("-" * 80)
    for res in top_10:
        print(f"{res['WeightedPercent']:15.2f} {res['Ticker']:8} {res['Window']:6} {res['Method']:30} {res['NonTrivial']}")


# =============================================================================
# =================================== MAIN ===================================
# =============================================================================
def main():
    # --- Top-Level Choice ---
    mode_choice = input("Do you want 'approximations & graphs' or 'probability & cumulative analysis'? (g/p): ").strip().lower()

    if mode_choice == "g":
        # --- Existing Graphing Flow ---
        ticker = input("Enter the stock ticker to analyze: ").upper()
        num_points = int(input("Enter number of price points to fetch: "))
        price_table = fetch_stock_prices(ticker, num_points)

        if price_table.empty:
            print("No data fetched. Exiting.")
            return

        print("\nFetched price table:\n")
        print(price_table)

        # --- Choose prediction method ---
        print("\nChoose a prediction method:")
        print("1. Rolling Linear Regression")
        print("2. Exponential Moving Average (EMA)")
        print("3. Rolling Polynomial Regression (curved)")
        print("4. Rolling Autoregressive (AR)")
        print("5. Rate-of-Change Extrapolation")

        method = input("Enter 1, 2, 3, 4, or 5: ").strip()
        if method not in ['1','2','3','4','5']:
            raise ValueError("Invalid method choice.")

        data_values = price_table["Price"].tolist()
        plt.figure(figsize=(14,6))
        plt.plot(data_values, 'o-', label='Actual Price')

        if method in ['1','2','3']:
            window_size = int(input("Enter window size (number of past points): ").strip())
            if window_size <= 0 or window_size > len(data_values):
                raise ValueError("Window size must be positive and <= number of points.")

        # --- Call the selected prediction method (existing logic) ---
        if method == '1':
            predicted_line = rolling_linear_regression_line(data_values, window=window_size)
            plt.plot(predicted_line, 'r--', label=f'Rolling Linear Regression (window={window_size})')
            plt.title(f"{ticker} - Rolling Linear Regression Prediction")
        elif method == '2':
            ema_line = exponential_moving_average(data_values, window=window_size)
            next_point = predict_next_ema(data_values, window=window_size)
            plt.plot(ema_line, 'g--', label=f'EMA (window={window_size})')
            plt.scatter(len(data_values), next_point, color='purple', s=100, label='Next EMA Prediction')
            plt.title(f"{ticker} - EMA Prediction")
        elif method == '3':
            degree = int(input("Enter polynomial degree (e.g., 2 for quadratic): ").strip())
            curve_segments = rolling_polynomial_curve(data_values, window=window_size, degree=degree)
            for x_smooth, y_smooth in curve_segments:
                plt.plot(x_smooth, y_smooth, 'm-', alpha=0.6)
            plt.title(f"{ticker} - Rolling Polynomial Regression (degree={degree})")
        elif method == '4':
            order = int(input("Enter AR order (number of past points, e.g., 3): ").strip())
            predicted_ar = rolling_autoregressive_predict(data_values, order=order)
            plt.plot(predicted_ar, 'b--', label=f'Rolling AR (order={order})')
            plt.title(f"{ticker} - Rolling Autoregressive (AR) Prediction")
        else:
            lookback = int(input("Enter lookback period (number of past points, e.g., 3): ").strip())
            next_value = rate_of_change_predict_next(data_values, lookback=lookback)
            plt.plot([len(data_values)-1, len(data_values)], [data_values[-1], next_value], 'c--', label=f'Rate-of-Change Prediction (lookback={lookback})')
            plt.scatter(len(data_values), next_value, color='c', s=100)
            plt.title(f"{ticker} - Rate-of-Change Extrapolation")

        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)
        plt.show()

    elif mode_choice == "p":
        print("\nChoose probability mode:")
        print("1. Directional Probability Mode")
        print("2. Weighted Directional Probability Mode")
        print("3. Confidence Adjusted Trend Score")

        prob_mode = input("Enter 1, 2, or 3: ").strip()

        # --- If user selects Directional Probability Mode ---
        if prob_mode == "1":
            print("\nDirectional Probability Mode Options:")
            print("1. Regular Directional Probability (manual window)")
            print("2. Auto-Window Directional Probability (single ticker)")
            print("3. Multi-Ticker Auto-Window Directional Probability (top N tickers)")

            dp_choice = input("Enter 1, 2, or 3: ").strip()

            if dp_choice == '1':
                ticker = input("Enter stock ticker: ").upper()
                window = int(input("Enter window size: "))
                Directional_Probability_Mode(ticker, window)

            elif dp_choice == '2':
                ticker = input("Enter stock ticker: ").upper()
                Directional_Probability_Mode_AutoWindow(ticker)

            elif dp_choice == '3':
                num_tickers = int(input("Enter number of tickers to analyze from top 100 (1-100): "))
                Directional_Probability_Mode_MultiTicker(num_tickers)

            else:
                print("Invalid choice. Exiting.")
                return

        # --- Weighted Mode ---
        elif prob_mode == "2":
            num_tickers = int(input("Enter number of tickers to analyze from top 100 (1-100): "))
            Weighted_Directional_Probability_Mode(num_tickers)

        # --- Confidence Mode (existing flow, still needs ticker & window) ---
        elif prob_mode == "3":
            ticker = input("Enter stock ticker: ").upper()
            window = int(input("Enter window size: "))
            Confidence_Adjusted_Trend_Score(ticker, window)

        else:
            print("Invalid mode selected. Exiting.")
            return

if __name__ == "__main__":
    main()