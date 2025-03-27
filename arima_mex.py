import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_absolute_error, mean_squared_error
from itertools import product
import warnings
import scipy.stats as stats

warnings.filterwarnings("ignore")  # Suppress warnings for cleaner output

# ============================
# Step 1: Load & Preprocess Data
# ============================

df = pd.read_excel("quarterly_rates_aggregates_v2.xlsx")
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Perform first-order differencing
df['NII_diff'] = df['swedb_nii'].diff()

# ============================
# Step 2: Function to Optimize ARIMA Order via Grid Search
# ============================

def optimize_arima(data, p_values, d_values, q_values, window=4):
    best_score, best_cfg = float("inf"), None
    results = []

    for p, d, q in product(p_values, d_values, q_values):
        try:
            _, mae, _, _ = rolling_backtest_arima(data, order=(p, d, q), window=window)
            results.append(((p, d, q), mae))
            if mae < best_score:
                best_score, best_cfg = mae, (p, d, q)
        except:
            continue  # Skip invalid models

    return best_cfg, sorted(results, key=lambda x: x[1])

# ============================
# Step 3: Rolling Window Backtesting
# ============================

def rolling_backtest_arima(data, order, window=4, plot=False):
    """
    Performs a rolling window backtest on ARIMA and optionally plots the forecast.

    Parameters:
    - data: Time series data (Pandas Series).
    - order: ARIMA model order (p, d, q).
    - window: Number of quarters to predict in each step.
    - plot: If True, plots the forecast vs actual data.

    Returns:
    - DataFrame with actual and forecasted values.
    - Performance metrics (MAE, RMSE).
    - Residuals for residual analysis.
    """
    actual_values, forecasted_values, dates, residuals = [], [], [], []

    for start in range(0, len(data) - window - 1):
        train = data[: start + window]
        test = data[start + window: start + window + 1]

        if len(test) == 0:
            break

        model = ARIMA(train, order=order)
        model_fit = model.fit()

        forecast = model_fit.forecast(steps=len(test))

        actual_values.extend(test.values)
        forecasted_values.extend(forecast)
        dates.extend(test.index)
        residuals.extend(test.values - forecast)

    backtest_results = pd.DataFrame({
        "Date": dates,
        "Actual NII": actual_values,
        "Forecasted NII": forecasted_values,
        "Residuals": residuals
    })

    mae = mean_absolute_error(actual_values, forecasted_values)
    rmse = np.sqrt(mean_squared_error(actual_values, forecasted_values))

    # ✅ **Only plot for the final model**
    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(data.index, data, label="Actual Data", color="black", alpha=0.6)
        plt.plot(dates, forecasted_values, label="Rolling Forecast", color="red", linestyle="dashed")
        plt.scatter(dates, forecasted_values, color="red", marker="o", label="Forecast Points")
        plt.legend()
        plt.xlabel("Date")
        plt.ylabel("NII Values")
        plt.title("Rolling Window Backtest: Actual vs. Forecasted")
        plt.grid()
        plt.show()

    return backtest_results, mae, rmse, residuals


# ============================
# Step 4: Residual Diagnostics
# ============================

def residual_diagnostics(residuals):
    print("\n **Residual Analysis** \n")

    # ADF Test for Stationarity
    adf_test = adfuller(residuals)
    print(f" ADF Test p-value: {adf_test[1]:.5f} (Should be < 0.05 for stationarity)")

    # Ljung-Box Test for Autocorrelation
    lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
    print(f" Ljung-Box p-value: {lb_test['lb_pvalue'].values[0]:.5f} (Should be > 0.05 for no autocorrelation)")

    # Shapiro-Wilk Test for Normality
    shapiro_test = stats.shapiro(residuals)
    print(f" Shapiro-Wilk p-value: {shapiro_test.pvalue:.5f} (Should be > 0.05 for normality)")

    # Q-Q Plot
    plt.figure(figsize=(6, 4))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title("Q-Q Plot of Residuals")
    plt.show()

    # Histogram of Residuals
    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=20, color='blue', alpha=0.7)
    plt.title("Residual Histogram")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.grid()
    plt.show()

# ============================
# Step 5: Forecast Future Values
# ============================

def forecast_future_values(data, order, steps=8):
    model = ARIMA(data, order=order)
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=steps)

    future_dates = pd.date_range(data.index[-1], periods=steps, freq='Q')

    # Plot Actual vs Forecast
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data, label="Actual", color="black")
    plt.plot(future_dates, forecast, label="Forecast", linestyle="dashed", color="red")
    plt.legend()
    plt.title("ARIMA Forecast for Future Periods")
    plt.show()

    return forecast

# ============================
# Step 6: Evaluate Model on Test Set
# ============================

def evaluate_on_test_set(data, order, split_ratio=0.8):
    train_size = int(len(data) * split_ratio)
    train, test = data[:train_size], data[train_size:]

    model = ARIMA(train, order=order)
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=len(test))

    mae = mean_absolute_error(test, forecast)
    rmse = np.sqrt(mean_squared_error(test, forecast))

    print(f"\n Test Set Evaluation:\n - MAE: {mae:.5f}\n - RMSE: {rmse:.5f}")

    # Plot Train, Test & Forecast
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train, label="Train Data", color="blue")
    plt.plot(test.index, test, label="Test Data", color="black")
    plt.plot(test.index, forecast, label="Forecast", linestyle="dashed", color="red")
    plt.legend()
    plt.title("Train-Test Forecasting")
    plt.show()

# ============================
# Run the Full ARIMA Pipeline
# ============================

# **1. Optimize ARIMA order (No Plots During Grid Search)**
p_values, d_values, q_values = range(0, 4), range(0, 2), range(0, 4)
best_order, _ = optimize_arima(df['swedb_nii'].dropna(), p_values, d_values, q_values, window=4)

print(f"\n Best ARIMA Order: {best_order}")

#  **2. Final Backtest Using the Optimal Parameters (Plot Only Once)**
backtest_results, _, _, residuals = rolling_backtest_arima(df['swedb_nii'].dropna(), order=best_order, window=4, plot=True)

# **3. Perform Residual Analysis**
residual_diagnostics(residuals)

#  **4. Forecast Future Values**
forecast_future_values(df['swedb_nii'].dropna(), order=best_order, steps=8)

#  **5. Evaluate on Test Set**
evaluate_on_test_set(df['swedb_nii'].dropna(), order=best_order, split_ratio=0.8)
