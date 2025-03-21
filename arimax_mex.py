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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



warnings.filterwarnings("ignore")  # Suppress warnings for cleaner output

# Load Data
df = pd.read_excel("quarterly_averages.xlsx")
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Target Variable
y = df['swedb_nii']

# Select Exogenous Variables (Example: Based on Correlation or Feature Importance)

#X = df[['STIBOR3M', 'unempl_rate', 'swedb_loan_deposit_ratio', 'import_msek',
#       'net_trade', 'swe_gdp', 'swedb_customer_loans_bnsek', 'swe_nat_debt',
#       'quarterly_inflation']]

X = df.drop(columns=["swedb_nii"])

# Ensure Data is Stationary (Differencing if Necessary)
y_diff = y.diff().dropna()
X_diff = X.diff().dropna()

# Align Indices
X_diff = X_diff.loc[y_diff.index]

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_diff), index=X_diff.index, columns=X_diff.columns)

# Apply PCA
pca = PCA(n_components=0.95)  # Retains 95% of the variance
X_pca = pca.fit_transform(X_scaled)

# Convert back to DataFrame with meaningful indices
X_pca = pd.DataFrame(X_pca, index=X_diff.index)

print(f"Number of PCA components retained: {pca.n_components_}")

# Ensure y (NII) and X_pca are properly aligned
y_diff = y_diff.loc[X_pca.index]

def optimize_arimax(y, X, p_values, d_values, q_values, window=4):
    best_score, best_cfg = float("inf"), None
    results = []

    for p, d, q in product(p_values, d_values, q_values):
        try:
            _, mae, _, _ = rolling_backtest_arimax(y, X, order=(p, d, q), window=window)
            results.append(((p, d, q), mae))
            if mae < best_score:
                best_score, best_cfg = mae, (p, d, q)
        except:
            continue  # Skip invalid models

    return best_cfg, sorted(results, key=lambda x: x[1])


def rolling_backtest_arimax(y, X, order, window=4, plot=False):

    actual_values, forecasted_values, dates, residuals = [], [], [], []

    for start in range(0, len(y) - window, window):
        train_y, train_X = y[: start + window], X[: start + window]
        test_y, test_X = y[start + window: start + 2 * window], X[start + window: start + 2 * window]

        if len(test_y) == 0:
            break

        model = ARIMA(train_y, order=order, exog=train_X)
        model_fit = model.fit()

        forecast = model_fit.forecast(steps=len(test_y), exog=test_X)

        actual_values.extend(test_y.values)
        forecasted_values.extend(forecast)
        dates.extend(test_y.index)
        residuals.extend(test_y.values - forecast)

    backtest_results = pd.DataFrame({
        "Date": dates,
        "Actual NII": actual_values,
        "Forecasted NII": forecasted_values,
        "Residuals": residuals
    })

    mae = mean_absolute_error(actual_values, forecasted_values)
    rmse = np.sqrt(mean_squared_error(actual_values, forecasted_values))

    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(y.index, y, label="Actual Data", color="black", alpha=0.6)
        plt.plot(dates, forecasted_values, label="Rolling Forecast", color="red", linestyle="dashed")
        plt.scatter(dates, forecasted_values, color="red", marker="o", label="Forecast Points")
        plt.legend()
        plt.xlabel("Date")
        plt.ylabel("NII Values")
        plt.title("Rolling Window Backtest: Actual vs. Forecasted")
        plt.grid()
        plt.show()

    return backtest_results, mae, rmse, residuals

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

def forecast_future_values_arimax(y, X, order, steps=8):

    model = ARIMA(y, order=order, exog=X)
    model_fit = model.fit()

    future_X = X.iloc[-steps:]  # Assuming you have future values of predictors
    forecast = model_fit.forecast(steps=steps, exog=future_X)

    future_dates = pd.date_range(y.index[-1], periods=steps, freq='Q')

    plt.figure(figsize=(12, 6))
    plt.plot(y.index, y, label="Actual", color="black")
    plt.plot(future_dates, forecast, label="Forecast", linestyle="dashed", color="red")
    plt.legend()
    plt.title("ARIMAX Forecast for Future Periods")
    plt.show()

    return forecast

def evaluate_on_test_set_arimax(y, X, order, split_ratio=0.8):

    train_size = int(len(y) * split_ratio)
    train_y, train_X = y[:train_size], X[:train_size]
    test_y, test_X = y[train_size:], X[train_size:]

    model = ARIMA(train_y, order=order, exog=train_X)
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=len(test_y), exog=test_X)

    mae = mean_absolute_error(test_y, forecast)
    rmse = np.sqrt(mean_squared_error(test_y, forecast))

    print(f"\n Test Set Evaluation:\n - MAE: {mae:.5f}\n - RMSE: {rmse:.5f}")

    plt.figure(figsize=(12, 6))
    plt.plot(train_y.index, train_y, label="Train Data", color="blue")
    plt.plot(test_y.index, test_y, label="Test Data", color="black")
    plt.plot(test_y.index, forecast, label="Forecast", linestyle="dashed", color="red")
    plt.legend()
    plt.title("Train-Test Forecasting (ARIMAX)")
    plt.show()


# **1. Optimize ARIMAX order (No Plots During Grid Search)**
p_values, d_values, q_values = range(0, 4), range(0, 2), range(0, 4)
best_order, _ = optimize_arimax(y_diff, X_pca, p_values, d_values, q_values, window=4)

print(f"\n Best ARIMAX Order: {best_order}")

# **2. Final Backtest Using the Optimal Parameters (Plot Only Once)**
backtest_results, _, _, residuals = rolling_backtest_arimax(y_diff, X_pca, order=best_order, window=4, plot=True)

# **3. Perform Residual Analysis**
residual_diagnostics(residuals)

# **4. Forecast Future Values**
forecast_future_values_arimax(y_diff, X_pca, order=best_order, steps=8)

# **5. Evaluate on Test Set**
evaluate_on_test_set_arimax(y_diff, X_pca, order=best_order, split_ratio=0.8)

