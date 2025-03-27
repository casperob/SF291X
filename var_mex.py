import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings
from tensorflow.python.ops.initializers_ns import variables
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import scipy.stats as stats
import statsmodels.api as sm

warnings.filterwarnings("ignore")

# Load Data
df = pd.read_excel("normalized_quarterly_averages.xlsx")
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Select Variables
variables = ['swedb_nii', 'EURIBOR3M', 'swe_gdp', 'swedb_loan_deposit_ratio']
data = df[variables]
#data = df.drop(columns="swedb_nii")

# Ensure Stationarity (First Differences)
data_diff = data.diff().dropna()

#pca = PCA(n_components=0.95)

#pca_data = pca.fit_transform(data_diff)
#print(f"Number of PCA components retained: {pca.n_components_}")


def select_var_lag(data_df, maxlags):
    """Selects the optimal lag order for VAR using AIC/BIC."""
    model = VAR(data)
    result = model.select_order(maxlags)
    return result.aic, result.bic, result.fpe, result.hqic, result.aic


# Find Optimal Lag
opt_lag = select_var_lag(data, maxlags=1)
lag_order = opt_lag[0]
print(f"Optimal VAR Lag Order (AIC): {lag_order}")


def rolling_backtest_var(data, lags, window=4):
    """Performs rolling backtesting on the VAR model."""
    actual_values, forecasted_values, dates = [], [], []

    for start in range(0, len(data) - window - 1):
        train = data.iloc[:start + window]
        test = data.iloc[start + window : start + window + 1]

        if len(test) == 0:
            break

        model = VAR(train)
        model_fitted = model.fit(lags)
        forecast = model_fitted.forecast(train.values[-lags:], steps=len(test))
        forecast_df = pd.DataFrame(forecast, index=test.index, columns=test.columns)

        actual_values.extend(test['swedb_nii'].values)
        forecasted_values.extend(forecast_df['swedb_nii'].values)
        dates.extend(test.index)

    backtest_results = pd.DataFrame({
        "Date": dates,
        "Actual NII": actual_values,
        "Forecasted NII": forecasted_values
    })

    mae = mean_absolute_error(actual_values, forecasted_values)
    rmse = np.sqrt(mean_squared_error(actual_values, forecasted_values))

    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['swedb_nii'], label="Actual Data", color="black", alpha=0.6)
    plt.plot(dates, forecasted_values, label="Rolling Forecast", color="red", linestyle="dashed")
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("NII Values")
    plt.title("Rolling Window Backtest: Actual vs. Forecasted")
    plt.grid()
    plt.show()

    return backtest_results, mae, rmse


# Perform Rolling Backtest
backtest_results, mae, rmse = rolling_backtest_var(data, lags=lag_order, window=4)
print(f"\nBacktest Performance:\n - MAE: {mae:.5f}\n - RMSE: {rmse:.5f}")




def forecast_future_values_var(data, lags, steps=8):
    """Forecasts future values using the fitted VAR model."""
    model = VAR(data)
    model_fitted = model.fit(lags)
    forecast = model_fitted.forecast(data.values[-lags:], steps=steps)
    forecast_df = pd.DataFrame(forecast, index=pd.date_range(data.index[-1], periods=steps, freq='Q'),
                               columns=data.columns)

    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['swedb_nii'], label="Actual", color="black")
    plt.plot(forecast_df.index, forecast_df['swedb_nii'], label="Forecast", linestyle="dashed", color="red")
    plt.legend()
    plt.title("VAR Forecast for Future Periods")
    plt.show()

    return forecast_df


# Forecast Future Values
forecast_df = forecast_future_values_var(data, lags=lag_order, steps=8)


def evaluate_on_test_set_var(data, lags, split_ratio=0.8):
    """Splits data into train-test sets and evaluates VAR forecasting."""
    train_size = int(len(data) * split_ratio)
    train, test = data[:train_size], data[train_size:]

    model = VAR(train)
    model_fitted = model.fit(lags)
    forecast = model_fitted.forecast(train.values[-lags:], steps=len(test))
    forecast_df = pd.DataFrame(forecast, index=test.index, columns=test.columns)

    mae = mean_absolute_error(test['swedb_nii'], forecast_df['swedb_nii'])
    rmse = np.sqrt(mean_squared_error(test['swedb_nii'], forecast_df['swedb_nii']))

    print(f"\n Test Set Evaluation:\n - MAE: {mae:.5f}\n - RMSE: {rmse:.5f}")

    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train['swedb_nii'], label="Train Data", color="blue")
    plt.plot(test.index, test['swedb_nii'], label="Test Data", color="black")
    plt.plot(test.index, forecast_df['swedb_nii'], label="Forecast", linestyle="dashed", color="red")
    plt.legend()
    plt.title("Train-Test Forecasting (VAR)")
    plt.show()



# Evaluate Model on Test Set
evaluate_on_test_set_var(data, lags=lag_order, split_ratio=0.8)

# Extract NII Equation Coefficients
model = VAR(data)
model_fitted = model.fit(lag_order)
nii_equation = model_fitted.params.loc[:, 'swedb_nii']
print("\nCoefficients in the NII equation:")
print(nii_equation)

# Residual Diagnostics
residuals = model_fitted.resid['swedb_nii']
print("\n**Residual Diagnostics for NII**")

# ADF Test
adf_result = adfuller(residuals)
print(f"ADF Test p-value: {adf_result[1]:.5f} (Should be < 0.05 for stationarity)")

# Ljung-Box Test
ljung_result = acorr_ljungbox(residuals, lags=[10], return_df=True)
print(f"Ljung-Box p-value: {ljung_result['lb_pvalue'].iloc[0]:.5f} (Should be > 0.05 for no autocorrelation)")

# Shapiro-Wilk Test
shapiro_result = stats.shapiro(residuals)
print(f"Shapiro-Wilk p-value: {shapiro_result.pvalue:.5f} (Should be > 0.05 for normality)")

# Q-Q Plot
plt.figure(figsize=(6, 4))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("Q-Q Plot of Residuals (NII)")
plt.grid()
plt.show()

# Histogram
plt.figure(figsize=(6, 4))
plt.hist(residuals, bins=20, color='blue', alpha=0.7)
plt.title("Residual Histogram (NII)")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.grid()
plt.show()
