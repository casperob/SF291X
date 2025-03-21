import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
import scipy.stats as stats
import warnings

warnings.filterwarnings("ignore")

# 1. Load and preprocess data
df = pd.read_excel("quarterly_averages.xlsx")
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

y = df['swedb_nii']
X = df.drop(columns=["swedb_nii"])

# Differencing
y_diff = y.diff().dropna()
X_diff = X.diff().dropna()
X_diff = X_diff.loc[y_diff.index]

# Standardize
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_diff), index=X_diff.index, columns=X_diff.columns)

# PCA
pca = PCA(n_components=0.95)
X_pca = pd.DataFrame(pca.fit_transform(X_scaled), index=X_scaled.index)
print(f"Number of PCA components retained: {pca.n_components_}")

y_diff = y_diff.loc[X_pca.index]

# Lag features
def create_lagged_features(y, X, lags=4):
    data = pd.DataFrame({'y': y})
    for lag in range(1, lags+1):
        data[f'y_lag_{lag}'] = y.shift(lag)
    X.columns = [f'PC{i+1}' for i in range(X.shape[1])]
    data = pd.concat([data, X], axis=1)
    return data.dropna()

lagged_data = create_lagged_features(y_diff, X_pca, lags=4)
y_supervised = lagged_data['y']
X_supervised = lagged_data.drop(columns='y')
X_supervised.columns = X_supervised.columns.astype(str)  # Fix for sklearn compatibility

# 2. Rolling forecast with MLPRegressor
def rolling_backtest_mlp(y, X, window=4, plot=False):
    actual, predicted, dates, residuals = [], [], [], []

    for start in range(0, len(y) - window, window):
        train_y, train_X = y[:start + window], X[:start + window]
        test_y, test_X = y[start + window:start + 2 * window], X[start + window:start + 2 * window]

        if len(test_y) == 0:
            break

        model = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu',
                             solver='adam', max_iter=500, random_state=42)
        model.fit(train_X, train_y)
        forecast = model.predict(test_X)

        actual.extend(test_y.values)
        predicted.extend(forecast)
        dates.extend(test_y.index)
        residuals.extend(test_y.values - forecast)

    backtest_df = pd.DataFrame({
        "Date": dates,
        "Actual": actual,
        "Forecast": predicted,
        "Residuals": residuals
    })

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(y.index, y, label="Actual", color="black", alpha=0.6)
        plt.plot(dates, predicted, label="NN Forecast", linestyle="--", color="green")
        plt.scatter(dates, predicted, color="green", label="Forecast Points")
        plt.title("MLP Neural Network Rolling Forecast")
        plt.xlabel("Date")
        plt.ylabel("NII (Differenced)")
        plt.legend()
        plt.grid()
        plt.show()

    return backtest_df, mae, rmse, residuals

# 3. Residual diagnostics (same as before)
def residual_diagnostics(residuals):
    print("\n **Residual Analysis** \n")
    adf_test = adfuller(residuals)
    print(f" ADF Test p-value: {adf_test[1]:.5f} (Should be < 0.05)")
    lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
    print(f" Ljung-Box p-value: {lb_test['lb_pvalue'].values[0]:.5f} (Should be > 0.05)")
    shapiro_test = stats.shapiro(residuals)
    print(f" Shapiro-Wilk p-value: {shapiro_test.pvalue:.5f} (Should be > 0.05)")

    plt.figure(figsize=(6, 4))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title("Q-Q Plot of Residuals")
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=20, alpha=0.7)
    plt.title("Residual Histogram")
    plt.xlabel("Residuals")
    plt.grid()
    plt.show()

# 4. Run rolling backtest
backtest_df, mae, rmse, residuals = rolling_backtest_mlp(y_supervised, X_supervised, window=4, plot=True)

print(f"\nMLP Regressor MAE: {mae:.5f}, RMSE: {rmse:.5f}")
residual_diagnostics(residuals)
