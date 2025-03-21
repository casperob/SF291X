import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.model_selection import ParameterGrid
import warnings

warnings.filterwarnings("ignore")

# Load Data
df = pd.read_excel("quarterly_averages.xlsx")
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Target & Features
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


# Lagged target (for supervised learning)
def create_lagged_features(y, X, lags=1):
    data = pd.DataFrame({'y': y})
    for lag in range(1, lags + 1):
        data[f'y_lag_{lag}'] = y.shift(lag)
    data = pd.concat([data, X], axis=1)
    return data.dropna()


lagged_data = create_lagged_features(y_diff, X_pca, lags=4)

# Split target and features again
y_supervised = lagged_data['y']
X_supervised = lagged_data.drop(columns='y')
X_supervised.columns = X_supervised.columns.astype(str)


param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [4, 8, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2']
}

def optimize_random_forest(y, X, param_grid, window=4):
    best_score = float("inf")
    best_params = None
    results = []

    for params in ParameterGrid(param_grid):
        try:
            model = RandomForestRegressor(random_state=42, **params)
            actual, predicted, dates, residuals = [], [], [], []

            for start in range(0, len(y) - window, window):
                train_y, train_X = y[:start + window], X[:start + window]
                test_y, test_X = y[start + window:start + 2 * window], X[start + window:start + 2 * window]

                if len(test_y) == 0:
                    break

                model.fit(train_X, train_y)
                forecast = model.predict(test_X)

                actual.extend(test_y.values)
                predicted.extend(forecast)

            mae = mean_absolute_error(actual, predicted)
            results.append((params, mae))

            if mae < best_score:
                best_score = mae
                best_params = params

        except Exception as e:
            print(f"Skipping due to error: {e}")
            continue

    return best_params, sorted(results, key=lambda x: x[1])

best_params, all_results = optimize_random_forest(y_supervised, X_supervised, param_grid, window=4)
print(f"Best Parameters: {best_params}")


# Rolling backtest
def rolling_backtest_rf(y, X, window=4, plot=False):
    actual, predicted, dates, residuals = [], [], [], []

    for start in range(0, len(y) - window, window):
        train_y, train_X = y[:start + window], X[:start + window]
        test_y, test_X = y[start + window:start + 2 * window], X[start + window:start + 2 * window]

        if len(test_y) == 0:
            break

        model = RandomForestRegressor(random_state=42, **best_params)
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
        plt.plot(dates, predicted, label="Forecast", linestyle="--", color="red")
        plt.scatter(dates, predicted, color="red", label="Forecast Points")
        plt.title("Random Forest Rolling Forecast")
        plt.xlabel("Date")
        plt.ylabel("NII (Differenced)")
        plt.legend()
        plt.grid()
        plt.show()

    return backtest_df, mae, rmse, residuals


# Residual Diagnostics
def residual_diagnostics(residuals):
    print("\n **Residual Analysis** \n")

    adf_test = adfuller(residuals)
    print(f" ADF Test p-value: {adf_test[1]:.5f} (Should be < 0.05)")

    lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
    print(f" Ljung-Box p-value: {lb_test['lb_pvalue'].values[0]:.5f} (Should be > 0.05)")

    shapiro_test = stats.shapiro(residuals)
    print(f" Shapiro-Wilk p-value: {shapiro_test.pvalue:.5f} (Should be > 0.05)")

    # Q-Q Plot
    plt.figure(figsize=(6, 4))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title("Q-Q Plot of Residuals")
    plt.show()

    # Histogram
    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=20, alpha=0.7)
    plt.title("Residual Histogram")
    plt.xlabel("Residuals")
    plt.grid()
    plt.show()


# Train-Test Evaluation
def evaluate_rf_on_test(y, X, split_ratio=0.8):
    split = int(len(y) * split_ratio)
    train_X, test_X = X[:split], X[split:]
    train_y, test_y = y[:split], y[split:]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(train_X, train_y)
    forecast = model.predict(test_X)

    mae = mean_absolute_error(test_y, forecast)
    rmse = np.sqrt(mean_squared_error(test_y, forecast))

    print(f"\nTest Set Evaluation:\n - MAE: {mae:.5f}\n - RMSE: {rmse:.5f}")

    plt.figure(figsize=(12, 6))
    plt.plot(train_y.index, train_y, label="Train", color="blue")
    plt.plot(test_y.index, test_y, label="Test", color="black")
    plt.plot(test_y.index, forecast, label="Forecast", linestyle='--', color="red")
    plt.legend()
    plt.title("Random Forest Train-Test Forecast")
    plt.grid()
    plt.show()


# Execute full pipeline
backtest_df, mae, rmse, residuals = rolling_backtest_rf(y_supervised, X_supervised, window=4, plot=True)
residual_diagnostics(residuals)
evaluate_rf_on_test(y_supervised, X_supervised, split_ratio=0.8)
