
def run_rf(y, X, window=4):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import numpy as np

    actual, forecasted = [], []

    for start in range(0, len(y) - window - 1):
        train_y = y[:start + window]
        train_X = X[:start + window]
        test_y = y[start + window : start + window + 1]
        test_X = X[start + window : start + window + 1]

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(train_X, train_y)
        fc = model.predict(test_X)

        forecasted.extend(fc)
        actual.extend(test_y.values)

    return {
        "model": "RandomForest",
        "MAE": mean_absolute_error(actual, forecasted),
        "RMSE": np.sqrt(mean_squared_error(actual, forecasted))
    }

