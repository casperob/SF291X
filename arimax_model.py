# models/arimax_model.py

def run_arimax(y, X, window=4, order=(2, 0, 2)):
    from statsmodels.tsa.arima.model import ARIMA
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import numpy as np

    actual, forecasted = [], []

    for start in range(0, len(y) - window - 1):
        train_y = y[:start + window]
        train_X = X[:start + window]
        test_y = y[start + window : start + window + 1]
        test_X = X[start + window : start + window + 1]

        try:
            model = ARIMA(train_y, order=order, exog=train_X)
            model_fit = model.fit()
            fc = model_fit.forecast(steps=1, exog=test_X)
            forecasted.extend(fc)
            actual.extend(test_y.values)
        except Exception as e:
            print(f"ARIMAX error: {e}")
            continue

    return {
        "model": "ARIMAX",
        "MAE": mean_absolute_error(actual, forecasted),
        "RMSE": np.sqrt(mean_squared_error(actual, forecasted))
    }

