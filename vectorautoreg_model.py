def run_var(y, X, window=4):
    from statsmodels.tsa.api import VAR
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import numpy as np
    import pandas as pd

    actual, forecasted = [], []
    data = pd.concat([y, X], axis=1).dropna()
    y_col = y.name or "target"

    for start in range(0, len(data) - window - 1):
        train = data.iloc[:start + window]
        test = data.iloc[start + window : start + window + 1]

        try:
            model = VAR(train)
            fit = model.fit(maxlags=1)
            fc = fit.forecast(train.values[-1:], steps=1)
            forecast_df = pd.DataFrame(fc, index=test.index, columns=train.columns)
            forecasted.append(forecast_df[y_col].iloc[0])
            actual.append(test[y_col].iloc[0])
        except Exception as e:
            print(f"VAR error: {e}")
            continue

    return {
        "model": "VAR",
        "MAE": mean_absolute_error(actual, forecasted),
        "RMSE": np.sqrt(mean_squared_error(actual, forecasted))
    }

