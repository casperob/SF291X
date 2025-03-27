import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Import your model wrappers
from models.arimax_model import run_arimax
from models.vectorautoreg_model import run_var
from models.rf_model import run_rf
from models.nn_model import run_nn

# Load data
df = pd.read_excel("quarterly_averages.xlsx")
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

y = df['swedb_nii']
X = df.drop(columns=["swedb_nii"])

# Full variable set for PCA
full_set_vars = list(X.columns)

# Portfolios for qualitative strategies
portfolios = {
    "PCA": full_set_vars,
    "Portfolio_A": [
        "Policy rate", "STIBOR3M", "EURIBOR3M",
        "swedb_customer_loans_bnsek", "swedb_customer_deposits_bnsek",
        "swedb_loan_deposit_ratio", "swe_gdp", "quarterly_inflation"
    ],
    "Portfolio_B": [
        "ECB Deposit rate", "EURIBOR3M", "USTREASURY3M",
        "sek5y_irswap_ask", "swedb_customer_loans_bnsek",
        "swedb_customer_deposits_bnsek", "swe_nat_debt", "unempl_rate"
    ],
    "Portfolio_C": [
        "Policy rate", "STIBOR_6M_3M_Spread", "EURIBOR_6M_3M_Spread",
        "swedb_customer_loans_bnsek", "swedb_customer_deposits_bnsek",
        "swedb_loan_deposit_ratio", "re_priceindex", "kpi_fixed_values"
    ]
}

# Model references
models = {
    "ARIMAX": run_arimax,
    "VAR": run_var,
    "RandomForest": run_rf,
    "NeuralNet": run_nn
}

# Rolling window sizes to test
window_sizes = [4, 6, 8]

# Run experiments
results = []

for strategy_name, variable_list in portfolios.items():
    print(f"\n>> Processing strategy: {strategy_name}")
    X_subset = X[variable_list]

    # Differencing
    y_diff = y.diff().dropna()
    X_diff = X_subset.diff().dropna()
    X_diff = X_diff.loc[y_diff.index]
    y_diff = y_diff.loc[X_diff.index]

    # Scale (and optionally PCA)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_diff), index=X_diff.index)

    if strategy_name == "PCA":
        pca = PCA(n_components=0.95)
        X_transformed = pd.DataFrame(pca.fit_transform(X_scaled), index=X_scaled.index)
    else:
        X_transformed = X_scaled.copy()

    for window in window_sizes:
        for model_name, model_func in models.items():
            print(f" - Running {model_name} | Window: {window}")
            try:
                result = model_func(y_diff, X_transformed, window=window)
                result.update({
                    "Model": model_name,
                    "Strategy": strategy_name,
                    "Window": window
                })
                results.append(result)
            except Exception as e:
                print(f"   ⚠️ {model_name} failed on {strategy_name} (window {window}): {e}")

# Save results
results_df = pd.DataFrame(results)
results_df.to_excel("model_comparison_results.xlsx", index=False)
print("All experiments completed. Results saved to model_comparison_results.xlsx")






