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
portfolio_dict = {
    "PCA": full_set_vars,
    "Portfolio 1 - Balanced Macro Focus": [
        "ECB Deposit rate", "EURIBOR3M",
        "swedb_customer_deposits_bnsek", "swedb_loan_deposit_ratio", "swedb_customer_loans_bnsek",
        "swe_gdp", "quarterly_inflation", "net_trade"
    ],
    "Portfolio 2 – Core Inflation Watch": [
        "Policy rate", "EURIBOR3M",
        "swedb_customer_deposits_bnsek", "swedb_customer_loans_bnsek", "swedb_loan_deposit_ratio",
        "re_priceindex", "swe_nat_debt", "quarterly_inflation"
    ],
    "Portfolio 3 – Economy": [
        "Policy rate", "EURIBOR3M",
        "swedb_loan_deposit_ratio", "swedb_customer_deposits_bnsek",
        "swe_gdp", "disp_income_msek", "gdp_growth_pct", "quarterly_inflation"
    ],
    "Portfolio 4 – Trade-Sensitive Model": [
        "Policy rate", "EURIBOR3M",
        "swedb_customer_loans_bnsek", "swedb_customer_deposits_bnsek",
        "unempl_rate", "re_priceindex", "export_msek", "import_msek"
    ],
    "Portfolio 5 – Inflation Stress Lens": [
        "ECB Deposit rate", "EURIBOR3M",
        "swedb_customer_loans_bnsek", "swedb_loan_deposit_ratio", "swedb_customer_deposits_bnsek",
        "swe_gdp", "quarterly_inflation", "unempl_rate"
    ],
    "Portfolio 6 – Long-Duration Exposure Placeholder": [
        "ECB Deposit rate", "sek10y_irswap_ask",
        "swedb_customer_loans_bnsek", "swedb_loan_deposit_ratio",
        "gdp_growth_pct", "import_msek", "disp_income_msek", "kpi_fixed_values"
    ],
    "Portfolio 7 – Domestic Activity Monitor": [
        "Policy rate", "sek5y_irswap_ask",
        "swedb_customer_deposits_bnsek", "swedb_loan_deposit_ratio",
        "kpi_fixed_values", "export_msek", "import_msek", "unempl_rate"
    ],
    "Portfolio 8 – Fiscal and Trade Watch": [
        "Policy rate", "sek10y_irswap_ask",
        "swedb_customer_loans_bnsek", "swedb_loan_deposit_ratio", "swedb_customer_deposits_bnsek",
        "swe_nat_debt", "re_priceindex", "net_trade"
    ],
    "Portfolio 9 – ": [
        "ECB Deposit rate", "sek1y_irswap_ask",
        "swedb_customer_loans_bnsek", "swedb_loan_deposit_ratio", "swedb_customer_deposits_bnsek",
        "disp_income_msek", "unempl_rate", "gdp_growth_pct"
    ],
    "Portfolio 10 – GDP Sensitivity Model": [
        "Policy rate", "sek5y_irswap_ask",
        "swedb_loan_deposit_ratio", "swedb_customer_loans_bnsek",
        "kpi_fixed_values", "swe_nat_debt", "swe_gdp", "net_trade"
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

for strategy_name, variable_list in portfolio_dict.items():
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






