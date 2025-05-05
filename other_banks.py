import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.decomposition import PCA
import warnings
from sklearn.exceptions import ConvergenceWarning
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Import model wrappers
from models.nn_model import run_nn
from models.rf_model import run_rf

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# === Load Data ===
df = pd.read_excel("quarterly_averages.xlsx")
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# === Convert Nordea NII from EUR to SEK ===
if "nordea_nii" in df.columns and "eur_sek" in df.columns:
    df["nordea_nii"] = df["nordea_nii"] * df["eur_sek"]
    print("Converted Nordea NII from EUR to SEK using quarterly average exchange rates.")
else:
    raise ValueError("Missing 'nordea_nii' or 'eur_sek' column in the Excel file.")

# === Define Portfolio Variables ===
portfolio_dict = {
   "Portfolio 1 - Balanced Macro Focus": [
        "ECB Deposit rate", "EURIBOR3M",
        "swedb_customer_deposits_bnsek", "swedb_loan_deposit_ratio", "swedb_customer_loans_bnsek",
        "swe_gdp", "quarterly_inflation", "net_trade"
    ],
    "Portfolio 2 - Core Inflation Watch": [
        "Policy rate", "EURIBOR3M",
        "swedb_customer_deposits_bnsek", "swedb_customer_loans_bnsek", "swedb_loan_deposit_ratio",
        "re_priceindex", "swe_nat_debt", "quarterly_inflation"
    ],
    "Portfolio 3 - Economy": [
        "Policy rate", "EURIBOR3M",
        "swedb_loan_deposit_ratio", "swedb_customer_deposits_bnsek",
        "swe_gdp", "disp_income_msek", "gdp_growth_pct", "quarterly_inflation"
    ],
    "Portfolio 4 - Trade-Sensitive Model": [
        "Policy rate", "EURIBOR3M",
        "swedb_customer_loans_bnsek", "swedb_customer_deposits_bnsek",
        "unempl_rate", "re_priceindex", "export_msek", "import_msek"
    ],
    "Portfolio 10 - GDP Sensitivity Model": [
        "Policy rate", "sek5y_irswap_ask",
        "swedb_loan_deposit_ratio", "swedb_customer_loans_bnsek",
        "kpi_fixed_values", "swe_nat_debt", "swe_gdp", "net_trade"
    ]
}

# === Define Top 5 Forecasting Scenarios (from report Table 6) ===
top5 = [
    {"Strategy": "Portfolio 1 - Balanced Macro Focus", "Window": 4, "Model": "Neural Network"},
    {"Strategy": "Portfolio 2 - Core Inflation Watch", "Window": 8, "Model": "Neural Network"},
    {"Strategy": "Portfolio 10 - GDP Sensitivity Model", "Window": 4, "Model": "Random Forest"},
    {"Strategy": "Portfolio 4 - Trade-Sensitive Model", "Window": 4, "Model": "Neural Network"},
    {"Strategy": "Portfolio 3 - Economy", "Window": 8, "Model": "Neural Network"}
]

# === Define Target Banks ===
banks = {
    "Handelsbanken": "handelsb_nii",
    "SEB": "seb_nii",
    "Nordea": "nordea_nii"
}

# === Run Evaluation & Collect Results for Combined Plots ===
all_results = []
plot_data = {}  # structure: plot_data[bank][model][strategy] = (time_index, actual_dNII, forecast_dNII)

for bank_name, target_col in banks.items():
    for scenario in top5:
        strategy = scenario['Strategy']
        window = int(scenario['Window'])
        model_name = scenario['Model']
        predictors = portfolio_dict[strategy]

        y = df[target_col].diff().dropna()
        X = df[predictors].diff().dropna()
        X, y = X.loc[y.index], y.loc[X.index]  # Align

        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)

        if model_name == "Neural Network":
            result = run_nn(y, X_scaled, window)
        elif model_name == "Random Forest":
            result = run_rf(y, X_scaled, window)
        else:
            raise ValueError(f"Unsupported model type: {model_name}")

        time_index = y.index[window + 1:]
        actual_diff = result['y_true']
        forecast_diff = result['y_pred']

        # Store for grouped plotting
        if bank_name not in plot_data:
            plot_data[bank_name] = {}
        if model_name not in plot_data[bank_name]:
            plot_data[bank_name][model_name] = {}
        plot_data[bank_name][model_name][strategy] = (time_index, actual_diff, forecast_diff)

        all_results.append({
            "Bank": bank_name,
            "Strategy": strategy,
            "Window": window,
            "Model": model_name,
            "MAE": result["MAE"],
            "RMSE": result["RMSE"]
        })

# === Plot per (Bank × Model), with all top strategies ===
for bank_name, models in plot_data.items():
    for model_name, strategies in models.items():
        plt.figure(figsize=(12, 5))

        for strategy, (time_index, actual, forecast) in strategies.items():
            plt.plot(time_index, actual, label=f'{strategy} — Actual')
            plt.plot(time_index, forecast, '--', label=f'{strategy} — Forecast')

        plt.title(f'{bank_name} — {model_name}')
        plt.xlabel("Year")
        plt.ylabel("Δ Net Interest Income (ΔNII)")
        plt.legend(fontsize='small', loc='upper left')
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.gca().xaxis.set_major_locator(mdates.YearLocator())
        plt.tight_layout()
        plt.show()


# === Save or Display ===
results_df = pd.DataFrame(all_results)
results_df.to_excel("peer_bank_top5_comparison.xlsx", index=False)
print("Comparison saved to peer_bank_top5_comparison.xlsx")

