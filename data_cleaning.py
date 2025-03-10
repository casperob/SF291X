import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns


riksbank_df = pd.read_excel("mex_data.xlsx", sheet_name= "Riksbanken", index_col=0)
ecb_df = pd.read_excel("mex_data.xlsx", sheet_name= "ECB", index_col=0)
market_rates_df = pd.read_excel("mex_data.xlsx", sheet_name= "Daily Adjusted", index_col=0)
monthly_df = pd.read_excel("mex_data.xlsx", sheet_name= "Monthly")
quarter_df = pd.read_excel("mex_data.xlsx", sheet_name= "Quarterly")
#######
stibor_6m_df = pd.read_excel("mex_data.xlsx", sheet_name="stibor_6m", index_col=0)
ustreasury_6m_df = pd.read_excel("mex_data.xlsx", sheet_name="ustreasury_6m", index_col=0)
usd1y_irswap_df = pd.read_excel("mex_data.xlsx", sheet_name="usd1y_irswap", index_col=0)
usd5y_irswap_df = pd.read_excel("mex_data.xlsx", sheet_name="usd5y_irswap", index_col=0)
usd10y_irswap_df = pd.read_excel("mex_data.xlsx", sheet_name="usd10y_irswap", index_col=0)
sonia_6m_df = pd.read_excel("mex_data.xlsx", sheet_name="sonia_6m", index_col=0)
sek1y_irswap_df = pd.read_excel("mex_data.xlsx", sheet_name="sek1y_irswap", index_col=0)
sek5y_irswap_df = pd.read_excel("mex_data.xlsx", sheet_name="sek5y_irswap", index_col=0)
sek10y_irswap_df = pd.read_excel("mex_data.xlsx", sheet_name="sek10y_irswap", index_col=0)
nibor_3m_df = pd.read_excel("mex_data.xlsx", sheet_name="nibor_3m", index_col=0)
nibor_6m_df = pd.read_excel("mex_data.xlsx", sheet_name="nibor_6m", index_col=0)
euro1y_irswap_df = pd.read_excel("mex_data.xlsx", sheet_name="euro1y_irswap", index_col=0)
euro5y_irswap_df = pd.read_excel("mex_data.xlsx", sheet_name="euro5y_irswap", index_col=0)
euro10y_irswap_df = pd.read_excel("mex_data.xlsx", sheet_name="euro10y_irswap", index_col=0)
euribor_6m_df = pd.read_excel("mex_data.xlsx", sheet_name="euribor_6m", index_col=0)



monthly_df["Date"] = pd.to_datetime(monthly_df["Date"], format="%YM%m")
monthly_df.set_index("Date", inplace=True)

quarter_df["Date"] = pd.PeriodIndex(quarter_df["Date"], freq="Q").to_timestamp(how="end").normalize()
quarter_df.set_index("Date", inplace=True)

#måste ändra här, vissa ska va summor, inte snitt pga konstigt annars
swe_macro_quarterly_sum = monthly_df[["import_msek", "export_msek", "net_trade"]].resample("QE").sum()
swe_macro_quarterly_avg = monthly_df[["swe_nat_debt","kpi_fixed_values"]].resample("QE").mean()
quarterly_inflation = (
    swe_macro_quarterly_avg["kpi_fixed_values"] / swe_macro_quarterly_avg["kpi_fixed_values"].shift(4) - 1
)

swe_macro_quarterly_avg["quarterly_inflation"] = quarterly_inflation
base_cpi = swe_macro_quarterly_avg["kpi_fixed_values"].iloc[0]

cpi_index = swe_macro_quarterly_avg["kpi_fixed_values"] / base_cpi

swe_macro_quarterly_sum = pd.concat([swe_macro_quarterly_sum, quarter_df[["swedb_nii", "swe_gdp", "swedb_customer_loans_bnsek", "swedb_customer_deposits_bnsek"]]], axis=1)
swe_macro_quarterly_sum = swe_macro_quarterly_sum.div(cpi_index, axis=0)

macro_df = pd.concat([quarter_df[["unempl_rate", "swedb_loan_deposit_ratio"]],swe_macro_quarterly_sum, swe_macro_quarterly_avg], axis=1)


## Add more data frames
rate_dfs = {
    "riksbank": riksbank_df,
    "ecb": ecb_df,
    "market_rates": market_rates_df,
    "stibor_6m": stibor_6m_df,
    "ustreasury_6m": ustreasury_6m_df,
    "usd1y_irswap_ask": usd1y_irswap_df[["usd1y_irswap_ask"]],
    "usd5y_irswap_ask": usd5y_irswap_df[["usd5y_irswap_ask"]],
    "usd10y_irswap_ask": usd10y_irswap_df[["usd10y_irswap_ask"]],
    "sek1y_irswap_ask": sek1y_irswap_df[["sek1y_irswap_ask"]],
    "sek5y_irswap_ask": sek5y_irswap_df[["sek5y_irswap_ask"]],
    "sek10y_irswap_ask": sek10y_irswap_df[["sek10y_irswap_ask"]],
    "euro1y_irswap_ask": euro1y_irswap_df[["euro1y_irswap_ask"]],
    "euro5y_irswap_ask": euro5y_irswap_df[["euro5y_irswap_ask"]],
    "euro10y_irswap_ask": euro10y_irswap_df[["euro10y_irswap_ask"]],
    "euribor_6m": euribor_6m_df,
    "sonia_6m": sonia_6m_df
}


def process_rates(rates, moving_avg_window):
    quarterly_results = []  # Store results for merging

    for df_name, df in rates.items():
        df = df.copy()  # Work on a copy to avoid modifying the original data

        # Ensure DateTime index
        df.index = pd.to_datetime(df.index)

        for col in df.columns:  # Iterate over each rate column in the DataFrame
            df_single = df[[col]].dropna()  # Work with one rate at a time

            # Compute quarterly average
            df_q = df_single.resample("QE").mean().rename(columns={col: f"{col}"})

            # Compute daily returns for volatility
            df_single["Daily Return"] = df_single.pct_change()
            df_volatility = df_single["Daily Return"].resample("QE").std().rename(f"{col}_Volatility")

            # Compute moving average (50-day by default)
            df_single[f"{col}_MA{moving_avg_window}"] = df_single[col].rolling(window=moving_avg_window, min_periods=1).mean()
            df_ma_q = df_single[f"{col}_MA{moving_avg_window}"].resample("QE").last().rename(f"{col}_MA{moving_avg_window}")

            # Merge results for this rate
            quarterly_result = pd.concat([df_q, df_volatility, df_ma_q], axis=1)
            quarterly_results.append(quarterly_result)

    # Merge all processed results into a single DataFrame
    final_quarterly_df = pd.concat(quarterly_results, axis=1)

    return final_quarterly_df

quarterly_df = process_rates(rate_dfs, 50)
quarterly_df = pd.concat([quarterly_df, macro_df], axis=1)
quarterly_df.fillna(value=0, inplace=True)
#quarterly_df.to_excel("quarterly_rates_aggregates_v2.xlsx")

predictors = quarterly_df.drop(columns=["swedb_nii"])

scaler = MinMaxScaler()  # Alternatives: MinMaxScaler(), RobustScaler(), StandardScaler

# Normalize predictors
normalized_predictors = pd.DataFrame(scaler.fit_transform(predictors), columns=predictors.columns, index=predictors.index)

# Add back the dependent variable (NII)
normalized_df = pd.concat([normalized_predictors, quarterly_df["swedb_nii"]], axis=1)

#normalized_df.to_excel("normalized_quarterly_aggregates.xlsx")

# Identify columns related to volatility and moving averages
volatility_cols = [col for col in normalized_df.columns if "_Volatility" in col]
moving_avg_cols = [col for col in normalized_df.columns if "_MA" in col]

# Create a new DataFrame with only volatility and moving average columns
volatility_moving_avg_df = quarterly_df[volatility_cols + moving_avg_cols]

# Create a DataFrame with only core values (excluding volatility and moving averages)
core_values_df = quarterly_df.drop(columns=volatility_cols + moving_avg_cols)

correlation_matrix = core_values_df.corr()

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=False, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap of Core Variables")
plt.show()
