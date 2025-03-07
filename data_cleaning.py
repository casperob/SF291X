import pandas as pd

riksbank_df = pd.read_excel("mex_data.xlsx", sheet_name= "Riksbanken", index_col=0)
ecb_df = pd.read_excel("mex_data.xlsx", sheet_name= "ECB", index_col=0)
market_rates_df = pd.read_excel("mex_data.xlsx", sheet_name= "Daily Adjusted", index_col=0)
monthly_df = pd.read_excel("mex_data.xlsx", sheet_name= "Monthly")
### Måste lägga in alla ny räntor också


monthly_df["Date"] = pd.to_datetime(monthly_df["Date"], format="%YM%m")
monthly_df.set_index("Date", inplace=True)

#måste ändra här, vissa ska va summor, inte snitt pga konstigt annars
swe_macro = monthly_df[["swe_nat_debt", "import_msek", "export_msek", "net_trade"]].resample("QE").sum()

cpi_quarterly_avg = monthly_df["kpi_fixed_values"].resample("QE").mean()
quarterly_inflation = (
    cpi_quarterly_avg / cpi_quarterly_avg.shift(4) - 1
)
base_cpi = cpi_quarterly_avg.iloc[0]

cpi_index = cpi_quarterly_avg / base_cpi

swe_macro = swe_macro.div(cpi_index, axis=0)

macro_df = pd.concat([swe_macro, quarterly_inflation,], axis=1)

## Add more data frames 
rate_dfs = {
    "riksbank": riksbank_df,
    "ecb": ecb_df,
    "market_rates": market_rates_df
}

"""
riksbank_df["polrate_return"] = riksbank_df["Policy rate"].pct_change()
riksbank_volatility = riksbank_df["polrate_return"].resample("QE").std().rename("polrate_vol")
riksbank_q_df = riksbank_df.resample("QE").mean()
ecb_df["deprate_return"] = ecb_df["ECB Deposit rate"].pct_change()
ecb_volatility = ecb_df["deprate_return"].resample("QE").std().rename("ecbrate_vol")
ecb_q_df = ecb_df.resample("QE").mean()
"""

def process_rates(rates, moving_avg_window):
    quarterly_results = []  # Store results for merging

    for df_name, df in rates.items():
        df = df.copy()  # Work on a copy to avoid modifying the original data

        # Ensure DateTime index
        df.index = pd.to_datetime(df.index)

        for col in df.columns:  # Iterate over each rate column in the DataFrame
            df_single = df[[col]].dropna()  # Work with one rate at a time

            # Compute quarterly average
            df_q = df_single.resample("QE").mean().rename(columns={col: f"{col}_Avg"})

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

#quarterly_df = process_rates(rate_dfs, 50)


#print(quarterly_df)

#quarterly_df.to_excel("quarterly_rates_aggregates.xlsx")
