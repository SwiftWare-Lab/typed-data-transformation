import pandas as pd
import numpy as np
from scipy.stats import gmean

# Load the CSV
file_path = "/mnt/c/Users/jamalids/Downloads/figs/nwrge.csv"
df = pd.read_csv(file_path)

# Filter for TDT and standard
df_filtered = df[df["RunType"].isin(["TDT", "standard"])]

# Pivot to align TDT and standard side-by-side
pivot_df = df_filtered.pivot_table(
    index=["DatasetName", "compression_tool"],
    columns="RunType",
    values="CompressionRatio",
    aggfunc="first"
).dropna()

# Calculate improvement ratio (TDT / Standard)
pivot_df["Ratio"] = (pivot_df["TDT"] / pivot_df["standard"])*100

# Group by compression tool and compute geometric mean of improvements
gmean_summary = pivot_df.groupby("compression_tool")["Ratio"].agg(gmean).reset_index()
gmean_summary.rename(columns={"Ratio": "GMean_TDT_Improvement"}, inplace=True)

# Save to CSV
gmean_summary.to_csv("/mnt/c/Users/jamalids/Downloads/gmean_tdt_improvement1.csv", index=False)
