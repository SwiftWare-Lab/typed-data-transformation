import os
import pandas as pd
import re

# Define the mapping from dataset name to Domain (from your table)
domain_mapping = {
    "msg_bt": "HPC",
    "num_brain": "HPC",
    "num_control": "HPC",
    "rsim": "HPC",
    "astro_mhd": "HPC",
    "astro_pt": "HPC",
    "miranda3d": "HPC",
    "turbulence": "HPC",
    "wave": "HPC",
    "hurricane": "HPC",
    "citytemp": "TS",
    "ts_gas": "TS",
    "wesad_chest": "TS",
    "jane_street": "TS",
    "gas_price": "TS",
    "solar_wind": "TS",
    "acs_wht": "OBS",
    "hdr_night": "OBS",
    "hdr_palermo": "OBS",
    "hst_wfc3_uvis": "OBS",
    "hst_wfc3_ir": "OBS",
    "spitzer_irac": "OBS",
    "g24_78_usb": "OBS",
    "jw_mirimage": "OBS",
    "tpcH_order": "DB",
    "tpcxbb_store": "DB",
    "tpcxbb_web": "DB",
    "tpcds_catalog": "DB",
    "tpcds_store": "DB",
    "tpch_order": "DB",
    "tpcds_web": "DB",
}

# Function to remove trailing numbers from a string
# Function to remove trailing numbers optionally preceded by an "f" or "F"
def remove_trailing_pattern(s):
    return re.sub(r'(_?[fF]\d*)$', '', s)

# Read the CSV file
df = pd.read_csv("/mnt/c/Users/jamalids/Downloads/combine.csv")

# Create a normalized dataset name by removing any trailing numbers
df['NormalizedDatasetName'] = df['DatasetName'].apply(remove_trailing_pattern)

# Map the normalized dataset name to a Domain value based on our dictionary
df['Domain'] = df['NormalizedDatasetName'].map(domain_mapping)

# --- New: Create a combined column "runytpe_compression_tool" ---
# If the column "runytpe" exists, combine it with "compression_tool"; otherwise, use "compression_tool" only.
if "RunType" in df.columns:
    df["runytpe_compression_tool"] = df["RunType"].astype(str) + "_" + df["compression_tool"].astype(str)
else:
    df["runytpe_compression_tool"] = df["compression_tool"]

# First, get all unique values from the combined column.
unique_tools = sorted(df["runytpe_compression_tool"].unique())
print("Unique combined runytpe+compression_tool:", unique_tools)

# Pivot the DataFrame so that:
# - The index is ["NormalizedDatasetName", "Domain"]
# - The columns come from the combined column "runytpe_compression_tool"
# - The values come from "CompressionRatio"
pivot_df = df.pivot_table(index=["NormalizedDatasetName", "Domain"],
                          columns="runytpe_compression_tool",
                          values="CompressionRatio",
                          aggfunc="first")

# Reindex the columns to include all unique values in the desired order
pivot_df = pivot_df.reindex(columns=unique_tools)
pivot_df = pivot_df.fillna("")  # fill missing values

# Reset the index so that "NormalizedDatasetName" and "Domain" become regular columns
pivot_df.reset_index(inplace=True)

# (Optional) Reorder the columns if you want the combined columns to come first.
final_columns = unique_tools + ["Domain", "NormalizedDatasetName"]
final_columns = [col for col in final_columns if col in pivot_df.columns]
pivot_df = pivot_df[final_columns]

# Print the resulting DataFrame to verify
print(pivot_df)
pivot_df.to_csv("/mnt/c/Users/jamalids/Downloads/merge.csv")
# Optionally, convert the pivoted DataFrame to a LaTeX table for Overleaf:
latex_table = pivot_df.to_latex(index=False,
                                column_format="|" + "l|" * len(pivot_df.columns),
                                escape=False)
print(latex_table)
