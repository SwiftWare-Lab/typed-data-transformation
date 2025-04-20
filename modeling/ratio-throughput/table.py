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
    "phone_gyro": "TS",
    "wesad_chest": "TS",
    "jane_street": "TS",
    "nyc_taxi2015": "TS",
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
    "tpch_lineitem": "DB",
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
df = pd.read_csv("/home/jamalids/Documents/max_compression_throughput_pairs.csv")

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
pivot_df.to_csv("/home/jamalids/Documents/combine-com-through/merge.csv")
# Optionally, convert the pivoted DataFrame to a LaTeX table for Overleaf:
latex_table = pivot_df.to_latex(index=False,
                                column_format="|" + "l|" * len(pivot_df.columns),
                                escape=False)
print(latex_table)
import os
import os
import pandas as pd
import re

# # Define the mapping from dataset name to Domain (from your table)
# domain_mapping = {
#     "msg_bt": "HPC",
#     "num_brain": "HPC",
#     "num_control": "HPC",
#     "rsim": "HPC",
#     "astro_mhd": "HPC",
#     "astro_pt": "HPC",
#     "miranda3d": "HPC",
#     "turbulence": "HPC",
#     "wave": "HPC",
#     "hurricane": "HPC",
#     "citytemp": "TS",
#     "ts_gas": "TS",
#     "phone_gyro": "TS",
#     "wesad_chest": "TS",
#     "jane_street": "TS",
#     "nyc_taxi2015": "TS",
#     "gas_price": "TS",
#     "solar_wind": "TS",
#     "acs_wht": "OBS",
#     "hdr_night": "OBS",
#     "hdr_palermo": "OBS",
#     "hst_wfc3_uvis": "OBS",
#     "hst_wfc3_ir": "OBS",
#     "spitzer_irac": "OBS",
#     "g24_78_usb": "OBS",
#     "jw_mirimage": "OBS",
#     "tpcH_order": "DB",
#     "tpcxbb_store": "DB",
#     "tpcxbb_web": "DB",
#     "tpch_lineitem": "DB",
#     "tpcds_catalog": "DB",
#     "tpcds_store": "DB",
#     "tpch_order": "DB",
#     "tpcds_web": "DB",
# }
#
# # Function to remove trailing numbers optionally preceded by an "f" or "F"
# def remove_trailing_pattern(s):
#     return re.sub(r'(_?[fF]\d*)$', '', s)
#
# # ---------------- Main CSV Processing ----------------
# main_csv_path = "/home/jamalids/Documents/combine-com-through/updated_combine_all.csv"
# df_main = pd.read_csv(main_csv_path)
# df_main['NormalizedDatasetName'] = df_main['DatasetName'].apply(remove_trailing_pattern)
# df_main['Domain'] = df_main['NormalizedDatasetName'].map(domain_mapping)
#
# # Create a combined column "runytpe_compression_tool"
# if "RunType" in df_main.columns:
#     df_main["runytpe_compression_tool"] = df_main["RunType"].astype(str) + "_" + df_main["compression_tool"].astype(str)
# else:
#     df_main["runytpe_compression_tool"] = df_main["compression_tool"]
#
# unique_tools = sorted(df_main["runytpe_compression_tool"].unique())
# print("Unique combined runytpe+compression_tool:", unique_tools)
#
# # Pivot the main CSV so that:
# #   Index: ["NormalizedDatasetName", "Domain"]
# #   Columns: combined "runytpe_compression_tool" values
# #   Values: "CompressionRatio"
# pivot_main = df_main.pivot_table(index=["NormalizedDatasetName", "Domain"],
#                                  columns="runytpe_compression_tool",
#                                  values="CompressionRatio",
#                                  aggfunc="first")
# pivot_main = pivot_main.reindex(columns=unique_tools).fillna("")
# pivot_main.reset_index(inplace=True)
#
# # ---------------- Second CSV Processing (Not Pivoted) ----------------
# # Read the second CSV (byte-bitshuffle results) without pivoting.
# byte_csv_path = "/home/jamalids/Documents/combine-com-through/byte-bitshuffle.csv"
# df_second = pd.read_csv(byte_csv_path)
# df_second['NormalizedDatasetName'] = df_second['DatasetName'].apply(remove_trailing_pattern)
# df_second['Domain'] = df_second['NormalizedDatasetName'].map(domain_mapping)
# # (We assume df_second has columns such as "compression_tool" and "CompressionRatio".)
#
# # ---------------- Merge (Outer Join on NormalizedDatasetName and Domain) ----------------
# merged = pd.merge(pivot_main, df_second, on=["NormalizedDatasetName", "Domain"], how="outer", suffixes=("", "_byte"))
#
# # Optionally drop duplicated key columns if present (e.g. "Domain_byte")
# if "Domain_byte" in merged.columns:
#     merged.drop(columns=["Domain_byte"], inplace=True)
#
# # ---------------- Column Reordering by Compression Tool ----------------
# # We want columns grouped by tool such as (e.g.) TDT_lz4, standard_lz4, bitshuffle_lz4_ratio, raw_lz4_ratio,...
#
# # Define the key columns that should appear first.
# key_cols = ["NormalizedDatasetName", "Domain"]
#
# # Get the remaining columns.
# other_cols = [col for col in merged.columns if col not in key_cols]
#
# def grouping_key(col):
#     """Return a tuple (group, col) for sorting columns.
#        The group is determined by checking if 'lz4' or 'zstd' is in the column name (case-insensitive)."""
#     clower = col.lower()
#     if "lz4" in clower:
#         group = "lz4"
#     elif "zstd" in clower:
#         group = "zstd"
#     else:
#         group = "other"
#     return (group, col)
#
# sorted_other = sorted(other_cols, key=grouping_key)
# final_columns = key_cols + sorted_other
# merged = merged[final_columns]
#
# # ---------------- Output ----------------
# print("Merged Results:")
# print(merged)
#
# # Save the merged DataFrame to CSV.
# merged_csv_path = "/home/jamalids/Documents/combine-com-through/merged_compression_stats.csv"
# merged.to_csv(merged_csv_path, index=False)
# print(f"Merged CSV saved to: {merged_csv_path}")
#
# # Convert the merged DataFrame to a LaTeX table.
# latex_table = merged.to_latex(index=False,
#                               column_format="|" + "l|" * merged.shape[1],
#                               escape=False)
# print("LaTeX Table:")
# print(latex_table)
