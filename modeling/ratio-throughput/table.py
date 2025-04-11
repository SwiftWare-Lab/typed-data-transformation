import pandas as pd


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
df = pd.read_csv("/home/jamalids/Documents/combine-com-through/combine-all.csv")

# Create a normalized dataset name by removing any trailing numbers
df['NormalizedDatasetName'] = df['DatasetName'].apply( remove_trailing_pattern)

# Map the normalized dataset name to a Domain value based on our dictionary
df['Domain'] = df['NormalizedDatasetName'].map(domain_mapping)

# (Optional) If you don't need the helper column, you can drop it:
# df.drop(columns=['NormalizedDatasetName'], inplace=True)

# Save the updated DataFrame to a new CSV file
updated_csv_path = "/home/jamalids/Documents/combine-com-through/combine_all1.csv"
df.to_csv(updated_csv_path, index=False)
print(f"Updated CSV saved to: {updated_csv_path}")
############################
df = pd.read_csv("/home/jamalids/Documents/combine-com-through/updated_combine_all.csv")
# First, get all unique compression_tool values
unique_tools = sorted(df["compression_tool"].unique())
print("Unique compression tools:", unique_tools)

# Pivot the DataFrame so that:
# - The index is ["NormalizedDatasetName", "Domain"]
# - The columns come from the unique compression_tool values
# - The values come from "CompressionRatio"
# Use pivot_table with aggfunc='first' if needed (in case of duplicate rows)
pivot_df = df.pivot_table(index=["NormalizedDatasetName", "Domain"],
                          columns="compression_tool",
                          values="CompressionRatio",
                          aggfunc="first")

# Reindex the columns to include all unique compression_tool values in the order we want
pivot_df = pivot_df.reindex(columns=unique_tools)

# Fill any missing values with an empty string (you can change this to a dash "-" if preferred)
pivot_df = pivot_df.fillna("")

# Reset the index so that "NormalizedDatasetName" and "Domain" become regular columns
pivot_df.reset_index(inplace=True)

# (Optional) Reorder the columns if you want the compression tool columns to come first.
# For example, here we place the unique_tools first, then Domain, then NormalizedDatasetName:
final_columns = unique_tools + ["Domain", "NormalizedDatasetName"]
# Ensure we only include columns that exist in pivot_df
final_columns = [col for col in final_columns if col in pivot_df.columns]
pivot_df = pivot_df[final_columns]

# Print the resulting DataFrame to verify
print(pivot_df)

# Optionally, convert the pivoted DataFrame to a LaTeX table for Overleaf:
latex_table = pivot_df.to_latex(index=False,
                                column_format="|" + "l|" * len(pivot_df.columns),
                                escape=False)
print(latex_table)