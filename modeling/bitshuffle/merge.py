import pandas as pd

# Paths to the two CSV files
blosc_csv = "/home/jamalids/Documents/combine-com-through/blosc_zstd_only_results.csv"        # e.g., blosc_zstd_only_results.csv
merged_csv = "/home/jamalids/Documents/combine-com-through/merged_compression_stats.csv"       # e.g., merged_compression_stats.csv

# Read both DataFrames
df_blosc = pd.read_csv(blosc_csv)
df_merged = pd.read_csv(merged_csv)

# If the shared column differs in name, rename one side or specify 'left_on' / 'right_on' in the merge.
# For this example, assume both files have a column named "dataset" that you want to join on.
# If it's named "DatasetName" in one file and "dataset" in the other, adapt accordingly.
# E.g., df_blosc = df_blosc.rename(columns={"dataset": "DatasetName"})

# Perform the merge. You can choose an "inner", "outer", "left", or "right" join:
# - how="inner" will keep only matching rows.
# - how="outer" will keep all rows from both.
df_combined = pd.merge(df_blosc, df_merged, on="DatasetName", how="inner")

# Save the joined result to a new CSV
output_csv = "/home/jamalids/Documents/combine-com-through/joined_compression_stats1.csv"
df_combined.to_csv(output_csv, index=False)
print(f"Joined CSV saved to: {output_csv}")
