import pandas as pd

# 1) Read each CSV
df_entropy = pd.read_csv("/home/jamalids/Documents/results-corrolation/k/all_entropy_summaries.csv")
df_comp = pd.read_csv("//home/jamalids/Documents/results-corrolation/k/entropy.csv")
df_inf = pd.read_csv("/home/jamalids/Documents/results-corrolation/k/combined_zstd.csv")

# 2) Rename columns to ensure consistency
df_entropy.rename(columns={"dataset": "dataset_name"}, inplace=True)
df_comp.rename(columns={"dataset name": "dataset_name"}, inplace=True)
df_inf.rename(columns={"dataset name": "dataset_name"}, inplace=True)

# 3) Print column names after renaming
print("After renaming -> df_entropy:", df_entropy.columns.tolist())
print("After renaming -> df_comp:", df_comp.columns.tolist())
print("After renaming -> df_inf:", df_inf.columns.tolist())

# 4) Ensure 'dataset_name' exists before merging
missing_columns = []
if "dataset_name" not in df_entropy.columns:
    missing_columns.append("df_entropy")
if "dataset_name" not in df_comp.columns:
    missing_columns.append("df_comp")
if "dataset_name" not in df_inf.columns:
    missing_columns.append("df_inf")

if missing_columns:
    print(f"Error: 'dataset_name' column missing in {missing_columns}")
    exit(1)  # Stop execution

# 5) Merge in TWO STEPS (df_entropy + df_comp, then merge the result with df_inf)
merged_df = pd.merge(df_entropy, df_comp, on=["dataset_name"], how="inner")
final_merged_df = pd.merge(merged_df, df_inf, on=["dataset_name"], how="inner")

# 6) Write the final merged results
final_merged_df.to_csv("/home/jamalids/Documents/merged22.csv", index=False)

print("Merged CSV created:", "/home/jamalids/Documents/new-merged.csv")
