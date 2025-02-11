import pandas as pd
import numpy as np
import matplotlib
import matplotlib as plt

matplotlib.use('TkAgg')  # Switch backend if needed
import os
import pandas as pd
import numpy as np

# 1. Specify your three CSV files
csv_files = [
    "/home/jamalids/Documents/corrolation/fastlz.csv",
    "/home/jamalids/Documents/corrolation/lz4.csv",
    "/home/jamalids/Documents/corrolation/zlib.csv"
]

# 2. Read them and add a column "compressmethod" to identify each
df_list = []
for csv_file in csv_files:
    # Extract method name from filename, e.g. "method1" from "method1.csv"
    method_name = os.path.splitext(os.path.basename(csv_file))[0]

    # Read the CSV into a DataFrame
    df_temp = pd.read_csv(csv_file)

    # Add a new column indicating the compression method
    df_temp["compressmethod"] = method_name

    # Append to list
    df_list.append(df_temp)

# 3. Concatenate them into one DataFrame
df_combined = pd.concat(df_list, ignore_index=True)

# 4. Clean up infinite or missing values in the columns we care about.
#    Suppose we want to measure correlation of EntropyRatio vs. CompRatioRatio
df_combined.replace([np.inf, -np.inf], np.nan, inplace=True)
df_combined.dropna(subset=["EntropyRatio", "CompRatioRatio"], inplace=True)

# 5. Measure correlation by compression method
methods = df_combined["compressmethod"].unique()
print("Correlation by method:")
for method in methods:
    df_m = df_combined[df_combined["compressmethod"] == method]
    # Pearson correlation
    pearson_r = df_m["EntropyRatio"].corr(df_m["CompRatioRatio"], method="pearson")
    # Spearman correlation
    spearman_r = df_m["EntropyRatio"].corr(df_m["CompRatioRatio"], method="spearman")

    print(f"- {method} -> Pearson: {pearson_r:.4f}, Spearman: {spearman_r:.4f}")

# 6. (Optional) Also measure correlation across *all* rows (i.e. ignoring the distinction of method)
pearson_all = df_combined["EntropyRatio"].corr(df_combined["CompRatioRatio"], method="pearson")
spearman_all = df_combined["EntropyRatio"].corr(df_combined["CompRatioRatio"], method="spearman")

print("\nCorrelation across ALL methods combined:")
print(f"Pearson:  {pearson_all:.4f}")
print(f"Spearman: {spearman_all:.4f}")

# 7. (Optional) Save df_combined to a new CSV with method labels included
df_combined.to_csv("combined_methods.csv", index=False)
print("\nSaved combined CSV to: combined_methods.csv")
