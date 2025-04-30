import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# VLDB-style formatting
matplotlib.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Latin Modern Roman", "Times", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# 1) Read CSV
df = pd.read_csv("/mnt/c/Users/jamalids/Downloads/figs/results/combine-all-config_zstd.csv")

# Relevant columns
row_ratio_col = "decomposed row-ordered zstd compression ratio"
col_ratio_col = "decomposed zstd compression ratio"

# 2) Round the ratio columns to 2 decimals
df[row_ratio_col] = df[row_ratio_col].round(2)
df[col_ratio_col] = df[col_ratio_col].round(2)

# Grouping by dataset
grouped = df.groupby("dataset name", as_index=False)
best_results = []

for dataset, group in grouped:
    max_row_ratio = group[row_ratio_col].max()
    max_col_ratio = group[col_ratio_col].max()

    same_best = group[
        (group[row_ratio_col] == max_row_ratio) &
        (group[col_ratio_col] == max_col_ratio)
    ]

    if not same_best.empty:
        best_both = same_best.iloc[0]
        best_results.append({
            "Dataset": dataset,
            "Best_RowOrder_Ratio": max_row_ratio,
            "Best_ColOrder_Ratio": max_col_ratio,
            "Config_For_Both": best_both["decomposition"],
            "RowOrder_Config": None,
            "ColOrder_Config": None
        })
    else:
        row_ok = group.dropna(subset=[row_ratio_col])
        if not row_ok.empty:
            row_best = row_ok.loc[row_ok[row_ratio_col].idxmax()]
            best_row_ratio = row_best[row_ratio_col]
            best_row_config = row_best["decomposition"]
        else:
            best_row_ratio = None
            best_row_config = None

        col_ok = group.dropna(subset=[col_ratio_col])
        if not col_ok.empty:
            col_best = col_ok.loc[col_ok[col_ratio_col].idxmax()]
            best_col_ratio = col_best[col_ratio_col]
            best_col_config = col_best["decomposition"]
        else:
            best_col_ratio = None
            best_col_config = None

        best_results.append({
            "Dataset": dataset,
            "Best_RowOrder_Ratio": best_row_ratio,
            "RowOrder_Config": best_row_config,
            "Best_ColOrder_Ratio": best_col_ratio,
            "ColOrder_Config": best_col_config,
            "Config_For_Both": None
        })

best_df = pd.DataFrame(best_results)
best_df_sorted = best_df.sort_values("Dataset")

# 3) Merge with DatasetID
mapping = pd.read_csv("/mnt/c/Users/jamalids/Downloads/figs/DatasetIdMapping.csv")
best_df_sorted = best_df_sorted.merge(mapping[['DatasetName', 'DatasetID']], left_on='Dataset', right_on='DatasetName', how='inner')

# 4) Plot (only one plot, no subplot)
plt.figure(figsize=(6.2, 2.5))

xvals = range(len(best_df_sorted))
plt.plot(xvals, best_df_sorted["Best_RowOrder_Ratio"], marker='o', label="Best RowOrder ratio")
plt.plot(xvals, best_df_sorted["Best_ColOrder_Ratio"], marker='s', label="Best ColOrder ratio")

plt.xlabel("Dataset ID")
plt.ylabel("Compression Ratio")
#plt.title("Best Decomposed Ratios (Row vs Col) Based on 2-decimal rounding")
plt.xticks(xvals, best_df_sorted["DatasetID"], rotation=90, ha='right')
plt.legend()
plt.tight_layout()

plt.savefig("/mnt/c/Users/jamalids/Downloads/zstd-col-row.pdf")
plt.close()
