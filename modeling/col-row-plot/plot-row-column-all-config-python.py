import pandas as pd
import matplotlib.pyplot as plt

# 1) Read CSV
df = pd.read_csv(
    "/home/jamalids/Documents/frame/new2/big-data-compression/modeling/col-row-plot/combine-all-config.csv")

# Relevant columns
row_ratio_col = "decomposed row-ordered zstd compression ratio"
col_ratio_col = "decomposed zstd compression ratio"

# 2) Round the ratio columns to 3 decimals immediately
df[row_ratio_col] = df[row_ratio_col].round(2)
df[col_ratio_col] = df[col_ratio_col].round(2)

# Now all comparisons for "best" are done on these 3-decimal values
grouped = df.groupby("dataset name", as_index=False)
best_results = []

for dataset, group in grouped:
    # Find the "max" (based on already-rounded values)
    max_row_ratio = group[row_ratio_col].max()
    max_col_ratio = group[col_ratio_col].max()

    # Check if there's one row that is best in both row & col
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
        # Otherwise pick them separately
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

# Print table of results
print("\nBest Decomposed Ratios and Configs for Each Dataset (3-decimal based selection):")
print(best_df_sorted)

# 3) Plot with two subplots
fig, (ax1, ax2) = plt.subplots(
    nrows=2,
    figsize=(10, 8),
    gridspec_kw={"height_ratios": [3, 1]}
)

# Top: line plot
xvals = range(len(best_df_sorted))
ax1.plot(
    xvals,
    best_df_sorted["Best_RowOrder_Ratio"],
    marker='o',
    label="Best RowOrder ratio"
)
ax1.plot(
    xvals,
    best_df_sorted["Best_ColOrder_Ratio"],
    marker='o',
    label="Best ColOrder ratio"
)
ax1.set_xlabel("Dataset")
ax1.set_ylabel("Compression Ratio")
ax1.set_title("Best Decomposed Ratios (Row vs Col) Based on 3-decimals")
ax1.set_xticks(xvals)
ax1.set_xticklabels(best_df_sorted["Dataset"], rotation=45, ha='right')
ax1.legend()

# Bottom: table
ax2.axis("off")
ax2.axis("tight")

table_data = best_df_sorted.values
column_labels = list(best_df_sorted.columns)

ax2.table(
    cellText=table_data,
    colLabels=column_labels,
    loc="center"
)

plt.tight_layout()
plt.savefig("/home/jamalids/Documents/frame/new2/big-data-compression/modeling/col-row-plot/zstd-col-row.png")

