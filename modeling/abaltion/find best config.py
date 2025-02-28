import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# 1. Read CSV file
df = pd.read_csv('/home/jamalids/Documents/frame/new/big-data-compression/modeling/clustering/logs-zstd/hdr_night_f32_decomposition_stats.csv')


# If there's NO date column, skip the following lines:
# df.set_index('Date', inplace=True)
# df.sort_index(inplace=True)

# Check columns to see if there is any date/time column at all
print("Columns in CSV:", df.columns)

# 2. Identify columns that contain "ratio" at the end of the name
ratio_columns = [col for col in df.columns if col.lower().endswith("ratio")]

print("Columns that end with 'ratio':", ratio_columns)

# 3. Find the "best" ratio and the corresponding row/column
best_ratio_value = None
best_ratio_column = None
best_ratio_index = None

for col in ratio_columns:
    max_in_col = df[col].max(skipna=True)
    if (best_ratio_value is None) or (max_in_col > best_ratio_value):
        best_ratio_value = max_in_col
        best_ratio_column = col
        best_ratio_index = df[col].idxmax(skipna=True)

print("\nBest ratio found in column:", best_ratio_column)
print("Best ratio value:", best_ratio_value)
print("Row index of best ratio:", best_ratio_index)

# (Optional) Inspect the entire row
best_ratio_row = df.loc[best_ratio_index]
print("\nData at row of best ratio:\n", best_ratio_row)

# 4. Perform time series decomposition on the "best ratio" column (only if you have valid time series data!)
# If you do NOT have a date column, or it's not truly time-series data, you should skip this:
# if best_ratio_column is not None:
#     decomposition = sm.tsa.seasonal_decompose(
#         df[best_ratio_column].dropna(),
#         model='additive',
#         period=12
#     )
#
#     fig = decomposition.plot()
#     fig.set_size_inches(10, 8)
#     plt.suptitle(f"Seasonal Decomposition of {best_ratio_column}", fontsize=14)
#     plt.show()
