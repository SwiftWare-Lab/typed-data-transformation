# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
#
# # 1. Read CSV file
# df = pd.read_csv('/home/jamalids/Documents/all-config-cluster/logs-zstd/combine.csv')
# print("Columns in CSV:", df.columns)
#
# # 2. Identify columns that contain "ratio" at the end of the name
# ratio_columns = [col for col in df.columns if col.lower().endswith("ratio")]
# print("Columns that end with 'ratio':", ratio_columns)
# if not ratio_columns:
#     raise Exception("No ratio columns found in the CSV.")
#
# # 3. Determine the dataset column
# # In your CSV the dataset column is named "dataset name"
# if 'dataset name' in df.columns:
#     dataset_col = 'dataset name'
# elif 'Dataset' in df.columns:
#     dataset_col = 'Dataset'
# elif 'dataset' in df.columns:
#     dataset_col = 'dataset'
# else:
#     raise Exception("No dataset column found in the CSV!")
#
# # 4. Process each dataset group: find best config and compute geometric mean (excluding best)
# agg_results = []  # Aggregated metrics per dataset
# best_configs = []  # Best configuration details per dataset
#
# for dataset, group in df.groupby(dataset_col):
#     best_ratio_value = None
#     best_ratio_column = None
#     best_ratio_index = None
#
#     # Loop over all ratio columns to find the best ratio across all columns for this dataset
#     for col in ratio_columns:
#         max_val = group[col].max(skipna=True)
#         if (best_ratio_value is None) or (max_val > best_ratio_value):
#             best_ratio_value = max_val
#             best_ratio_column = col
#             best_ratio_index = group[col].idxmax(skipna=True)
#
#     print(f"\nDataset: {dataset}")
#     print("Best ratio found in column:", best_ratio_column)
#     print("Best ratio value:", best_ratio_value)
#     print("Row index of best ratio:", best_ratio_index)
#
#     # Retrieve the best configuration row for this dataset
#     best_config_row = group.loc[best_ratio_index]
#
#     # Exclude the best configuration from the group to compute geometric mean
#     non_best = group.drop(index=best_ratio_index)
#     if len(non_best) > 0:
#         # Compute the geometric mean for the selected best_ratio_column (using natural logs)
#         gmean_excluding_best = np.exp(np.log(non_best[best_ratio_column]).mean())
#     else:
#         gmean_excluding_best = np.nan
#
#     # Save aggregated metrics for this dataset
#     agg_results.append({
#         dataset_col: dataset,
#         'best_ratio_column': best_ratio_column,
#         'best_ratio': best_ratio_value,
#         'gmean_excluding_best': gmean_excluding_best
#     })
#
#     # Save the best configuration details
#     best_configs.append(best_config_row)
#
# # 5. Create DataFrames and save the aggregated metrics and best config details to CSV files
# agg_df = pd.DataFrame(agg_results)
# best_config_df = pd.DataFrame(best_configs)
#
# agg_df.to_csv('aggregated_gmean_ratios.csv', index=False)
# best_config_df.to_csv('/home/jamalids/Documents/frame/new/big-data-compression/modeling/abaltion/best_config_details.csv', index=False)
#
# print("\nCSV files saved: 'aggregated_gmean_ratios.csv' and 'best_config_details.csv'.")
#
# # 6. Plot the results: x-axis = dataset, y-axis = ratio values (geometric mean excluding best and best ratio)
# x = np.arange(len(agg_df))
# width = 0.35
#
# fig, ax = plt.subplots(figsize=(10, 6))
# ax.bar(x - width / 2, agg_df['gmean_excluding_best'], width, label='Gmean (Excluding Best)')
# ax.bar(x + width / 2, agg_df['best_ratio'], width, label='Best Config Ratio')
#
# ax.set_xlabel('Dataset')
# ax.set_ylabel('Ratio Value')
# ax.set_title('Gmean (Excluding Best) vs. Best Config Ratio per Dataset')
# ax.set_xticks(x)
# ax.set_xticklabels(agg_df[dataset_col])
# ax.legend()
#
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.savefig('/home/jamalids/Documents/frame/new/big-data-compression/modeling/abaltion/gmean_ratios.png')
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def canonicalize_config(config_val, subtract=0):
    """
    Parse a configuration value (which may be a string representing a tuple of tuples)
    into a canonical string representation with no spaces.

    For each integer encountered, subtract `subtract` from its value.
    For example, if config_val is "((1,2,3)|(4))" and subtract=1, then it will be processed
    into a canonical string "((0,1,2),(3,))".

    This function uses ast.literal_eval to parse the string. If parsing fails,
    it falls back to simply removing spaces.
    """
    try:
        config_str = str(config_val)
        if '|' in config_str:
            groups = config_str.split('|')
            processed_groups = []
            for grp in groups:
                grp = grp.strip()
                if grp.startswith('(') and grp.endswith(')'):
                    grp = grp[1:-1]
                parts = [p.strip() for p in grp.split(',') if p.strip() != '']
                new_parts = []
                for part in parts:
                    try:
                        num = int(part)
                        new_parts.append(num - subtract)
                    except ValueError:
                        new_parts.append(part)
                processed_groups.append(tuple(new_parts))
            return "(" + ",".join(_format_tuple(g) for g in processed_groups) + ")"
        else:
            parsed = ast.literal_eval(config_str)

            def process(item):
                if isinstance(item, int):
                    return item - subtract
                elif isinstance(item, (list, tuple)):
                    return tuple(process(x) for x in item)
                else:
                    return item

            processed = process(parsed)
            if isinstance(processed, tuple):
                return "(" + ",".join(_format_tuple(x) for x in processed) + ")"
            else:
                return str(processed).replace(" ", "")
    except Exception:
        return str(config_val).replace(" ", "")


def _format_tuple(t):
    """Format a tuple of integers (or strings) without any spaces."""
    if isinstance(t, tuple):
        return "(" + ",".join(str(x) for x in t) + ")"
    else:
        return str(t)


# --- Step 1: Process best_results32H.csv ---
best_df = pd.read_csv('/home/jamalids/Documents/frame/new/big-data-compression/modeling/abaltion/best_results32H.csv')
# Check required columns exist
for col in ['config', 'FeatureScenario', 'Metric']:
    if col not in best_df.columns:
        raise Exception(f"Column '{col}' not found in best_results32H.csv")

# Filter to rows with FeatureScenario "Frequency" (case-insensitive) and Metric "DaviesBouldin"
best_df = best_df[(best_df['FeatureScenario'].str.lower() == 'frequency') &
                  (best_df['Metric'] == 'DaviesBouldin')]
if best_df.empty:
    raise Exception("No rows found in best_results32H.csv with FeatureScenario 'Frequency' and Metric 'DaviesBouldin'.")

# Determine dataset column in best_df
for col in ['dataset name', 'Dataset', 'dataset']:
    if col in best_df.columns:
        best_dataset_col = col
        break
else:
    raise Exception("No dataset column found in best_results32H.csv")

# Build mapping: dataset -> canonical best config (with subtract=1)
best_config_mapping = {}
for _, row in best_df.iterrows():
    dataset = row[best_dataset_col]
    config_val = row['config']
    canonical = canonicalize_config(config_val, subtract=1)
    best_config_mapping[dataset] = canonical

print("Best config mapping (canonicalized):")
for ds, cfg in best_config_mapping.items():
    print(f"Dataset '{ds}': {cfg}")

# --- Step 2: Process combine.csv ---
combine_df = pd.read_csv('/home/jamalids/Documents/frame/new/big-data-compression/modeling/abaltion/combine.csv')
print("\nColumns in combine.csv:", combine_df.columns.tolist())

# We'll use the "decomposed zstd compression ratio" for all measurements.
chosen_ratio_column = "decomposed zstd compression ratio"
if chosen_ratio_column not in combine_df.columns:
    raise Exception(f"Column '{chosen_ratio_column}' not found in combine.csv")

# Determine dataset column in combine_df
for col in ['dataset name', 'Dataset', 'dataset']:
    if col in combine_df.columns:
        combine_dataset_col = col
        break
else:
    raise Exception("No dataset column found in combine.csv")

# Ensure a decomposition column exists (rename if necessary)
if 'decomposition' not in combine_df.columns and 'Decomposition' in combine_df.columns:
    combine_df.rename(columns={'Decomposition': 'decomposition'}, inplace=True)
if 'decomposition' not in combine_df.columns:
    raise Exception("No 'decomposition' column found in combine.csv")

# Create a canonical version of the decomposition column (no subtraction for combine.csv)
combine_df['decomposition_canonical'] = combine_df['decomposition'].apply(lambda x: canonicalize_config(x, subtract=0))

# --- Step 3: For each dataset, select best config row and compute geometric mean excluding best config rows ---
agg_results = []  # Aggregated metrics per dataset
best_config_details = []  # Best config row details from combine.csv

for dataset, group in combine_df.groupby(combine_dataset_col):
    if dataset not in best_config_mapping:
        print(f"Dataset '{dataset}' not found in best config mapping; skipping.")
        continue
    best_config = best_config_mapping[dataset]

    # Select rows with canonical decomposition matching best_config exactly
    group_best = group[group['decomposition_canonical'] == best_config]
    if group_best.empty:
        print(f"Dataset '{dataset}': no rows with best config '{best_config}' found; skipping.")
        sample_vals = group['decomposition_canonical'].unique()[:5]
        print(f"Sample decomposition_canonical values for '{dataset}': {sample_vals}")
        continue

    # Determine the best config row using the chosen ratio column
    best_ratio_value = group_best[chosen_ratio_column].max(skipna=True)
    best_ratio_index = group_best[chosen_ratio_column].idxmax(skipna=True)

    print(f"\nDataset: {dataset}")
    print("Best ratio found in column:", chosen_ratio_column)
    print("Best ratio value:", best_ratio_value)
    print("Row index of best ratio:", best_ratio_index)

    best_row = group_best.loc[best_ratio_index]

    # For the geometric mean, exclude all rows that have the best config
    non_best = group[group['decomposition_canonical'] != best_config]
    if len(non_best) > 0:
        valid_vals = non_best[chosen_ratio_column][non_best[chosen_ratio_column] > 0]
        if len(valid_vals) > 0:
            gmean_excluding = np.exp(np.mean(np.log(valid_vals)))
        else:
            gmean_excluding = np.nan
    else:
        gmean_excluding = np.nan

    agg_results.append({
        combine_dataset_col: dataset,
        'best_config': best_config,
        'best_ratio_column': chosen_ratio_column,
        'best_ratio': best_ratio_value,
        'gmean_excluding_best': gmean_excluding
    })

    best_config_details.append(best_row)

# --- Step 4: Save aggregated metrics and best config details to CSV files ---
agg_df = pd.DataFrame(agg_results)
best_config_df = pd.DataFrame(best_config_details)

agg_df.to_csv('/home/jamalids/Documents/frame/new/big-data-compression/modeling/abaltion/aggregated_gmean_ratios.csv',
              index=False)
best_config_df.to_csv(
    '/home/jamalids/Documents/frame/new/big-data-compression/modeling/abaltion/best_config_details.csv', index=False)

print("\nCSV files saved: 'aggregated_gmean_ratios.csv' and 'best_config_details.csv'.")

# --- Step 5: Plot the Results ---
x = np.arange(len(agg_df))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 8))
bar1 = ax.bar(x - width / 2, agg_df['gmean_excluding_best'], width,
              label='Geometric Mean (Excluding Best Config)', color='skyblue')
bar2 = ax.bar(x + width / 2, agg_df['best_ratio'], width,
              label='Best Config Ratio', color='orange')

ax.set_xlabel('Dataset', fontsize=12)
ax.set_ylabel('Ratio Value', fontsize=12)
ax.set_title('Comparison of Geometric Mean (Excluding Best Config) and Best Config Ratio per Dataset', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(agg_df[combine_dataset_col], rotation=45, ha='right', fontsize=10)
ax.legend(fontsize=12)


def autolabel(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)


autolabel(bar1)
autolabel(bar2)

plt.tight_layout()
plt.savefig('/home/jamalids/Documents/frame/new/big-data-compression/modeling/abaltion/gmean_ratios2.png')

