import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def canonicalize_config(config_val, subtract=0):
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
    if isinstance(t, tuple):
        return "(" + ",".join(str(x) for x in t) + ")"
    else:
        return str(t)


def normalize_config(config_str):
    try:
        config = ast.literal_eval(config_str)
        if isinstance(config, (list, tuple)):
            normalized_inner = []
            for inner in config:
                if not isinstance(inner, (list, tuple)):
                    inner = (inner,)
                else:
                    inner = tuple(inner)
                normalized_inner.append(tuple(sorted(inner)))
            normalized_outer = tuple(sorted(normalized_inner))
            return normalized_outer
        else:
            return config_str
    except Exception:
        return config_str


# Step 1: Load clustering (best config) results
best_df = pd.read_csv('/home/jamalids/Documents/frame/new2/big-data-compression/modeling/abaltion/exhusive/best_results_all.csv')
for col in ['config', 'FeatureScenario', 'Metric']:
    if col not in best_df.columns:
        raise Exception(f"Column '{col}' not found in best_results32.csv")

best_df = best_df[(best_df['FeatureScenario'].str.lower() == 'frequency') &
                  (best_df['Metric'] == 'DaviesBouldin')]
if best_df.empty:
    raise Exception("No rows found with FeatureScenario 'Frequency' and Metric 'DaviesBouldin'.")

for col in ['dataset name', 'Dataset', 'dataset']:
    if col in best_df.columns:
        best_dataset_col = col
        break
else:
    raise Exception("No dataset column found in best_results32.csv")

best_config_mapping = {}
for _, row in best_df.iterrows():
    dataset = row[best_dataset_col]
    config_val = row['config']
    canonical = canonicalize_config(config_val, subtract=1)
    best_config_mapping[dataset] = canonical


# Step 2: Load combine.csv
combine_df = pd.read_csv('/home/jamalids/Documents/frame/new2/big-data-compression/modeling/abaltion/exhusive/combine_all.csv')
chosen_ratio_column = "decomposed zstd compression ratio"
if chosen_ratio_column not in combine_df.columns:
    raise Exception(f"Column '{chosen_ratio_column}' not found in combine.csv")

for col in ['dataset name', 'Dataset', 'dataset']:
    if col in combine_df.columns:
        combine_dataset_col = col
        break
else:
    raise Exception("No dataset column found in combine.csv")

if 'decomposition' not in combine_df.columns and 'Decomposition' in combine_df.columns:
    combine_df.rename(columns={'Decomposition': 'decomposition'}, inplace=True)
if 'decomposition' not in combine_df.columns:
    raise Exception("No 'decomposition' column found in combine.csv")

combine_df['decomposition_canonical'] = combine_df['decomposition'].apply(lambda x: canonicalize_config(x, subtract=0))


# Step 3: Compare clustering vs best exhaustive
agg_results = []
best_exhaustive_details = []

for dataset, group in combine_df.groupby(combine_dataset_col):
    if dataset not in best_config_mapping:
        print(f"Dataset '{dataset}' not found in best config mapping; skipping.")
        continue

    clustering_config = best_config_mapping[dataset]
    clustering_norm = normalize_config(clustering_config)

    clustering_rows = group[group['decomposition_canonical'].apply(lambda x: normalize_config(x) == clustering_norm)]
    clustering_best_ratio = clustering_rows[chosen_ratio_column].max(skipna=True) if not clustering_rows.empty else np.nan

    best_exhaustive_row = group.loc[group[chosen_ratio_column].idxmax()]
    best_exhaustive_ratio = best_exhaustive_row[chosen_ratio_column]

    agg_results.append({
        combine_dataset_col: dataset,
        'clustering_config': clustering_config,
        'clustering_ratio': clustering_best_ratio,
        'best_exhaustive_ratio': best_exhaustive_ratio
    })

    best_exhaustive_details.append(best_exhaustive_row)

agg_df = pd.DataFrame(agg_results)
best_exhaustive_df = pd.DataFrame(best_exhaustive_details)

# Save to CSV
agg_df.to_csv('/home/jamalids/Documents/frame/new2/big-data-compression/modeling/abaltion/clustering_vs_exhaustive_all.csv', index=False)
best_exhaustive_df.to_csv('/home/jamalids/Documents/frame/new2/big-data-compression/modeling/abaltion/best_exhaustive_details_all.csv', index=False)

print("\nCSV files saved: 'clustering_vs_exhaustive.csv' and 'best_exhaustive_details.csv'.")


# Step 4: Plot clustering vs best exhaustive
x = np.arange(len(agg_df))
width = 0.35

fig, ax = plt.subplots(figsize=(14, 8))
bar1 = ax.bar(x - width/2, agg_df['clustering_ratio'], width, label='Clustering Config', color='orange')
bar2 = ax.bar(x + width/2, agg_df['best_exhaustive_ratio'], width, label='Best Exhaustive Config', color='skyblue')

ax.set_xlabel('Dataset', fontsize=12)
ax.set_ylabel('Decomposed Zstd Compression Ratio', fontsize=12)
ax.set_title('Clustering vs. Best Exhaustive Config', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(agg_df[combine_dataset_col], rotation=45, ha='right', fontsize=10)
ax.legend(fontsize=12)

def autolabel(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

autolabel(bar1)
autolabel(bar2)

# Step 5: Table under first plot
table_data = []
for idx, row in agg_df.iterrows():
    dataset = row[combine_dataset_col]
    clustering_config = row['clustering_config']

    best_exhaustive_row = [r for r in best_exhaustive_details if r[combine_dataset_col] == dataset]
    best_exhaustive_config = best_exhaustive_row[0]['decomposition_canonical'] if best_exhaustive_row else ""

    table_data.append([dataset, clustering_config, best_exhaustive_config])
#
# table = ax.table(cellText=table_data,
#                  colLabels=["Dataset", "Clustering Config", "Best Exhaustive Config"],
#                  cellLoc="center",
#                  loc="bottom",
#                  bbox=[0, -1.3, 1, 0.8])

plt.subplots_adjust(bottom=0.25)
plt.savefig('/home/jamalids/Documents/frame/new2/big-data-compression/modeling/abaltion/clustering_vs_exhaustive_all.png')


# Step 6: Min-Max plot across all configs
min_max_data = []
for dataset, group in combine_df.groupby(combine_dataset_col):
    min_ratio = group[chosen_ratio_column].min(skipna=True)
    max_ratio = group[chosen_ratio_column].max(skipna=True)
    min_max_data.append({
        combine_dataset_col: dataset,
        'min_ratio': min_ratio,
        'max_ratio': max_ratio
    })

minmax_df = pd.DataFrame(min_max_data)
minmax_df = minmax_df.sort_values(by=combine_dataset_col).reset_index(drop=True)

# Plotting min-max range
x = np.arange(len(minmax_df))
fig, ax = plt.subplots(figsize=(14, 8))

# Vertical lines
for i, row in minmax_df.iterrows():
    ax.plot([x[i], x[i]], [row['min_ratio'], row['max_ratio']], color='gray', linewidth=2)

# Dots at min and max
ax.scatter(x, minmax_df['max_ratio'], color='blue', label='Max Compression Ratio', zorder=5)
ax.scatter(x, minmax_df['min_ratio'], color='red', label='Min Compression Ratio', zorder=5)

ax.set_xlabel('Dataset', fontsize=12)
ax.set_ylabel('Decomposed Zstd Compression Ratio', fontsize=12)
ax.set_title('Min-Max Compression Ratio per Dataset (All Configs)', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(minmax_df[combine_dataset_col], rotation=45, ha='right', fontsize=10)
ax.legend(fontsize=12)

plt.tight_layout()
plt.savefig('/home/jamalids/Documents/frame/new2/big-data-compression/modeling/abaltion/min_max_compression_ratios_all.png')
