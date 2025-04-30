import ast
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# VLDB-style font settings
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

# =============================
# Your original logic unchanged
# =============================

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

# Step 1: Load best clustering results
best_df = pd.read_csv("/mnt/c/Users/jamalids/Downloads/figs/results/best_results32.csv")
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
combine_df = pd.read_csv('/mnt/c/Users/jamalids/Downloads/figs/results/combine_all.csv')
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

# Load DatasetIdMapping and merge
mapping = pd.read_csv("/mnt/c/Users/jamalids/Downloads/figs/DatasetIdMapping.csv")
combine_df = pd.merge(combine_df, mapping[['DatasetName', 'DatasetID']], left_on=combine_dataset_col, right_on='DatasetName', how='inner')

# Step 3: Compare clustering vs best exhaustive
agg_results = []
best_exhaustive_details = []

for dataset, group in combine_df.groupby('DatasetID'):
    dataset_name = group['DatasetName'].iloc[0]

    if dataset_name not in best_config_mapping:
        print(f"Dataset '{dataset_name}' not found in best config mapping; skipping.")
        continue

    clustering_config = best_config_mapping[dataset_name]
    clustering_norm = normalize_config(clustering_config)

    clustering_rows = group[group['decomposition_canonical'].apply(lambda x: normalize_config(x) == clustering_norm)]
    clustering_best_ratio = clustering_rows[chosen_ratio_column].max(skipna=True) if not clustering_rows.empty else np.nan

    best_exhaustive_row = group.loc[group[chosen_ratio_column].idxmax()]
    best_exhaustive_ratio = best_exhaustive_row[chosen_ratio_column]

    agg_results.append({
        'DatasetID': dataset,
        'clustering_config': clustering_config,
        'clustering_ratio': clustering_best_ratio,
        'best_exhaustive_ratio': best_exhaustive_ratio
    })

    best_exhaustive_details.append(best_exhaustive_row)

agg_df = pd.DataFrame(agg_results)
best_exhaustive_df = pd.DataFrame(best_exhaustive_details)

# Save to CSV
agg_df.to_csv('/mnt/c/Users/jamalids/Downloads/figs/clustering_vs_exhaustive_all.csv', index=False)
best_exhaustive_df.to_csv('/mnt/c/Users/jamalids/Downloads/figs/best_exhaustive_details_all.csv', index=False)

print("\nCSV files saved: 'clustering_vs_exhaustive_all.csv' and 'best_exhaustive_details_all.csv'.")

# Step 4: Plot clustering vs best exhaustive
x = np.arange(len(agg_df))
width = 0.35

fig, ax = plt.subplots(figsize=(6.2, 2.5))  # VLDB figure size
bar1 = ax.bar(x - width/2, agg_df['clustering_ratio'], width, label='Clustering Config', color='orange')
bar2 = ax.bar(x + width/2, agg_df['best_exhaustive_ratio'], width, label='Best Exhaustive Config', color='skyblue')

ax.set_xlabel('Dataset ID')
ax.set_ylabel('CR (TDT )')
#ax.set_title('Clustering vs. Best Exhaustive Config')
ax.set_xticks(x)
ax.set_xticklabels(agg_df['DatasetID'], rotation=90, ha='right')
ax.legend()

def autolabel(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

#autolabel(bar1)
#autolabel(bar2)

plt.tight_layout()
plt.savefig('/mnt/c/Users/jamalids/Downloads/figs/clustering_vs_exhaustive_all.pdf')  # Save as PDF
plt.close()
