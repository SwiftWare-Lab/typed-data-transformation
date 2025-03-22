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


def normalize_config(config_str):
    """
    Given a canonical configuration string (e.g. "((0,1),(2),(3))"),
    return a normalized representation that is invariant to the order within
    each cluster and the order of clusters.

    The function parses the string into a tuple of tuples, then sorts each inner
    tuple and sorts the outer collection. The returned value is a tuple of sorted tuples.
    """
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


# --- Step 1: Create best config mapping from best_results32H.csv ---
best_df = pd.read_csv('/home/jamalids/Documents/frame/new/big-data-compression/modeling/abaltion/best_results32H.csv')
for col in ['config', 'FeatureScenario', 'Metric']:
    if col not in best_df.columns:
        raise Exception(f"Column '{col}' not found in best_results32H.csv")

best_df = best_df[(best_df['FeatureScenario'].str.lower() == 'frequency') &
                  (best_df['Metric'] == 'DaviesBouldin')]
if best_df.empty:
    raise Exception("No rows found in best_results32H.csv with FeatureScenario 'Frequency' and Metric 'DaviesBouldin'.")

for col in ['dataset name', 'Dataset', 'dataset']:
    if col in best_df.columns:
        best_dataset_col = col
        break
else:
    raise Exception("No dataset column found in best_results32H.csv")

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


# --- Step 3: For each dataset, compute mapping best config ratio and remove all equivalent rows ---
# Then, from the remaining rows, find the best remaining config ratio.
agg_results = []  # Aggregated metrics per dataset
remaining_best_details = []  # Best remaining config row details for each dataset

for dataset, group in combine_df.groupby(combine_dataset_col):
    if dataset not in best_config_mapping:
        print(f"Dataset '{dataset}' not found in best config mapping; skipping.")
        continue
    mapping_best_config = best_config_mapping[dataset]
    norm_mapping = normalize_config(mapping_best_config)

    # Select rows equivalent (in any ordering) to the mapping best config.
    group_mapping = group[group['decomposition_canonical'].apply(lambda x: normalize_config(x) == norm_mapping)]
    if group_mapping.empty:
        print(f"Dataset '{dataset}': no rows found matching mapping best config '{mapping_best_config}'.")
        mapping_best_ratio = np.nan
    else:
        mapping_best_ratio = group_mapping[chosen_ratio_column].max(skipna=True)

    # Remove all rows equivalent to the mapping best config.
    remaining = group[group['decomposition_canonical'].apply(lambda x: normalize_config(x) != norm_mapping)]
    if remaining.empty:
        print(f"Dataset '{dataset}': no remaining rows after removing mapping best config equivalents.")
        remaining_best_ratio = np.nan
        remaining_best_row = None
    else:
        remaining_best_row = remaining.loc[remaining[chosen_ratio_column].idxmax()]
        remaining_best_ratio = remaining_best_row[chosen_ratio_column]

    agg_results.append({
        combine_dataset_col: dataset,
        'mapping_best_config': mapping_best_config,
        'mapping_best_ratio': mapping_best_ratio,
        'remaining_best_ratio': remaining_best_ratio
    })
    remaining_best_details.append(remaining_best_row)

# --- Step 4: Save aggregated metrics and remaining best config details to CSV files ---
agg_df = pd.DataFrame(agg_results)
remaining_best_df = pd.DataFrame([row for row in remaining_best_details if row is not None])

agg_df.to_csv('/home/jamalids/Documents/frame/new2/big-data-compression/modeling/abaltion/aggregated_remaining_best_ratios.csv', index=False)
remaining_best_df.to_csv('/home/jamalids/Documents/frame/new2/big-data-compression/modeling/abaltion/remaining_best_config_details.csv', index=False)

print("\nCSV files saved: 'aggregated_remaining_best_ratios.csv' and 'remaining_best_config_details.csv'.")


# --- Step 5: Plot the Results ---
x = np.arange(len(agg_df))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 8))
bar1 = ax.bar(x - width/2, agg_df['mapping_best_ratio'], width,
              label='Clustering Config', color='orange')
bar2 = ax.bar(x + width/2, agg_df['remaining_best_ratio'], width,
              label='Exhaustive Config', color='skyblue')

ax.set_xlabel('Dataset', fontsize=12)
ax.set_ylabel('Decomposed Zstd Compression Ratio', fontsize=12)
ax.set_title('Clustering Config vs. Exhaustive Config Ratios', fontsize=14)
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

# --- Add a table below the plot showing the clustering config and the exhaustive config for each dataset ---
table_data = []
for idx, row in agg_df.iterrows():
    rem_best_config = ""
    match = remaining_best_df[remaining_best_df[combine_dataset_col] == row[combine_dataset_col]]
    if not match.empty:
        rem_best_config = match.iloc[0]['decomposition_canonical']
    table_data.append([row[combine_dataset_col], row['mapping_best_config'], rem_best_config])

table = ax.table(cellText=table_data,
                 colLabels=["Dataset", "Clustering Config", "Exhaustive Config"],
                 cellLoc="center",
                 loc="bottom",
                 bbox=[0, -1, 1, 0.7])

plt.subplots_adjust(bottom=0.45)
plt.savefig('/home/jamalids/Documents/frame/new2/big-data-compression/modeling/abaltion/remaining_best_ratios.png')

