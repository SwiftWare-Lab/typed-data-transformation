import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Define allowed FeatureScenarios
allowed_scenarios = ['all_features', 'entropy', 'frequency']

# Read the CSV file
df = pd.read_csv(r'/mnt/c/Users/jamalids/Downloads/code/final-python/big-data-compression/modeling/clustering/30%/32.csv')

# Read DatasetIdMapping
mapping = pd.read_csv("/mnt/c/Users/jamalids/Downloads/figs/DatasetIdMapping.csv")

# --------------------------
# Step 1: Compute Best Valid Results per Dataset, FeatureScenario, and Metric
# --------------------------

def geom_mean(series):
    positive = series[series > 0]
    if len(positive) == 0:
        return np.nan
    return np.exp(np.mean(np.log(positive)))

metrics = {
    'Silhouette': 'max',
    'DaviesBouldin': 'min',
    'CalinskiHarabasz': 'max',
    'GapStatistic': 'max'
}

best_results = []

for dataset, dgroup in df.groupby('Dataset'):
    for scenario, group in dgroup.groupby('FeatureScenario'):
        if scenario.strip().lower() not in allowed_scenarios:
            continue
        for metric, rule in metrics.items():
            valid_group = group[(group[metric] != -1) & (~np.isinf(group[metric]))]
            if valid_group.empty:
                continue
            if rule == 'max':
                best_metric_value = valid_group[metric].max()
                best_row = valid_group[valid_group[metric] == best_metric_value].iloc[0]
            else:
                best_metric_value = valid_group[metric].min()
                best_row = valid_group[valid_group[metric] == best_metric_value].iloc[0]
            best_results.append({
                'Dataset': dataset,
                'FeatureScenario': scenario,
                'Metric': metric,
                'BestMetricValue': best_metric_value,
                'DecomposedRatio': best_row['DecomposedRatio_ColOrder'],
                'k': best_row['k'],
                'config': best_row['ClusterConfig']
            })

# Save best results
best_df = pd.DataFrame(best_results)
best_df.to_csv("/mnt/c/Users/jamalids/Downloads/best_results32.csv", index=False)
print("Saved best valid results to best_results32.csv")

# ================================
# Part 2: Best DaviesBouldin Rows Plot
# ================================

# Filter out invalid DaviesBouldin values
df_valid = df[(df['DaviesBouldin'] != -1) & (~np.isinf(df['DaviesBouldin']))].copy()

# Select best DaviesBouldin row per (Dataset, FeatureScenario)
best_silhouette_rows = df_valid.loc[df_valid.groupby(['Dataset', 'FeatureScenario'])['DaviesBouldin'].idxmin()]
best_silhouette_rows = best_silhouette_rows[best_silhouette_rows['FeatureScenario'].str.lower().isin(allowed_scenarios)]

# Merge DatasetIdMapping to add 'DatasetID'
best_silhouette_rows = pd.merge(
    best_silhouette_rows,
    mapping[['DatasetName', 'DatasetID']],
    left_on='Dataset',
    right_on='DatasetName',
    how='inner'
)

# Set VLDB-style font and figure settings
mpl.rcParams.update({
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

plt.figure(figsize=(6.2, 2.5))
ax = plt.gca()

for scenario, group in best_silhouette_rows.groupby('FeatureScenario'):
    group_sorted = group.sort_values('DatasetID')
    lower_scenario = scenario.strip().lower()

    if lower_scenario == "all_features":
        ax.plot(group_sorted['DatasetID'], group_sorted['DecomposedRatio_ColOrder'],
                marker='*', markersize=6, linewidth=1.5, linestyle='-', color='red', label=scenario)
    elif lower_scenario == "entropy":
        ax.plot(group_sorted['DatasetID'], group_sorted['DecomposedRatio_ColOrder'],
                marker='^', markersize=6, linewidth=1.5, linestyle='-', color='orange', label=scenario)
    elif lower_scenario == "frequency":
        ax.plot(group_sorted['DatasetID'], group_sorted['DecomposedRatio_ColOrder'],
                marker='s', markersize=6, linewidth=1.5, linestyle='-', color='green', label=scenario)

    # Annotate with k value
    # for idx, row in group_sorted.iterrows():
    #     ax.text(row['DatasetID'], row['DecomposedRatio_ColOrder'], str(row['k']),
    #             ha='center', va='bottom', fontsize=7)

ax.set_xlabel("Dataset ID")
ax.set_ylabel("log(CR)")
#ax.set_title("Best DaviesBouldin Metric: Compression Ratio per Dataset")
ax.set_yscale('log')
ax.legend(title="FeatureScenario", loc='best')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("/mnt/c/Users/jamalids/Downloads/gmean_decomp_ratio_30.png")

plt.savefig("/mnt/c/Users/jamalids/Downloads/ratio30.pdf", bbox_inches='tight')
plt.close()

plt.close()
