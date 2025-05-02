import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# VLDB-style formatting


matplotlib.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Latin Modern Roman", "Times", "DejaVu Serif"],
    "font.size": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})
# -------------------------------
# Load and Filter
# -------------------------------
df = pd.read_csv('/mnt/c/Users/jamalids/Downloads/figs/combine-64.CSV')
allowed_scenarios = ['all_features', 'entropy', 'frequency']

metrics = {
    'Silhouette': 'max',
    'DaviesBouldin': 'min',
    'CalinskiHarabasz': 'max',
    'GapStatistic': 'max'
}

# -------------------------------
# Select Best Rows
# -------------------------------
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
                best_row = valid_group[valid_group[metric] == valid_group[metric].max()].iloc[0]
            else:
                best_row = valid_group[valid_group[metric] == valid_group[metric].min()].iloc[0]
            best_results.append({
                'Dataset': dataset,
                'FeatureScenario': scenario,
                'Metric': metric,
                'BestMetricValue': best_row[metric],
                'DecomposedRatio': best_row['DecomposedRatio_ColOrder'],
                'k': best_row['k'],
                'config': best_row['ClusterConfig']
            })

best_df = pd.DataFrame(best_results)
best_df.to_csv("/mnt/c/Users/jamalids/Downloads/best_results64.csv", index=False)
print("Saved best valid results to best_results64.csv")

# -------------------------------
# Geometric Mean Calculation
# -------------------------------
def geom_mean(series):
    positive = series[series > 0]
    if len(positive) == 0:
        return np.nan
    return np.exp(np.mean(np.log(positive)))

gmean_results = []
for (scenario, metric), group in best_df.groupby(['FeatureScenario', 'Metric']):
    gm_value = geom_mean(group['DecomposedRatio'])
    gmean_results.append({
        'FeatureScenario': scenario,
        'Metric': metric,
        'GeomMeanDecomposedRatio': gm_value
    })
gmean_df = pd.DataFrame(gmean_results)
gmean_df.to_csv("/mnt/c/Users/jamalids/Downloads/gmean_results64.csv", index=False)
print("Saved geometric mean results to gmean_results64.csv")

# -------------------------------
# Plot in VLDB style
# -------------------------------
plt.figure(figsize=(6.8, 3))
for metric in metrics.keys():
    subdf = gmean_df[gmean_df['Metric'] == metric].sort_values('FeatureScenario')
    plt.plot(subdf['FeatureScenario'], subdf['GeomMeanDecomposedRatio'], marker='o', label=metric)

plt.xlabel("FeatureScenario")
plt.ylabel("GMean of DTD (64 bits)")
# plt.title("Geometric Mean Compression Ratio per FeatureScenario for Each Metric")  # VLDB: omit title
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("/mnt/c/Users/jamalids/Downloads/gmean_decomp_ratio64.pdf")
plt.close()
