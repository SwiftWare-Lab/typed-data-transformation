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
#########################################################
import os
import pandas as pd

best_df = pd.read_csv("/mnt/c/Users/jamalids/Downloads/figs/results/best_results32.csv")
#best_df  = pd.read_csv('/mnt/c/Users/jamalids/Downloads/figs/combine-64.CSV')
# Define allowed FeatureScenarios (case-insensitive)
allowed_scenarios = ['all_features', 'entropy', 'frequency']
metrics = {
    'Silhouette': 'max',
    'DaviesBouldin': 'min',
    'CalinskiHarabasz': 'max',
    'GapStatistic': 'max'
}

# --------------------------
# Step 1: Compute Best Valid Results per Dataset, FeatureScenario, and Metric
# --------------------------
def geom_mean(series):
    # Compute geometric mean using only positive values (to avoid issues with log)
    positive = series[series > 0]
    if len(positive) == 0:
        return np.nan
    return np.exp(np.mean(np.log(positive)))

# --------------------------
# Step 3: Compute Geometric Mean of Compression Ratio per FeatureScenario and Metric (Across Datasets)
# --------------------------
gmean_results = []
# Group the best results by FeatureScenario and Metric
for (scenario, metric), group in best_df.groupby(['FeatureScenario', 'Metric']):
    gm_value = geom_mean(group['DecomposedRatio'])
    gmean_results.append({
        'FeatureScenario': scenario,
        'Metric': metric,
        'GeomMeanDecomposedRatio': gm_value
    })
gmean_df = pd.DataFrame(gmean_results)
gmean_df.to_csv("/mnt/c/Users/jamalids/Downloads/gmean_results64.csv", index=False)
print("Saved geometric mean results to gmean_results32H.csv")

# --------------------------
# Step 4: Plot Geometric Mean of Compression Ratio per FeatureScenario for each Metric
# --------------------------
plt.figure(figsize=(6.8, 3))  # <-- VLDB size
for metric in metrics.keys():
    subdf = gmean_df[gmean_df['Metric'] == metric].sort_values('FeatureScenario')
    plt.plot(subdf['FeatureScenario'], subdf['GeomMeanDecomposedRatio'], marker='o', label=metric)

plt.xlabel("FeatureScenario")
plt.ylabel("GMean of DTD(32 bits)")
#plt.title("Geometric Mean Compression Ratio per FeatureScenario for Each Metric")
plt.legend( loc='upper right')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("/mnt/c/Users/jamalids/Downloads/gmean_decomp_ratio32.pdf")  # <-- Save as PDF
plt.close()
