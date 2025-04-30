import pandas as pd
import numpy as np
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

#########################################################
import os
import pandas as pd

best_df = pd.read_csv("/mnt/c/Users/jamalids/Downloads/figs/results/best_results32.csv")
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
gmean_df.to_csv("/mnt/c/Users/jamalids/Downloads/gmean_results32.csv", index=False)
print("Saved geometric mean results to gmean_results32H.csv")

# --------------------------
# Step 4: Plot Geometric Mean of Compression Ratio per FeatureScenario for each Metric
# --------------------------
plt.figure(figsize=(6.2, 2.5))  # <-- VLDB size
for metric in metrics.keys():
    subdf = gmean_df[gmean_df['Metric'] == metric].sort_values('FeatureScenario')
    plt.plot(subdf['FeatureScenario'], subdf['GeomMeanDecomposedRatio'], marker='o', label=metric)

plt.xlabel("FeatureScenario")
plt.ylabel("GMean of DTD(32 bits)")
#plt.title("Geometric Mean Compression Ratio per FeatureScenario for Each Metric")
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("/mnt/c/Users/jamalids/Downloads/gmean_decomp_ratio32.pdf")  # <-- Save as PDF
plt.close()
