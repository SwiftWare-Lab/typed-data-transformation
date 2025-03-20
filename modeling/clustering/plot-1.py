import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

# Read the CSV file
df = pd.read_csv(r"C:\Users\jamalids\Downloads\dataset\Low-Entropy\Low-Entropy\32\clustering_compression_results.csv")

# Define which metric to use and the corresponding optimization direction:
# "max" means highest is best; "min" means lowest is best.
metrics = {
    'Silhouette': ('max', 'Silhouette'),
    'DaviesBouldin': ('min', 'DaviesBouldin'),
    'CalinskiHarabasz': ('max', 'CalinskiHarabasz'),
    'GapStatistic': ('max', 'GapStatistic')
}

# Get unique feature scenarios from the data.
feature_scenarios = df['FeatureScenario'].unique()
n_scenarios = len(feature_scenarios)

# Define grid dimensions: for example, 2 columns and enough rows to cover all scenarios.
ncols = 2
nrows = math.ceil(n_scenarios / ncols)

# Create subplots in a grid.
fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows), sharex=True, sharey=True)
axes = axes.flatten()  # flatten for easier iteration

# Loop over each FeatureScenario and plot in its subplot.
for ax, scenario in zip(axes, feature_scenarios):
    df_scenario = df[df['FeatureScenario'] == scenario]

    # Create a dictionary to store the best rows for each metric.
    best_rows = {}
    for metric_name, (opt, col) in metrics.items():
        if opt == 'max':
            best = df_scenario.groupby('Dataset').apply(lambda g: g.loc[g[col].idxmax()])
        else:
            best = df_scenario.groupby('Dataset').apply(lambda g: g.loc[g[col].idxmin()])
        best = best.reset_index(drop=True)
        best_rows[metric_name] = best

    # Dictionary to record annotated points per dataset (to avoid duplicate annotations).
    annotated = {}  # keys: dataset; values: set of rounded y-values annotated.

    # Plot each metric's line.
    for m_idx, (metric_name, best_df) in enumerate(best_rows.items()):
        best_df_sorted = best_df.sort_values('Dataset')
        x = best_df_sorted['Dataset']
        y = best_df_sorted['DecomposedRatio_ColOrder']

        ax.plot(x, y, marker='o', label=metric_name)

        # Annotate each point with its ClusterConfig.
        for i, row in best_df_sorted.iterrows():
            dataset = row['Dataset']
            y_val = row['DecomposedRatio_ColOrder']
            y_round = round(y_val, 3)
            if dataset not in annotated:
                annotated[dataset] = set()
            # If already annotated at this y, skip annotation.
            if y_round in annotated[dataset]:
                continue
            annotated[dataset].add(y_round)
            # Add a small offset based on the metric index to separate overlapping annotations.
            offset = m_idx * 0.005  # adjust offset as needed
            ax.text(dataset, y_val + offset, row['ClusterConfig'],
                    fontsize=8, rotation=45, ha='left', va='bottom')

    ax.set_title(f"FeatureScenario: {scenario}")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("DecomposedRatio_ColOrder")
    ax.legend(title="Metric")

# Hide any unused subplots.
for i in range(len(feature_scenarios), len(axes)):
    fig.delaxes(axes[i])

# Rotate x-axis tick labels for each subplot individually.
for ax in axes:
    for label in ax.get_xticklabels():
        label.set_rotation(45)

plt.tight_layout()
plt.savefig("plot1.png")
plt.show()
######################
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

# Read the CSV file
df = pd.read_csv(r"C:\Users\jamalids\Downloads\dataset\Low-Entropy\Low-Entropy\32\clustering_compression_results.csv")

# Define which metric to use and the corresponding optimization direction:
# "max" means highest is best; "min" means lowest is best.
metrics = {
    'Silhouette': ('max', 'Silhouette'),
    'DaviesBouldin': ('min', 'DaviesBouldin'),
    'CalinskiHarabasz': ('max', 'CalinskiHarabasz'),
    'GapStatistic': ('max', 'GapStatistic')
}

# Get unique feature scenarios from the data.
feature_scenarios = df['FeatureScenario'].unique()
n_scenarios = len(feature_scenarios)

# Define grid dimensions: for example, 2 columns and enough rows to cover all scenarios.
ncols = 2
nrows = math.ceil(n_scenarios / ncols)

# Create subplots in a grid.
fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows), sharex=True, sharey=True)
axes = axes.flatten()  # flatten for easier iteration

# Loop over each FeatureScenario and plot in its subplot.
for ax, scenario in zip(axes, feature_scenarios):
    df_scenario = df[df['FeatureScenario'] == scenario]

    # Create a dictionary to store the best rows for each metric.
    best_rows = {}
    for metric_name, (opt, col) in metrics.items():
        if opt == 'max':
            best = df_scenario.groupby('Dataset').apply(lambda g: g.loc[g[col].idxmax()])
        else:
            best = df_scenario.groupby('Dataset').apply(lambda g: g.loc[g[col].idxmin()])
        best = best.reset_index(drop=True)
        best_rows[metric_name] = best

    # Dictionary to record annotated points per dataset (to avoid duplicate annotations).
    annotated = {}  # keys: dataset; values: set of rounded y-values annotated.

    # Plot each metric's line.
    for m_idx, (metric_name, best_df) in enumerate(best_rows.items()):
        best_df_sorted = best_df.sort_values('Dataset')
        x = best_df_sorted['Dataset']
        y = best_df_sorted['DecomposedRatio_ColOrder']

        ax.plot(x, y, marker='o', label=metric_name)

        # Annotate each point with its k (number of clusters) instead of ClusterConfig.
        for i, row in best_df_sorted.iterrows():
            dataset = row['Dataset']
            y_val = row['DecomposedRatio_ColOrder']
            y_round = round(y_val, 3)
            if dataset not in annotated:
                annotated[dataset] = set()
            # If already annotated at this y, skip annotation.
            if y_round in annotated[dataset]:
                continue
            annotated[dataset].add(y_round)
            # Add a small offset based on the metric index to separate overlapping annotations.
            offset = m_idx * 0.005  # adjust offset as needed
            ax.text(dataset, y_val + offset, str(row['k']),
                    fontsize=8, rotation=45, ha='left', va='bottom')

    ax.set_title(f"FeatureScenario: {scenario}")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("DecomposedRatio_ColOrder")
    ax.legend(title="Metric")

# Hide any unused subplots.
for i in range(len(feature_scenarios), len(axes)):
    fig.delaxes(axes[i])

# Rotate x-axis tick labels for each subplot individually.
for ax in axes:
    for label in ax.get_xticklabels():
        label.set_rotation(45)

plt.tight_layout()
plt.savefig("plot2.png")
plt.show()

