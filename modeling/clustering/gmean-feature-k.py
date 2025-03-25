# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
#
# # Read the CSV file
# df = pd.read_csv('/home/jamalids/Downloads/OneDrive_1_17-03-2025/32.csv')
#
#
# # --------------------------
# # Step 1: Compute Best Valid Results per Dataset, FeatureScenario, and Metric
# # --------------------------
# def geom_mean(series):
#     # Compute geometric mean using only positive values (to avoid issues with log)
#     positive = series[series > 0]
#     if len(positive) == 0:
#         return np.nan
#     return np.exp(np.mean(np.log(positive)))
#
# # Define metrics and their "best" rule:
# # For Silhouette, CalinskiHarabasz, and GapStatistic the best is the maximum.
# # For DaviesBouldin the best is the minimum.
# metrics = {
#     'Silhouette': 'max',
#     'DaviesBouldin': 'min',
#     'CalinskiHarabasz': 'max',
#     'GapStatistic': 'max'
# }
#
# # List to store best valid rows for each combination
# best_results = []
#
# # Group by Dataset and then by FeatureScenario
# for dataset, dgroup in df.groupby('Dataset'):
#     for scenario, group in dgroup.groupby('FeatureScenario'):
#         for metric, rule in metrics.items():
#             # Remove rows with invalid metric values (inf or -1)
#             valid_group = group[(group[metric] != -1) & (~np.isinf(group[metric]))]
#             if valid_group.empty:
#                 # If no valid row exists, skip this combination
#                 continue
#             if rule == 'max':
#                 best_metric_value = valid_group[metric].max()
#                 best_row = valid_group[valid_group[metric] == best_metric_value].iloc[0]
#             else:  # For 'min' rule
#                 best_metric_value = valid_group[metric].min()
#                 best_row = valid_group[valid_group[metric] == best_metric_value].iloc[0]
#             best_results.append({
#                 'Dataset': dataset,
#                 'FeatureScenario': scenario,
#                 'Metric': metric,
#                 'BestMetricValue': best_metric_value,
#                 'DecomposedRatio': best_row['DecomposedRatio_ColOrder'],
#                 'k': best_row['k']
#             })
#
# # Convert the best results to a DataFrame and save to CSV
# best_df = pd.DataFrame(best_results)
# best_df.to_csv("best_results32H.csv", index=False)
# print("Saved best valid results to best_results.csv")
#
# # --------------------------
# # Step 2: Plot Best Compression Ratio per Dataset for each Metric
# # --------------------------
# fig, ax = plt.subplots(figsize=(12, 8))
# for metric in metrics.keys():
#     subdf = best_df[best_df['Metric'] == metric].sort_values('Dataset')
#     ax.plot(subdf['Dataset'], subdf['DecomposedRatio'], marker='o', label=metric)
#     # Annotate each point with the corresponding k value
#     for idx, row in subdf.iterrows():
#         ax.text(row['Dataset'], row['DecomposedRatio'], str(row['k']),
#                 ha='center', va='bottom', fontsize=9)
# ax.set_xlabel("Dataset")
# ax.set_ylabel("Best DecomposedRatio_ColOrder")
# ax.set_title("Best Compression Ratio per Dataset for Each Metric")
# ax.legend()
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.savefig("best_decomp_ratio32H.png")
#
#
# # --------------------------
# # Step 3: Compute Geometric Mean of Compression Ratio per FeatureScenario and Metric (Across Datasets)
# # --------------------------
# gmean_results = []
# # Group the best results by FeatureScenario and Metric
# for (scenario, metric), group in best_df.groupby(['FeatureScenario', 'Metric']):
#     gm_value = geom_mean(group['DecomposedRatio'])
#     gmean_results.append({
#         'FeatureScenario': scenario,
#         'Metric': metric,
#         'GeomMeanDecomposedRatio': gm_value
#     })
# gmean_df = pd.DataFrame(gmean_results)
# gmean_df.to_csv("gmean_results32H.csv", index=False)
# print("Saved geometric mean results to gmean_results.csv")
#
# # --------------------------
# # Step 4: Plot Geometric Mean of Compression Ratio per FeatureScenario for each Metric
# # --------------------------
# fig, ax = plt.subplots(figsize=(12, 8))
# for metric in metrics.keys():
#     subdf = gmean_df[gmean_df['Metric'] == metric].sort_values('FeatureScenario')
#     ax.plot(subdf['FeatureScenario'], subdf['GeomMeanDecomposedRatio'], marker='o', label=metric)
# ax.set_xlabel("FeatureScenario")
# ax.set_ylabel("Geometric Mean of DecomposedRatio_ColOrder")
# ax.set_title("Geometric Mean Compression Ratio per FeatureScenario for Each Metric")
# ax.legend()
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.savefig("gmean_decomp_ratio32H.png")
# ###############
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
#
# # ================================
# # Part 1: Best Silhouette Rows Plot
# # ================================
#
# # Step 1: Read the CSV file
# df = pd.read_csv('/home/jamalids/Downloads/OneDrive_1_17-03-2025/32.csv')
#
# # Step 2: Filter out invalid Silhouette values (i.e. -1 or infinite)
# df_valid = df[(df['DaviesBouldin'] != -1) & (~np.isinf(df['DaviesBouldin']))].copy()
#
# # Step 3: For each (Dataset, FeatureScenario) combination, select the row with the best Silhouette metric
# # (Using idxmax ensures that for each group we pick the row with maximum Silhouette value.)
# best_silhouette_rows = df_valid.loc[df_valid.groupby(['Dataset', 'FeatureScenario'])['DaviesBouldin'].idxmax()]
#
# # Step 4: Create the line plot for best Silhouette rows, one line per FeatureScenario
# fig, ax = plt.subplots(figsize=(12, 8))
#
# # Loop over each FeatureScenario group
# for scenario, group in best_silhouette_rows.groupby('FeatureScenario'):
#     group_sorted = group.sort_values('Dataset')
#     # Check if the scenario is "all_features" (case-insensitive)
#     if scenario.strip().lower() == "all_features" :
#         ax.plot(group_sorted['Dataset'], group_sorted['DecomposedRatio_ColOrder'],
#                 marker='*', markersize=12, linestyle='-', color='red', label=scenario)
#     elif  scenario.strip().lower() == "Entropy":
#         ax.plot(group_sorted['Dataset'], group_sorted['DecomposedRatio_ColOrder'],
#                 marker='$', markersize=12, linestyle='-', color='yellow', label=scenario)
#
#     else:
#         ax.plot(group_sorted['Dataset'], group_sorted['DecomposedRatio_ColOrder'],
#                 marker='o', linestyle='-', label=scenario)
#     # Annotate each point with the corresponding k value
#     for idx, row in group_sorted.iterrows():
#         ax.text(row['Dataset'], row['DecomposedRatio_ColOrder'], str(row['k']),
#                 ha='center', va='bottom', fontsize=9)
#
# ax.set_xlabel("Dataset")
# ax.set_ylabel("DecomposedRatio_ColOrder (Compression Ratio)")
# ax.set_title("Best DaviesBouldin Metric: Compression Ratio per Dataset for each FeatureScenario")
# ax.set_yscale('log')  # Set y-axis to log scale
# ax.legend(title="FeatureScenario")
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.savefig("silhouette_best_decomp_ratio_by_feature32H.png")
#
#
# # ==========================================
# # Part 2: Standard Deviation of Compression Ratio
# # ==========================================
#
# # Compute standard deviation of the compression ratio for each (Dataset, FeatureScenario) combination
# std_df = df_valid.groupby(['Dataset', 'FeatureScenario'])['DecomposedRatio_ColOrder'].std().reset_index()
#
# # Create a line plot of standard deviation per Dataset for each FeatureScenario
# fig, ax = plt.subplots(figsize=(12, 8))
#
# for scenario, group in std_df.groupby('FeatureScenario'):
#     group_sorted = group.sort_values('Dataset')
#     if scenario.strip().lower() == "all_features":
#         ax.plot(group_sorted['Dataset'], group_sorted['DecomposedRatio_ColOrder'],
#                 marker='*', markersize=12, linestyle='-', color='red', label=scenario)
#     else:
#         ax.plot(group_sorted['Dataset'], group_sorted['DecomposedRatio_ColOrder'],
#                 marker='o', linestyle='-', label=scenario)
#     # Optionally, annotate the std values
#     for idx, row in group_sorted.iterrows():
#         ax.text(row['Dataset'], row['DecomposedRatio_ColOrder'], f"{row['DecomposedRatio_ColOrder']:.2f}",
#                 ha='center', va='bottom', fontsize=9)
#
# ax.set_xlabel("Dataset")
# ax.set_ylabel("Std of DecomposedRatio_ColOrder")
# ax.set_title("Std of Compression Ratio per Dataset for each FeatureScenario")
# ax.legend(title="FeatureScenario")
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.savefig("std_decomp_ratio_by_feature32H.png")
# ############
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
#
# # Step 1: Read the CSV file
# df = pd.read_csv('/home/jamalids/Downloads/OneDrive_1_17-03-2025/32.csv')
#
# # Step 2: Filter out rows with invalid Silhouette values (i.e. -1 or infinite)
# df_valid = df[(df['DaviesBouldin'] != -1) & (~np.isinf(df['DaviesBouldin']))].copy()
#
# # Step 3: For each (Dataset, FeatureScenario) combination, select the row with the best (maximum) Silhouette value
# best_silhouette_rows = df_valid.loc[df_valid.groupby(['Dataset', 'FeatureScenario'])['DaviesBouldin'].idxmax()]
#
# # Step 4: For each FeatureScenario, compute the standard deviation of the compression ratio
# # (DecomposedRatio_ColOrder) across datasets from the best Silhouette rows.
# std_by_feature = best_silhouette_rows.groupby('FeatureScenario')['DecomposedRatio_ColOrder'].std().reset_index()
#
# # Rename the column for clarity
# std_by_feature.rename(columns={'DecomposedRatio_ColOrder': 'StdCompressionRatio'}, inplace=True)
#
# # Step 5: Plot a bar chart of the average (std) compression ratio per FeatureScenario
# fig, ax = plt.subplots(figsize=(8, 6))
# ax.bar(std_by_feature['FeatureScenario'], std_by_feature['StdCompressionRatio'], color='skyblue')
# ax.set_xlabel("FeatureScenario")
# ax.set_ylabel("Std of DecomposedRatio_ColOrder")
# ax.set_title("Std of Compression Ratio (Best DaviesBouldin) per FeatureScenario")
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.savefig("std_best_silhouette_by_feature32H.png")
#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define allowed FeatureScenarios (case-insensitive)
allowed_scenarios = ['all_features', 'entropy', 'frequency']

# Read the CSV file
#df = pd.read_csv('/home/jamalids/Downloads/OneDrive_1_17-03-2025/32.csv')
#df = pd.read_csv('/home/jamalids/Documents/feature-k/64.csv')
df = pd.read_csv('/home/jamalids/Documents/frame/new2/big-data-compression/modeling/clustering/512/32.csv')


# --------------------------
# Step 1: Compute Best Valid Results per Dataset, FeatureScenario, and Metric
# --------------------------
def geom_mean(series):
    # Compute geometric mean using only positive values (to avoid issues with log)
    positive = series[series > 0]
    if len(positive) == 0:
        return np.nan
    return np.exp(np.mean(np.log(positive)))


# Define metrics and their "best" rule:
# For Silhouette, CalinskiHarabasz, and GapStatistic the best is the maximum.
# For DaviesBouldin the best is the minimum.
metrics = {
    'Silhouette': 'max',
    'DaviesBouldin': 'min',
    'CalinskiHarabasz': 'max',
    'GapStatistic': 'max'
}

# List to store best valid rows for each combination
best_results = []

# Group by Dataset and then by FeatureScenario, filtering by allowed scenarios
for dataset, dgroup in df.groupby('Dataset'):
    for scenario, group in dgroup.groupby('FeatureScenario'):
        if scenario.strip().lower() not in allowed_scenarios:
            continue  # Skip scenarios not in the allowed list
        for metric, rule in metrics.items():
            # Remove rows with invalid metric values (inf or -1)
            valid_group = group[(group[metric] != -1) & (~np.isinf(group[metric]))]
            if valid_group.empty:
                continue  # Skip if no valid rows exist for this combination
            if rule == 'max':
                best_metric_value = valid_group[metric].max()
                best_row = valid_group[valid_group[metric] == best_metric_value].iloc[0]
            else:  # rule == 'min'
                best_metric_value = valid_group[metric].min()
                best_row = valid_group[valid_group[metric] == best_metric_value].iloc[0]
            best_results.append({
                'Dataset': dataset,
                'FeatureScenario': scenario,
                'Metric': metric,
                'BestMetricValue': best_metric_value,
                'DecomposedRatio': best_row['DecomposedRatio_ColOrder'],
                'k': best_row['k'],
                'config':best_row['ClusterConfig']
            })

# Convert the best results to a DataFrame and save to CSV
best_df = pd.DataFrame(best_results)
best_df.to_csv("best_results32.csv", index=False)
print("Saved best valid results to best_results32.csv")

# --------------------------
# Step 2: Plot Best Compression Ratio per Dataset for each Metric
# --------------------------
fig, ax = plt.subplots(figsize=(12, 8))
for metric in metrics.keys():
    subdf = best_df[best_df['Metric'] == metric].sort_values('Dataset')
    ax.plot(subdf['Dataset'], subdf['DecomposedRatio'], marker='o', label=metric)
    # Annotate each point with the corresponding k value
    for idx, row in subdf.iterrows():
        ax.text(row['Dataset'], row['DecomposedRatio'], str(row['k']),
                ha='center', va='bottom', fontsize=9)
ax.set_xlabel("Dataset")
ax.set_ylabel("Best DecomposedRatio_ColOrder")
ax.set_title("Best Compression Ratio per Dataset for Each Metric")
ax.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("best_decomp_ratio.png")
plt.close()

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
gmean_df.to_csv("gmean_results32.csv", index=False)
print("Saved geometric mean results to gmean_results32H.csv")

# --------------------------
# Step 4: Plot Geometric Mean of Compression Ratio per FeatureScenario for each Metric
# --------------------------
fig, ax = plt.subplots(figsize=(12, 8))
for metric in metrics.keys():
    subdf = gmean_df[gmean_df['Metric'] == metric].sort_values('FeatureScenario')
    ax.plot(subdf['FeatureScenario'], subdf['GeomMeanDecomposedRatio'], marker='o', label=metric)
ax.set_xlabel("FeatureScenario")
ax.set_ylabel("Geometric Mean of DecomposedRatio")
ax.set_title("Geometric Mean Compression Ratio per FeatureScenario for Each Metric")
ax.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("gmean_decomp_ratio32.png")
plt.close()

# ================================
# Part 1: Best DaviesBouldin Rows Plot (Filtered to Allowed Scenarios)
# ================================

# Step 1: Filter out invalid DaviesBouldin values (i.e. -1 or infinite)
df_valid = df[(df['DaviesBouldin'] != -1) & (~np.isinf(df['DaviesBouldin']))].copy()
#df_valid = df[(df['Silhouette'] != -1) & (~np.isinf(df['Silhouette']))].copy()

# Step 2: For each (Dataset, FeatureScenario) combination, select the row with the best DaviesBouldin metric
best_silhouette_rows = df_valid.loc[df_valid.groupby(['Dataset', 'FeatureScenario'])['DaviesBouldin'].idxmin()]
#best_silhouette_rows = df_valid.loc[df_valid.groupby(['Dataset', 'FeatureScenario'])['Silhouette'].idxmax()]



# Filter to only include allowed FeatureScenarios
best_silhouette_rows = best_silhouette_rows[best_silhouette_rows['FeatureScenario'].str.lower().isin(allowed_scenarios)]

# Step 3: Create the line plot for best DaviesBouldin rows, one line per FeatureScenario
fig, ax = plt.subplots(figsize=(12, 8))

for scenario, group in best_silhouette_rows.groupby('FeatureScenario'):
    group_sorted = group.sort_values('Dataset')
    lower_scenario = scenario.strip().lower()
    # Customize markers and colors for each allowed scenario
    if lower_scenario == "all_features":
        ax.plot(group_sorted['Dataset'], group_sorted['DecomposedRatio_ColOrder'],
                marker='*', markersize=12, linestyle='-', color='red', label=scenario)
    elif lower_scenario == "entropy":
        ax.plot(group_sorted['Dataset'], group_sorted['DecomposedRatio_ColOrder'],
                marker='^', markersize=12, linestyle='-', color='orange', label=scenario)
    elif lower_scenario == "frequency":
        ax.plot(group_sorted['Dataset'], group_sorted['DecomposedRatio_ColOrder'],
                marker='s', markersize=12, linestyle='-', color='green', label=scenario)

    # Annotate each point with the corresponding k value
    for idx, row in group_sorted.iterrows():
        ax.text(row['Dataset'], row['DecomposedRatio_ColOrder'], str(row['k']),
                ha='center', va='bottom', fontsize=9)

ax.set_xlabel("Dataset")
ax.set_ylabel("DecomposedRatio (Compression Ratio)")
#ax.set_title("Best Silhouette Metric: Compression Ratio per Dataset\nfor Allowed FeatureScenarios")
ax.set_title("Best DaviesBouldin Metric: Compression Ratio per Dataset\nfor Allowed FeatureScenarios")
ax.set_yscale('log')  # Set y-axis to log scale
ax.legend(title="FeatureScenario")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("silhouette_best_decomp_ratio_by_feature32.png")
plt.close()

# ==========================================
# Part 2: Standard Deviation of Compression Ratio (Filtered)
# ==========================================

# Compute standard deviation of the compression ratio for each (Dataset, FeatureScenario) combination
std_df = df_valid.groupby(['Dataset', 'FeatureScenario'])['DecomposedRatio_ColOrder'].std().reset_index()
# Filter to only allowed scenarios
std_df = std_df[std_df['FeatureScenario'].str.lower().isin(allowed_scenarios)]

fig, ax = plt.subplots(figsize=(12, 8))
for scenario, group in std_df.groupby('FeatureScenario'):
    group_sorted = group.sort_values('Dataset')
    lower_scenario = scenario.strip().lower()
    if lower_scenario == "all_features":
        ax.plot(group_sorted['Dataset'], group_sorted['DecomposedRatio_ColOrder'],
                marker='*', markersize=12, linestyle='-', color='red', label=scenario)
    elif lower_scenario == "entropy":
        ax.plot(group_sorted['Dataset'], group_sorted['DecomposedRatio_ColOrder'],
                marker='^', markersize=12, linestyle='-', color='orange', label=scenario)
    elif lower_scenario == "frequency":
        ax.plot(group_sorted['Dataset'], group_sorted['DecomposedRatio_ColOrder'],
                marker='s', markersize=12, linestyle='-', color='green', label=scenario)
    # Annotate the standard deviation values
    for idx, row in group_sorted.iterrows():
        ax.text(row['Dataset'], row['DecomposedRatio_ColOrder'], f"{row['DecomposedRatio_ColOrder']:.2f}",
                ha='center', va='bottom', fontsize=9)

ax.set_xlabel("Dataset")
ax.set_ylabel("Std of DecomposedRatio_ColOrder")
ax.set_title("Standard Deviation of Compression Ratio per Dataset\nfor Allowed FeatureScenarios")
ax.legend(title="FeatureScenario")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("std_decomp_ratio_by_feature32.png")
plt.close()

# ==========================================
# Part 3: Bar Chart of Std Compression Ratio per Allowed FeatureScenario
# ==========================================

# For each FeatureScenario, compute the standard deviation of the compression ratio (from best DaviesBouldin rows)
std_by_feature = best_silhouette_rows.groupby('FeatureScenario')['DecomposedRatio_ColOrder'].std().reset_index()
std_by_feature.rename(columns={'DecomposedRatio_ColOrder': 'StdCompressionRatio'}, inplace=True)
# Filter to allowed scenarios
std_by_feature = std_by_feature[std_by_feature['FeatureScenario'].str.lower().isin(allowed_scenarios)]

fig, ax = plt.subplots(figsize=(8, 6))
ax.bar(std_by_feature['FeatureScenario'], std_by_feature['StdCompressionRatio'], color='skyblue')
ax.set_xlabel("FeatureScenario")
ax.set_ylabel("Std of DecomposedRatio_ColOrder")
ax.set_title("Std of Compression Ratio (Best DaviesBouldin) per Allowed FeatureScenario")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("std_best_silhouette_by_feature32.png")
plt.close()
