import pandas as pd

# 1) Read the CSV
df = pd.read_csv("/home/jamalids/Documents/feature-k/64.csv")

# 2) Quick debug: check columns and unique FeatureScenario values
print("Columns in CSV:", df.columns.tolist())
if "FeatureScenario" in df.columns:
    print("Unique values in 'FeatureScenario':", df["FeatureScenario"].unique())
else:
    print("No 'FeatureScenario' column found. Check your CSV column names!")

# 3) Filter: only rows where FeatureScenario is exactly "is all feature"
#    We also strip whitespace from the column to avoid trailing spaces.
df["FeatureScenario"] = df["FeatureScenario"].astype(str).str.strip()
df_filtered = df[df["FeatureScenario"] == "All_Features"].copy()

if df_filtered.empty:
    print("WARNING: No rows found where FeatureScenario == 'is all feature'!")
    print("Check spelling, spacing, or your CSV data.")
else:
    print(f"Found {len(df_filtered)} rows matching 'is all feature'.")

    # 4) For each dataset, find best Silhouette and best GapStatistic
    datasets = df_filtered["Dataset"].unique()
    cluster_config = {}

    for ds in datasets:
        subset = df_filtered[df_filtered["Dataset"] == ds]

        # a) Row with best (largest) Silhouette
        best_sil_idx = subset["Silhouette"].idxmax()
        best_sil_row = subset.loc[best_sil_idx]

        # b) Row with best (largest) GapStatistic
        best_gap_idx = subset["GapStatistic"].idxmax()
        best_gap_row = subset.loc[best_gap_idx]

        # c) Extract the cluster assignments
        best_sil_clusters = best_sil_row["ClusterConfig"]
        best_gap_clusters = best_gap_row["ClusterConfig"]

        # d) Store them
        cluster_config[ds] = [best_sil_clusters, best_gap_clusters]

    # 5) Generate C++ code
    cpp_output = []
    cpp_output.append('std::map<std::string, std::vector<std::vector<std::vector<int>>>> clusterSolutions = {')

    for ds, (sil_assign, gap_assign) in cluster_config.items():
        line = f'    {{"{ds}", {{\n        {sil_assign},\n        {gap_assign}\n    }} }},'
        cpp_output.append(line)

    cpp_output.append('};')

    # 6) Print the final snippet
    print("\n".join(cpp_output))
