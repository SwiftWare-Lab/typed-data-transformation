import pandas as pd

# Load the results CSV
df = pd.read_csv('/mnt/c/Users/jamalids/Downloads/dataset/HPC-64/clustering_compression_results1.csv')

# Keep only the four scenarios you care about
wanted = ["Frequency", "Entropy", "Delta", "All_Features"]
df = df[df['FeatureScenario'].isin(wanted)]

# Identify (Dataset, k) groups with multiple distinct ClusterConfig
grouped = (
    df
    .groupby(['Dataset', 'k'])['ClusterConfig']
    .nunique()
    .reset_index(name='NumConfigs')
)
diff_keys = grouped[grouped['NumConfigs'] > 1][['Dataset', 'k']]

# Extract the different configurations for these keys
diff_configs = (
    df
    .merge(diff_keys, on=['Dataset', 'k'])
    .groupby(['Dataset', 'k'])['ClusterConfig']
    .unique()
    .reset_index()
    .rename(columns={'ClusterConfig': 'ClusterConfigs'})
)

# Save to CSV
output_path = '/mnt/c/Users/jamalids/Downloads/dataset/HPC-64/different_clusterconfigs_filtered.csv'
diff_configs.to_csv(output_path, index=False)

print(f"Done — see: {output_path}")
