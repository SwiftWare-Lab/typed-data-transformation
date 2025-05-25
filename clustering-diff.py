import pandas as pd

# Load the results CSV
df = pd.read_csv( '/mnt/c/Users/jamalids/Downloads/dataset/OBS/clustering_compression_results1.csv')

# Identify (Dataset, FeatureScenario, k) groups with multiple distinct ClusterConfig
grouped = df.groupby(['Dataset', 'k'])['ClusterConfig'].nunique().reset_index(name='NumConfigs')
#diff_keys = grouped[grouped['NumConfigs'] > 1][['Dataset', 'FeatureScenario', 'k']]
#
# # Extract the different configurations for these keys
# diff_configs = (
#     df.merge(diff_keys, on=['Dataset', 'FeatureScenario', 'k'])
#       .groupby(['Dataset', 'FeatureScenario', 'k'])['ClusterConfig']
#       .unique()
#       .reset_index()
#       .rename(columns={'ClusterConfig': 'ClusterConfigs'})
# )
#
# # Save to CSV
# output_path = '/mnt/c/Users/jamalids/Downloads/dataset/OBS/different_clusterconfigs.csv'
# diff_configs.to_csv(output_path, index=False)
