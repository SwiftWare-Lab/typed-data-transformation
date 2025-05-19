# Full code to do the conversion from beginning
import pandas as pd

# Load the uploaded file
file_path = r"C:\Users\jamalids\Downloads\dataset\convert-64.csv"
df = pd.read_csv(file_path)

# Step 1: Filter rows where FeatureScenario == 'Frequency' and DaviesBouldin != -1
#filtered_df = df[(df['FeatureScenario'] == 'Frequency') & (df['DaviesBouldin'] != -1)]
filtered_df = df[(df['FeatureScenario'] == 'All_Features') & (df['GapStatistic'] != -1)]
# Step 2: For each Dataset, select the row with minimum DaviesBouldin
best_config_df = filtered_df.loc[filtered_df.groupby('Dataset')['GapStatistic'].idxmax()]

# Step 3: Select necessary columns
result_df = best_config_df[['Dataset', 'ClusterConfig']].copy()

# Step 4: Convert ClusterConfig from (1,2)|(3)|(4) format to {{1,2}, {3}, {4}}
def convert_cluster_config(config_string):
    parts = config_string.split('|')
    formatted_parts = []
    for part in parts:
        part = part.strip('()')
        numbers = part.split(',')
        numbers = [num.strip() for num in numbers]
        formatted_part = '{' + ','.join(numbers) + '}'
        formatted_parts.append(formatted_part)
    return '{' + ', '.join(formatted_parts) + '}'

result_df['ConvertedClusterConfig'] = result_df['ClusterConfig'].apply(convert_cluster_config)
result_df.to_csv(r'C:\Users\jamalids\Downloads\64-config-g.csv', index=False)