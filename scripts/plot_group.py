import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset
df = pd.read_csv('results12.csv')
df['dataset_name'] = 'ACSF1'
# Assuming 'group_id' is the column you want to group by
groups = df.groupby('group_id')

# Loop through each group and plot
for group_id, group in groups:
    plt.figure(figsize=(12, 6))

    # Adjusting the bar width for comparison
    bar_width = 0.1
    index = range(len(group))

    # Plotting for each group
    plt.bar(index, group['com_ratio'], bar_width, label='Com Ratio without Dictionary')
    plt.bar([i + bar_width for i in index], group['com_ratio_dict'], bar_width, label='Com Ratio with Dictionary')
    plt.bar([i + 2 * bar_width for i in index], group['com_ratio_tot_d '], bar_width,
            label='Normal Com Ratio with Dictionary')
    plt.bar([i + 3 * bar_width for i in index], group['com_ratio_tot'], bar_width, label='Normal Com Ratio')
    plt.bar([i + 4 * bar_width for i in index], group['comp_ratio_zstd_default'], bar_width,
            label='comp_ratio_zstd_default')
    plt.bar([i + 5 * bar_width for i in index], group['comp_ratio_l22'], bar_width, label='comp_ratio_l22')

    plt.title(f'Compression Ratio for Group ID: {group_id}')
    plt.xlabel('Dataset Name')
    plt.ylabel('Compression Ratio')
    plt.xticks([i + bar_width / 2 for i in index], group['dataset_name'], rotation=45)
    plt.legend()
    plt.tight_layout()

    # Show and save the plot for each group

    plt.savefig(f'CompressionRatio_Group_{group_id}.jpg')
    plt.show()
