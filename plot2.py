import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
df = pd.read_csv('results_rsim_f32.csv')

# Plotting the bar chart
fig, ax = plt.subplots(figsize=(12, 6))

bar_width = 0.1
index = range(len(df))

# Plotting bars for each category
bar1 = plt.bar(index, df['pattern_comp'], bar_width, label='Pattern Comp')
bar2 = plt.bar([i + bar_width for i in index], df['pattern_comp_dict_zstd'], bar_width, label='Pattern Comp Dict Zstd')
bar3 = plt.bar([i + 2 * bar_width for i in index], df['zstd_comp_size'], bar_width, label='Zstd Comp Size')
bar4 = plt.bar([i + 3 * bar_width for i in index], df['snappy_comp_size'], bar_width, label='Snappy Comp Size')
bar5 = plt.bar([i + 4 * bar_width for i in index], df['pattern_comp_dict_int_zstd'], bar_width, label='Pattern Comp Dict Int Zstd')
bar6 = plt.bar([i + 5 * bar_width for i in index], df['pattern_comp_dict_int_snappy'], bar_width, label='Pattern Comp Dict Int Snappy')
bar7 = plt.bar([i + 6 * bar_width for i in index], df['pattern_comp_dict_int_zstd_delta'], bar_width, label='Pattern Comp Dict Int Zstd Delta')
bar8 = plt.bar([i + 7 * bar_width for i in index], df['pattern_comp_dict_int_snappy_delta'], bar_width, label='Pattern Comp Dict Int Snappy Delta')

plt.xlabel('Dataset Name')
plt.ylabel('Compressed Data Size (bytes)')
plt.title('Comparison of Compressed Data Sizes for Different Methods')

# Set the x-axis to have one central label for the dataset name
central_position = (len(df) * 7 * bar_width) / 2  # Calculate central position
plt.xticks([central_position], ['rsim_f32'])  # Set a single x-axis label

plt.legend()
plt.tight_layout()

# Save the figure
plt.savefig('rsim_f32.jpg', format='jpg', dpi=300)  # Save as JPEG with high resolution

# Show the plot
plt.show()
