import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Load the CSV Data ---
data = pd.read_csv('/home/jamalids/Documents/combined_all_data.csv')

# Values of Threads to analyze
thread_values = [1, 8, 16]

# Create a figure with 3 subplots (one for each thread value)
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6), sharey=False)

for idx, thread_val in enumerate(thread_values):
    # --- 2. Filter the Data for the RunType "Decompose_Chunk_Parallel" and the specified thread count ---
    filtered = data[
        (data['RunType'] == 'Decompose_Chunk_Parallel') &
        (data['Threads'] == thread_val)
        ]

    # --- 3. Group by BlockSize and Compute Mean Metrics ---
    grouped = filtered.groupby('BlockSize').agg({
        'CompressionThroughput': 'mean',
        'DecompressionThroughput': 'mean',
        'CompressionRatio': 'mean'
    }).reset_index()

    # If there's a chance the subset is empty or too small, handle that:
    if grouped.empty:
        axes[idx].set_title(f'Threads = {thread_val}\n(No Data)')
        continue

    # Sort by BlockSize (for clarity in plotting)
    grouped.sort_values('BlockSize', inplace=True)

    # --- 4. Normalize the Metrics to [0, 1] ---
    grouped['CompressionThroughput_norm'] = (
            (grouped['CompressionThroughput'] - grouped['CompressionThroughput'].min()) /
            (grouped['CompressionThroughput'].max() - grouped['CompressionThroughput'].min())
    )
    grouped['DecompressionThroughput_norm'] = (
            (grouped['DecompressionThroughput'] - grouped['DecompressionThroughput'].min()) /
            (grouped['DecompressionThroughput'].max() - grouped['DecompressionThroughput'].min())
    )
    grouped['CompressionRatio_norm'] = (
            (grouped['CompressionRatio'] - grouped['CompressionRatio'].min()) /
            (grouped['CompressionRatio'].max() - grouped['CompressionRatio'].min())
    )

    # --- 5. Combine Throughput and Compression Ratio Metrics ---
    # Average the two throughput metrics
    grouped['Throughput_norm'] = (
            (grouped['CompressionThroughput_norm'] + grouped['DecompressionThroughput_norm']) / 2
    )
    # Create a combined score
    grouped['Combined_Score'] = grouped['Throughput_norm'] + grouped['CompressionRatio_norm']

    # --- 6. Identify the Best BlockSize Based on the Highest Combined Score ---
    best_row = grouped.loc[grouped['Combined_Score'].idxmax()]
    best_blocksize = best_row['BlockSize']

    print(f"[Threads = {thread_val}] Best BlockSize:", best_blocksize)
    print("Metrics for Best BlockSize:")
    print(best_row)
    print("-----------")

    # --- 7. Plot the Normalized Metrics with BlockSize as Text Labels on the X-Axis ---
    x = np.arange(len(grouped))

    axes[idx].plot(x, grouped['Throughput_norm'], 'g-o', label='Throughput (norm)')
    axes[idx].plot(x, grouped['CompressionRatio_norm'], 'b-o', label='Compression Ratio (norm)')
    axes[idx].plot(x, grouped['Combined_Score'], 'r-o', label='Combined Score')

    axes[idx].set_xlabel('BlockSize')
    if idx == 0:  # Label the y-axis on the first subplot
        axes[idx].set_ylabel('Normalized Value')
    axes[idx].set_title(f'Threads = {thread_val}')

    # Set x-ticks using the index positions and label them with exact BlockSize values
    axes[idx].set_xticks(x)
    axes[idx].set_xticklabels(grouped['BlockSize'].astype(str), rotation=45, ha='right')

# Put a legend on the last subplot (or overall)
axes[-1].legend(loc='best')

plt.tight_layout()

# Save and close the plot
output_plot_path = '/home/jamalids/Documents/plot_best_blocksize_three_subplots.png'
plt.savefig(output_plot_path)
plt.close()

print(f"Plot with three subplots (Threads=1,8,16) saved to {output_plot_path}")
###################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Load the CSV Data and filter for the desired RunType ---
data = pd.read_csv('/home/jamalids/Documents/combined_all_data.csv')
data = data[data['RunType'] == 'Decompose_Chunk_Parallel']

# Define the thread values (datasets)
thread_values = [1, 8, 16]

# --- 2. Create a figure with 3 subplots (one for each thread value) ---
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

for idx, thread_val in enumerate(thread_values):
    # Filter data for the specific thread count
    df_thread = data[data['Threads'] == thread_val]

    # Group data by BlockSize and compute max and min for each metric
    grouped = df_thread.groupby('BlockSize').agg({
        'CompressionThroughput': ['max', 'min'],
        'DecompressionThroughput': ['max', 'min'],
        'CompressionRatio': ['max', 'min']
    })

    # If no data available for this thread, show a message
    if grouped.empty:
        axes[idx].set_title(f"Threads = {thread_val} (No Data)")
        continue

    # Flatten the MultiIndex columns
    grouped.columns = ['_'.join(col) for col in grouped.columns]
    grouped = grouped.reset_index()
    grouped.sort_values('BlockSize', inplace=True)

    # Convert BlockSize to string for better x-axis labels
    block_sizes = grouped['BlockSize'].astype(str)

    ax = axes[idx]

    # Plot max and min for Compression Throughput
    ax.plot(block_sizes, grouped['CompressionThroughput_max'], marker='o', label='CT Max')
    ax.plot(block_sizes, grouped['CompressionThroughput_min'], marker='o', label='CT Min')

    # Plot max and min for Decompression Throughput
    ax.plot(block_sizes, grouped['DecompressionThroughput_max'], marker='o', label='DT Max')
    ax.plot(block_sizes, grouped['DecompressionThroughput_min'], marker='o', label='DT Min')

    # Plot max and min for Compression Ratio
    ax.plot(block_sizes, grouped['CompressionRatio_max'], marker='o', label='CR Max')
    ax.plot(block_sizes, grouped['CompressionRatio_min'], marker='o', label='CR Min')

    ax.set_title(f"Threads = {thread_val}")
    ax.set_xlabel("BlockSize")
    ax.set_ylabel("Metric Value")
    ax.tick_params(axis='x', rotation=45)
    ax.legend(loc='best')

plt.tight_layout()

# Save the figure
output_path = '/home/jamalids/Documents/three_subplots_max_min_by_blocksize.png'
plt.savefig(output_path)
plt.close()

print(f"Plot saved to {output_path}")
