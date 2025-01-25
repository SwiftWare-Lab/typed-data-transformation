import pandas as pd
import os
import matplotlib.pyplot as plt
import math

# Load the combined results file
file_path = '/home/jamalids/Documents/2D/CR-Ct-DT/python-results/logs/32/compression_results/combined_top_5_compression_results.csv'
data = pd.read_csv(file_path)

# Output directory for plots
output_plots_dir = '/home/jamalids/Documents/2D/CR-Ct-DT/python-results/logs/32/compression_results/plots'
os.makedirs(output_plots_dir, exist_ok=True)

# Replace variation names for easier reading
data['variation'] = data['variation'].str.replace(' compressed size (B)', '', regex=False)

# Get unique datasets
datasets = data['dataset name'].unique()

# Create a plot for each dataset
for dataset in datasets:
    dataset_data = data[data['dataset name'] == dataset]

    # Get unique variations for the current dataset
    variations = dataset_data['variation'].unique()

    # Determine grid size for subplots
    n_variations = len(variations)
    n_rows = math.ceil(math.sqrt(n_variations))  # Rows for a roughly square grid
    n_cols = math.ceil(n_variations / n_rows)   # Columns to fit all subplots

    # Create a figure with subplots
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(16, 12),
        constrained_layout=True
    )
    axes = axes.flatten()  # Flatten axes array for easier indexing

    for i, variation in enumerate(variations):
        # Get data for the current variation
        variation_data = dataset_data[dataset_data['variation'] == variation]

        # Select the top 3 compression ratios for this variation
        top_3 = variation_data.nlargest(3, 'compression ratio')

        # Plot on the corresponding subplot
        ax = axes[i]
        bars = ax.bar(
            top_3['decomposition'],
            top_3['compression ratio'],
            color='skyblue'
        )

        # Annotate each bar with its compression ratio
        for bar, ratio in zip(bars, top_3['compression ratio']):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{ratio:.2f}",
                ha='center',
                va='bottom',
                fontsize=8
            )

        # Set the title and labels for each subplot
        ax.set_title(variation, fontsize=10)
        ax.set_xlabel('Decomposition', fontsize=8)
        ax.set_ylabel('Compression Ratio', fontsize=8)
        ax.tick_params(axis='x', labelrotation=45, labelsize=7)
        ax.tick_params(axis='y', labelsize=7)

    # Turn off unused subplots if there are fewer variations than the grid size
    for j in range(n_variations, len(axes)):
        axes[j].axis('off')

    # Set the main title for the figure
    fig.suptitle(f'Top 3 Compression Ratios for Each Variation in {dataset}', fontsize=16)

    # Save the plot for this dataset
    output_file = os.path.join(output_plots_dir, f'{dataset}_top_3_compression_ratios_by_variation.png')
    plt.savefig(output_file)
    plt.close(fig)  # Close the figure to avoid overlapping plots

    print(f"Plot saved for dataset '{dataset}' at: {output_file}")

# Display message after processing all datasets
print(f"All plots saved to: {output_plots_dir}")
