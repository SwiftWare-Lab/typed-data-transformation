import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import seaborn as sns


matplotlib.use('Agg')  # Use a non-interactive backend

# Load the processed CSV file
file_path = '/home/jamalids/Documents/results/combine/combined_median.csv'
data = pd.read_csv(file_path)
# Create a new column for combined labels

data["Dataset_ComponentSizes"] = data["DatasetName"].astype(str) + " (" + data["ConfigString"].astype(str) + ")"

# Ensure dataset names are categorical for proper x-axis ordering
data["Dataset_ComponentSizes"] = pd.Categorical(
    data["Dataset_ComponentSizes"], categories=data["Dataset_ComponentSizes"].unique(), ordered=True
)

# Sort the data by dataset for consistent plotting
data = data.sort_values("Dataset_ComponentSizes")

# Melt data for visualization with multiple thread numbers
df_plot = data.melt(
    id_vars=["Dataset_ComponentSizes", "Threads", "RunType"],
    value_vars=["CompressionThroughput", "DecompressionThroughput"],
    var_name="Metric",
    value_name="Throughput"
)

# Map CompressionThroughput, DecompressionThroughput to labels
df_plot["Method"] = df_plot.apply(
    lambda row: f"Parallel (Th:{int(row['Threads'])})" if row["RunType"] == "Parallel" else "Full",
    axis=1
)

# Define number of datasets per figure
datasets_per_figure = 6  # Number of datasets to include per figure
num_figures = len(data["Dataset_ComponentSizes"].unique()) // datasets_per_figure + 1

# Generate multiple figures
for fig_idx in range(num_figures):
    fig, axs = plt.subplots(datasets_per_figure, 2, figsize=(15, 5 * datasets_per_figure), sharex=False)

    # Ensure axs is always iterable
    axs = np.array(axs).reshape(datasets_per_figure, 2)

    # Define dataset range for this figure
    start = fig_idx * datasets_per_figure
    end = min((fig_idx + 1) * datasets_per_figure, len(data["Dataset_ComponentSizes"].unique()))

    # Select only relevant datasets for this figure
    dataset_subset = data["Dataset_ComponentSizes"].unique()[start:end]

    for i, dataset in enumerate(dataset_subset):
        # Filter data for the current dataset
        subset_ds = df_plot[df_plot["Dataset_ComponentSizes"] == dataset]

        # Compression Throughput
        # sns.barplot(x="Threads", y="Throughput", hue="Method",
        #             data=subset_ds[subset_ds["Metric"] == "CompressionThroughput"], ax=axs[i, 0])
        sns.barplot(x="Threads", y="Throughput", hue="Method",
                    data=subset_ds[subset_ds["Metric"] == "CompressionThroughput"], ax=axs[i, 0], ci=None)

        axs[i, 0].set_title(f"Compression Throughput - {dataset}")
        axs[i, 0].set_ylabel("Compression (GB/s)")
        axs[i, 0].set_xlabel("Threads")

        # Annotate values on bars
        for bar, val in zip(axs[i, 0].containers[0], subset_ds[subset_ds["Metric"] == "CompressionThroughput"]["Throughput"]):
            axs[i, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{val:.2f}",
                           ha="center", va="bottom", fontsize=8)

        # Decompression Throughput
        # sns.barplot(x="Threads", y="Throughput", hue="Method",
        #             data=subset_ds[subset_ds["Metric"] == "DecompressionThroughput"], ax=axs[i, 1])
        sns.barplot(x="Threads", y="Throughput", hue="Method",
                    data=subset_ds[subset_ds["Metric"] == "DecompressionThroughput"], ax=axs[i, 1], ci=None)
        axs[i, 1].set_title(f"Decompression Throughput - {dataset}")
        axs[i, 1].set_ylabel("Decompression (GB/s)")
        axs[i, 1].set_xlabel("Threads")

        # Annotate values on bars
        # for bar, val in zip(axs[i, 1].containers[0], subset_ds[subset_ds["Metric"] == "DecompressionThroughput"]["Throughput"]):
        #     axs[i, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{val:.2f}",
        #                    ha="center", va="bottom", fontsize=8)

    # Adjust layout for better readability
    plt.tight_layout()

    # Save each figure with a unique filename
    output_path = f"/home/jamalids/Documents/results/combine/compression_analysis_paired_threads_{fig_idx + 1}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")

    print(f"Figure {fig_idx + 1} saved to {output_path}")
