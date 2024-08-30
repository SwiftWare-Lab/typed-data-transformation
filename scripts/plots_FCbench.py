import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

PLOTING_DISABLE = False
def plot_bar(values, x_labels, y_label, ax=None):
    if PLOTING_DISABLE:
        return
    ax.bar(np.arange(len(values)), values, color='b')
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=40, ha='right')
    ax.set_ylabel(y_label)
    # Set the x-axis label explicitly to the bottom-left corner
    label = ax.set_xlabel('X-axis Label')


def plot_multiple_lines(series, configs, labels, ax=None, y_label="Entropy", xlabel="Configurations", rotation=40):
    if PLOTING_DISABLE:
        return
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each series with a label
    for values, label in zip(series, labels):
        ax.plot(configs, values, marker='o', linestyle='-', label=label)

    # Set labels and rotation
    ax.set_xlabel(xlabel)
    ax.set_ylabel(y_label)
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs, rotation=rotation, ha='right')

    # Add grid and legend for better readability
    ax.grid(True)
    ax.legend()

dataset_path="/home/jamalids/Documents/compression-part3/logE/"
datasets = [os.path.join(dp, f) for dp, dn, filenames in os.walk(dataset_path) for f in filenames if f.endswith('.csv')]
for dataset in datasets:
    fig, axs = plt.subplots(3, 2, figsize=(20, 20))  # Adjust the subplot grid and figure size as needed
    plt.subplots_adjust(hspace=1)  # Adjust the space between rows

    df_results = pd.read_csv(dataset)
    dataset_name = os.path.basename(dataset).replace('.csv', '')
    com_ratio_cols = [col for col in df_results.columns if col.startswith("com_ratio_b")]
    if com_ratio_cols:  # Ensure the list is not empty
        df_results["max_com_ratio"] = df_results[com_ratio_cols].max(axis=1)
        Decomposion_pattern = df_results["max_com_ratio"].max()
    else:
        Decomposion_pattern = 0  # Fallback value if no columns exist

    # Similarly handle the entropy columns dynamically
    entropy_cols = [col for col in df_results.columns if col.endswith("_entropy")]
    if entropy_cols:  # Ensure the list is not empty
        entropy_full_data = df_results[entropy_cols].max().max()
    else:
        entropy_full_data = 0  # Fallback value if no columns exist

    # Handle t_com_ratio columns
    t_com_ratio_cols = [col for col in df_results.columns if col.startswith("t_com_ratio_b")]
    if t_com_ratio_cols:  # Ensure the list is not empty
        df_results["t-max_com_ratio"] = df_results[t_com_ratio_cols].max(axis=1)
        Decomposion_pattern_with_dict = df_results["t-max_com_ratio"].max()
    else:
        Decomposion_pattern_with_dict = 0  # Fallback value if no columns exist

    comp_ratio_zstd_default = df_results.get("comp_ratio_zstd_default", pd.Series([0])).max()
    comp_ratio_l22 = df_results.get("comp_ratio_l22", pd.Series([0])).max()
    Non_uniform_1x4 = df_results.get("Non_uniform_1x4", pd.Series([0])).max()
    bool_array_size_bits = df_results.get("bool_array_size_bits", pd.Series([0])).max()

    comp_ratio_array = np.array([
        comp_ratio_zstd_default,
        comp_ratio_l22,
        bool_array_size_bits / Non_uniform_1x4 if Non_uniform_1x4 else 0,
        Decomposion_pattern,
        Decomposion_pattern_with_dict
    ])
    plot_bar(comp_ratio_array, ["Zstd Default-3", "Zstd Ultimate-22", "Huffman 1x4", "Decomposion pattern",
                                "Decomposion pattern with dict"], "Compression Ratio", axs[0, 0])

    # Dynamically find all entropy columns
    entropy_cols = [col for col in df_results.columns if col.endswith("_entropy")]

    # Collect the max values of these entropy columns
    entropy_array = np.array([df_results[col].max() for col in entropy_cols])

    # Add the full entropy data if needed
    entropy_array = np.append(entropy_array, entropy_full_data)

    # Create labels dynamically for these entropy columns
    entropy_labels = entropy_cols + ["entropy_full_data"]

    # Plot the entropy array with the corresponding labels
    plot_bar(entropy_array, entropy_labels, "Entropy", axs[0, 1])
    configs = [f"{m} x {n}" for m, n in zip(df_results['M'], df_results['N'])]
    series3 = [df_results[col].tolist() for col in entropy_cols if col in df_results]
    plot_multiple_lines(series3, configs, entropy_cols, ax=axs[1, 0], y_label="Entropy", xlabel="Configurations")
    series1 = [df_results[col].tolist() for col in com_ratio_cols if col in df_results]
    plot_multiple_lines(series1, configs, com_ratio_cols, ax=axs[1, 1], y_label="Com_Ratio", xlabel="Configurations")

    series2 = [df_results[col].tolist() for col in t_com_ratio_cols if col in df_results]
    plot_multiple_lines(series2, configs, t_com_ratio_cols, ax=axs[2, 0], y_label="Com_Ratio_Dic",
                        xlabel="Configurations")

    # Add the dataset name to the last subplot
    axs[2, 1].axis('off')  # Turn off the axis
    axs[2, 1].text(0.5, 0.5, dataset_name, ha='center', va='center', fontsize=18)  # Add the dataset name in the center

    if not PLOTING_DISABLE:
        plt.savefig(f"results/{dataset_name}.png")
        plt.close(fig)