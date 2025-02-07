import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import FuncFormatter

# Use a non-interactive backend if needed.
matplotlib.use("Agg")

# ------------------------------
# Specify the folder containing your CSV files.
# ------------------------------
folder_path = "/home/jamalids/Documents/results1"

# List all CSV files in the folder (include full paths).
dataset_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".csv")]

if not dataset_files:
    print("No CSV files found in the specified folder.")
    exit()

for file_path in dataset_files:
    # Extract the dataset name (without extension)
    dataset_name = os.path.splitext(os.path.basename(file_path))[0]
    print(f"\nProcessing dataset: {dataset_name}")

    # ------------------------------
    # Load and Prepare Data
    # ------------------------------
    df = pd.read_csv(file_path, delimiter=";", engine="python")
    print("Columns in CSV:", df.columns)

    # Force ConfigString to a string and strip extra spaces.
    df["ConfigString"] = df["ConfigString"].astype(str).str.strip()

    # Replace missing string values ("nan" and "None") with "{ [0] }"
    df.loc[df["ConfigString"].isin(["nan", "None"]), "ConfigString"] = "{ [0] }"

    # Replace "Full" RunType BlockSize with TotalValues * 4 (assuming float32)
    df.loc[df["RunType"] == "Full", "BlockSize"] = df["TotalValues"] * 4

    # Convert relevant columns to numeric values
    df["BlockSize"] = pd.to_numeric(df["BlockSize"], errors="coerce")
    df["TotalTimeCompressed"] = pd.to_numeric(df["TotalTimeCompressed"], errors="coerce")
    df["TotalTimeDecompressed"] = pd.to_numeric(df["TotalTimeDecompressed"], errors="coerce")
    df["CompressionRatio"] = pd.to_numeric(df["CompressionRatio"], errors="coerce")

    # ------------------------------
    # Ensure "Full" appears in all block size categories
    # ------------------------------
    unique_block_sizes = sorted(df["BlockSize"].dropna().unique())  # Remove NaN before mapping

    # Get all "Full" rows
    full_rows = df[df["RunType"] == "Full"]

    # Duplicate "Full" rows for every block size to make it appear across all sizes
    full_expanded = pd.concat(
        [full_rows.assign(BlockSize=bs) for bs in unique_block_sizes],
        ignore_index=True
    )

    # Append the expanded "Full" rows back to the dataset
    df = pd.concat([df, full_expanded], ignore_index=True)

    # ------------------------------
    # Prepare Grouping and Categories
    # ------------------------------
    categories = [str(int(x)) for x in unique_block_sizes]
    mapping = {size: i for i, size in enumerate(unique_block_sizes)}

    # Create a combined grouping field from RunType and ConfigString
    df["RunType1"] = df["RunType"].astype(str) + " " + df["ConfigString"].astype(str)
    grouped = df.groupby("RunType1")

    # ------------------------------
    # Define a function to plot safely
    # ------------------------------
    def safe_plot(df_group, x_col, y_col, title, xlabel, ylabel, filename):
        """ Ensure x and y values have the same shape before plotting. """
        plt.figure(figsize=(10, 6))
        for label, group in df_group:
            valid_data = group.dropna(subset=[x_col, y_col])  # Drop NaN values
            if valid_data.empty:
                continue  # Skip empty groups

            x = valid_data[x_col].map(mapping).dropna().values  # Ensure valid x values
            y = valid_data[y_col].values

            if len(x) == len(y) and len(x) > 0:  # Ensure matching dimensions
                plt.plot(x, y, marker="o", linestyle="-", label=label)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.xticks(np.arange(len(categories)), categories)
        plt.tight_layout()

        plt.savefig(filename)
        plt.close()

    # ------------------------------
    # Plot 1: Compression Time vs Block Size
    # ------------------------------
    safe_plot(
        grouped,
        "BlockSize",
        "TotalTimeCompressed",
        f"Compression Time vs Block Size ({dataset_name})",
        "Block Size (bytes)",
        "Compression Time (seconds)",
        os.path.join(folder_path, f"{dataset_name}_compression_time_vs_block_size.png"),
    )

    # ------------------------------
    # Plot 2: Decompression Time vs Block Size
    # ------------------------------
    safe_plot(
        grouped,
        "BlockSize",
        "TotalTimeDecompressed",
        f"Decompression Time vs Block Size ({dataset_name})",
        "Block Size (bytes)",
        "Decompression Time (seconds)",
        os.path.join(folder_path, f"{dataset_name}_decompression_time_vs_block_size.png"),
    )

    # ------------------------------
    # Plot 3: Compression Ratio vs Block Size
    # ------------------------------
    safe_plot(
        grouped,
        "BlockSize",
        "CompressionRatio",
        f"Compression Ratio vs Block Size ({dataset_name})",
        "Block Size (bytes)",
        "Compression Ratio",
        os.path.join(folder_path, f"{dataset_name}_compression_ratio_vs_block_size.png"),
    )

    print(f"Plots for dataset '{dataset_name}' saved successfully.\n")
