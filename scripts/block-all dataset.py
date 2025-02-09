import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from matplotlib.ticker import FuncFormatter

# Use a non-interactive backend if needed (e.g. on headless systems).
matplotlib.use("Agg")

# ------------------------------
# Specify the folder containing your CSV files.
# ------------------------------
folder_path = "/home/jamalids/Documents"
#folder_path = "/home/jamalids/Documents"

# List all CSV files in the folder (include full paths).
dataset_files = [
    os.path.join(folder_path, f)
    for f in os.listdir(folder_path)
    if f.endswith(".csv")
]

if not dataset_files:
    print("No CSV files found in the specified folder.")
    exit()

for file_path in dataset_files:
    dataset_name = os.path.splitext(os.path.basename(file_path))[0]
    print(f"\nProcessing dataset: {dataset_name}")

    # ------------------------------
    # Load and Prepare Data
    # ------------------------------
    df = pd.read_csv(file_path, delimiter=";", engine="python")
    print("Columns in CSV:", df.columns)

    # 1) Clean/prepare ConfigString
    if "ConfigString" in df.columns:
        df["ConfigString"] = df["ConfigString"].astype(str).str.strip()
        df.loc[df["ConfigString"].isin(["nan", "None"]), "ConfigString"] = "{ [0] }"

    # 2) If BlockSize is "{ [0] }", remove those rows (if you still want that logic)
    if "BlockSize" in df.columns:
        df = df[df["BlockSize"] != "{ [0] }"]

    # -----------------------------------------------------
    # 3) Replace "Full" RunType BlockSize with TotalValues*4
    #    (assuming float32 => 4 bytes per value).
    # -----------------------------------------------------
    # Only do this if "RunType" and "TotalValues" exist in your CSV.
    if "RunType" in df.columns and "TotalValues" in df.columns:
        # For rows where RunType == "Full", set BlockSize = TotalValues * 4
        df.loc[(df["RunType"] == "Full") | (df["RunType"] == "Decompose_NonChunked"), "BlockSize"] = df["TotalValues"] * 4


    # -----------------------------------------------------
    # 4) Convert your known numeric columns
    # -----------------------------------------------------
    numeric_cols = [
        "Threads",
        "BlockSize",
        "TotalTimeCompressed",
        "TotalTimeDecompressed",
        "CompressionRatio",
        "CompressionThroughput",
        "DecompressionThroughput",
        "TotalValues"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 5) Drop rows missing any of these "required" columns
    required_cols = [
        "Threads",
        "BlockSize",
        "ConfigString",
        "RunType",
        "TotalTimeCompressed",
        "TotalTimeDecompressed",
        "CompressionRatio",
    ]
    df_numeric = df.dropna(subset=required_cols)

    # ------------------------------
    # Compute median, plot, etc. (as before)
    # ------------------------------
    group_cols = ["Threads", "BlockSize", "ConfigString", "RunType"]
    agg_dict = {
        "TotalTimeCompressed": "median",
        "TotalTimeDecompressed": "median",
        "CompressionRatio": "median",
        "CompressionThroughput": "median",
        "DecompressionThroughput": "median",
    }

    df_median = df_numeric.groupby(group_cols, as_index=False).agg(agg_dict)

    # Build a label for plotting
    df_median["GroupLabel"] = (
        df_median["Threads"].astype(int).astype(str)
        + " Thr, "
        + df_median["ConfigString"].astype(str)
        + ", "
        + df_median["RunType"].astype(str)
    )

    # Sort / map your block sizes to x-axis
    unique_block_sizes = sorted(df_median["BlockSize"].unique())
    categories = [str(int(x)) for x in unique_block_sizes]
    mapping = {size: i for i, size in enumerate(unique_block_sizes)}

    # Make the figure
    # After computing df_median, preparing unique_block_sizes, etc.:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    ax_ctime, ax_dtime, ax_cratio = axes

    grouped = df_median.groupby("GroupLabel")

    for label, group in grouped:
        # Determine if this group is "Full" or "Decompose_NonChunked"
        is_full = (group["RunType"] == "Full").all()
        is_decompose_nonchunked = (group["RunType"] == "Decompose_NonChunked").all()

        if is_full:
            # Get the (single) median values from that group.
            comp_time = group["TotalTimeCompressed"].iloc[0]
            decomp_time = group["TotalTimeDecompressed"].iloc[0]
            ratio = group["CompressionRatio"].iloc[0]

            # Plot horizontal lines across the entire x-range.
            ax_ctime.axhline(y=comp_time, label=label, color="red", linestyle="--", linewidth=2)
            ax_dtime.axhline(y=decomp_time, label=label, color="red", linestyle="--", linewidth=2)
            ax_cratio.axhline(y=ratio, label=label, color="red", linestyle="--", linewidth=2)

        elif is_decompose_nonchunked:
            comp_time = group["TotalTimeCompressed"].iloc[0]
            decomp_time = group["TotalTimeDecompressed"].iloc[0]
            ratio = group["CompressionRatio"].iloc[0]

            ax_ctime.axhline(y=comp_time, label=label, color="black", linestyle="--", linewidth=2)
            ax_dtime.axhline(y=decomp_time, label=label, color="black", linestyle="--", linewidth=2)
            ax_cratio.axhline(y=ratio, label=label, color="black", linestyle="--", linewidth=2)

        else:
            # Normal runs: plot versus numeric BlockSize.
            x = group["BlockSize"].map(mapping).values
            ax_ctime.plot(x, group["TotalTimeCompressed"], marker="o", linestyle="-", label=label)
            ax_dtime.plot(x, group["TotalTimeDecompressed"], marker="o", linestyle="-", label=label)
            ax_cratio.plot(x, group["CompressionRatio"], marker="o", linestyle="-", label=label)
    # Now finalize each subplot as before:
    ax_ctime.set_title(f"Compression Time vs Block Size\n({dataset_name})")
    ax_ctime.set_xlabel("Block Size (bytes)")
    ax_ctime.set_ylabel("Median Compression Time (s)")
    ax_ctime.set_xticks(np.arange(len(categories)))
    ax_ctime.set_xticklabels(categories, rotation=45)
    ax_ctime.grid(True)

    ax_dtime.set_title(f"Decompression Time vs Block Size\n({dataset_name})")
    ax_dtime.set_xlabel("Block Size (bytes)")
    ax_dtime.set_ylabel("Median Decompression Time (s)")
    ax_dtime.set_xticks(np.arange(len(categories)))
    ax_dtime.set_xticklabels(categories, rotation=45)
    ax_dtime.grid(True)

    ax_cratio.set_title(f"Compression Ratio vs Block Size\n({dataset_name})")
    ax_cratio.set_xlabel("Block Size (bytes)")
    ax_cratio.set_ylabel("Median Compression Ratio")
    ax_cratio.set_xticks(np.arange(len(categories)))
    ax_cratio.set_xticklabels(categories, rotation=45)
    ax_cratio.grid(True)

    # Put a single legend on, for example, the first subplot:
    ax_ctime.legend(loc="best", fontsize="small")

    plt.tight_layout()
    summary_plot_path = os.path.join(folder_path, f"{dataset_name}_summary_plots.png")
    plt.savefig(summary_plot_path)
    plt.close()
