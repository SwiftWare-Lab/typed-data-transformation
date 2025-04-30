
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # For non-interactive backends (no GUI)
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib as mpl

# === Match Overleaf font and style ===
mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Latin Modern Roman", "Times", "DejaVu Serif"],
    "font.size": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def process_and_plot(directory_path, save_prefix, col_name, bar_color="#4E79A7"):

    # Step 1: Load CSVs
    directory = Path(directory_path)
    dataframes = []

    for file in directory.glob("*.csv"):
        try:
            df = pd.read_csv(file)
            df.fillna(0, inplace=True)
            # Fix column names in your main DataFrame
            # Standardize column name
            if 'decomposed xor col-order ratio' in df.columns:
                df.rename(
                    columns={'decomposed xor col-order ratio': 'decomposed xor compression ratio'},
                    inplace=True)
            if 'standard xor ratio' in df.columns:
                df.rename(
                    columns={'standard xor ratio': 'standard xor compression ratio'},
                    inplace=True)

            if "DatasetName" not in df.columns and "dataset name" in df.columns:
                df.rename(columns={"dataset name": "DatasetName"}, inplace=True)

            if "DatasetName" not in df.columns:
                df["DatasetName"] = file.stem

            dataframes.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")

    if not dataframes:
        print("No CSV files found.")
        return

    combined_df = pd.concat(dataframes, ignore_index=True)
    if "Index" in combined_df.columns:
        combined_df.drop(columns=["Index"], inplace=True)

    combined_path = Path(f"/mnt/c/Users/jamalids/Downloads/figs/combined_all_{save_prefix}.csv")
    combined_df.to_csv(combined_path, index=False)

    # Step 2: Load meta
    meta_path = Path("/mnt/c/Users/jamalids/Downloads/figs/DatasetIdMapping.csv")
    main_df = pd.read_csv(combined_path)
    meta_df = pd.read_csv(meta_path)

    merged_df = pd.merge(main_df, meta_df, on="DatasetName")
    ####################################################
    if "huffman" in col_name.lower():
        if "decomposed huffman_compress col-order ratio" in merged_df.columns:
            merged_df.rename(columns={
                "decomposed huffman_compress col-order ratio": "decomposed huffman compression ratio",
                "standard huffman_compress ratio": "standard huffman compression ratio"

            }, inplace=True)
            print("✅ Renamed Huffman decomposed column successfully.")

    merged_df["Ratio"] = merged_df[f"decomposed {col_name} compression ratio"] / merged_df[f"standard {col_name} compression ratio"]

    # #sorted_df = merged_df.sort_values(by=["Domain", "Entropy"])
    # sorted_df = merged_df.sort_values(by="DatasetID")
    sorted_df = merged_df.copy()
    sorted_df["DatasetID_SortKey"] = sorted_df["DatasetID"].str.extract(r'D(\d+)').astype(int)
    sorted_df = sorted_df.sort_values(by="DatasetID_SortKey")

    # === Special fix for Huffman case ===

    # Step 3: Plot
    fig, ax = plt.subplots(figsize=(6.8, 3))
    bars = ax.bar(sorted_df["DatasetID"], sorted_df["Ratio"],
                  width=0.8, color=bar_color, edgecolor="black", linewidth=0.3)

    # ax.set_xlabel(r"Dataset ID", labelpad=6)
    ax.set_ylabel(rf"TDT / Standard CR ({col_name.upper()})", labelpad=6)

    # X ticks
    if len(sorted_df) > 40:
        step = 5
        ticks = sorted_df["DatasetID"][::step]
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticks, rotation=45)
    else:
        ax.set_xticks(sorted_df["DatasetID"])
        ax.set_xticklabels(sorted_df["DatasetID"], rotation=45)
    # Add horizontal red line at y=1
    ax.axhline(y=1, color='red', linestyle='--', linewidth=1)
    # # Value labels (optional)
    # for i, bar in enumerate(bars):
    #     height = bar.get_height()
    #     if height < 10:
    #         ax.text(bar.get_x() + bar.get_width() / 2, height + 0.03, f"{height:.2f}",
    #                 ha="center", va="bottom", fontsize=7)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout(pad=0.5)

    output_base = f"/mnt/c/Users/jamalids/Downloads/figs/{save_prefix}"
    fig.savefig(f"{output_base}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(f"{output_base}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)

# === Run for XOR ===
# XOR
process_and_plot(
    directory_path="/mnt/c/Users/jamalids/Downloads/figs/results/xor1",
    save_prefix="xor",
    col_name="xor",
    bar_color="#7f7f7f"
)

# # RLE
# process_and_plot(
#     directory_path="/mnt/c/Users/jamalids/Downloads/figs/results/rle",
#     save_prefix="rle",
#     col_name="rle",
#     bar_color="#F28E2B"
# )

# Huffman
process_and_plot(
    directory_path="/mnt/c/Users/jamalids/Downloads/figs/results/huffman",
    save_prefix="huffman",
    col_name="huffman",
    bar_color="#8c564b"
)
