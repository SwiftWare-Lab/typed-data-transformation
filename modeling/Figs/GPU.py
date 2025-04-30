import os
import pandas as pd
import numpy as np
import matplotlib

  # or use another backend if desired
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

directories = ['/mnt/c/Users/jamalids/Downloads/figs/results/GPU-final']
dataframes = []

for directory_path in directories:
    for file in os.listdir(directory_path):
        if file.endswith('.csv'):
            file_path = os.path.join(directory_path, file)
            try:
                # Read CSV file using semicolon as delimiter.
                df = pd.read_csv(file_path, sep=',')
                df.fillna(0, inplace=True)
                # Ensure a "DatasetName" column exists.
                if 'DatasetName' not in df.columns and 'dataset' in df.columns:
                    df.rename(columns={'dataset': 'DatasetName'}, inplace=True)
                if 'DatasetName' not in df.columns:
                    df['DatasetName'] = os.path.basename(file_path).replace('.csv', '')
                dataframes.append(df)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

if not dataframes:
    print("No CSV files were processed. Please check the directories.")
    exit()

combined_df = pd.concat(dataframes, ignore_index=True)
combined_df.fillna(0, inplace=True)
if 'Index' in combined_df.columns:
    combined_df.drop(columns=['Index'], inplace=True)

print("Columns in combined_df:", combined_df.columns.tolist())

all_data_output_path = '/mnt/c/Users/jamalids/Downloads/combined_all_data.csv'
combined_df.to_csv(all_data_output_path, index=False)
print(f'Combined CSV with all data saved to {all_data_output_path}')

# ====================================
# 2. Median Aggregation
# ====================================


# ====================================
# 2. Median Aggregation
# ====================================
required_cols = [
    'Dataset',  'Mode', 'Config',

    'CompThroughput', 'DecompThroughput', 'Ratio'
]
missing_cols = [col for col in required_cols if col not in combined_df.columns]
if missing_cols:
    print("Missing columns in combined_df:", missing_cols)
    exit()

group_columns = ['Dataset', 'Mode',  'Config']
median_df = combined_df.groupby(group_columns, as_index=False).median(numeric_only=True)

median_output_path = '/mnt/c/Users/jamalids/Downloads/combined_median_rows.csv'
median_df.to_csv(median_output_path, index=False)
print(f'Combined CSV with median-based values saved to {median_output_path}')

############################################
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === 1. Load max_compression_throughput_pairs.csv and entropy results ===
df_pairs = pd.read_csv('/mnt/c/Users/jamalids/Downloads/combined_median_rows.csv')
df_pairs.rename(columns={'Dataset': 'DatasetName'}, inplace=True)

df_entropy = pd.read_csv(("/mnt/c/Users/jamalids/Downloads/figs/DatasetIdMapping.csv"))

final_df_A = pd.merge(df_pairs, df_entropy[['DatasetName', 'DatasetID', 'Entropy', 'Domain']], on='DatasetName', how='left')
final_df_A = final_df_A.groupby(['DatasetName', 'Mode'], as_index=False).mean(numeric_only=True)

pivot_cr = final_df_A.pivot(index='DatasetName', columns='Mode', values='Ratio')
pivot_cr = pivot_cr.dropna(subset=['Whole', 'Component'])



pivot_cr = pivot_cr.merge(df_entropy[['DatasetName', 'DatasetID', 'Entropy', 'Domain']], on='DatasetName', how='inner')
#pivot_cr = pivot_cr.sort_values(by=['Domain', 'Entropy']).reset_index(drop=True)
pivot_cr["DatasetID_SortKey"] = pivot_cr["DatasetID"].str.extract(r'D(\d+)').astype(int)
pivot_cr = pivot_cr.sort_values(by="DatasetID_SortKey")


# Compute compression ratio (TDT / Whole)
ratio_series = pivot_cr['Component'] / pivot_cr['Whole']


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set font configuration
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

# === Prepare Data ===
# df_pairs and df_entropy are already merged and prepared
# You have 'pivot_cr' and 'ratio_series' ready

sorted_df = pivot_cr.copy()
sorted_df["Ratio"] = sorted_df["Component"] / sorted_df["Whole"]


# === Step 3: Plot ===
fig, ax = plt.subplots(figsize=(6.8, 3))

# Set your bar color for Zstd plot
#bar_color = "#1f77b4"  # Example color, you can change

#bar_color = "#d62728"
#bar_color ='#17becf'#zlib
#bar_color ='#2ca02c',#bzip
bar_color ='#bcbd22'
bars = ax.bar(sorted_df["DatasetID"], sorted_df["Ratio"],
              width=0.8, color=bar_color, edgecolor="black", linewidth=0.3)

# Labels
#ax.set_xlabel(r"Dataset ID", labelpad=6)
ax.set_ylabel(r"TDT / Standard CR (nvCOMP)", labelpad=6)

# X ticks
if len(sorted_df) > 40:
    step = 5
    ticks = sorted_df["DatasetID"][::step]
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks, rotation=45)
else:
    ax.set_xticks(sorted_df["DatasetID"])
    ax.set_xticklabels(sorted_df["DatasetID"], rotation=45)

# Value labels
# for i, bar in enumerate(bars):
#     height = bar.get_height()
#     if height < 10:
#         ax.text(bar.get_x() + bar.get_width() / 2, height + 0.03, f"{height:.2f}",
#                 ha="center", va="bottom", fontsize=6)

# Remove top and right spines
ax.axhline(y=1, color='black', linestyle='--', linewidth=1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.tight_layout(pad=0.5)

# === Save ===
output_base = "/mnt/c/Users/jamalids/Downloads/gpu"
fig.savefig(f"{output_base}.pdf", bbox_inches="tight", dpi=300)
fig.savefig(f"{output_base}.png", bbox_inches="tight", dpi=300)
plt.close(fig)

print(f"✅ Plot and dataset ID mapping saved successWholey!")

