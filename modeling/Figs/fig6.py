# import matplotlib as mpl
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import seaborn as sns
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score
# from scipy.stats import spearmanr
#
# # 🌟 Set global font and plot style (VLDB style)
# mpl.rcParams.update({
#     "text.usetex": False,
#     "font.family": "serif",
#     "font.serif": ["Times New Roman", "Latin Modern Roman", "Times", "DejaVu Serif"],
#     "font.size": 9,
#     "axes.labelsize": 9,
#     "xtick.labelsize": 8,
#     "ytick.labelsize": 8,
#     "legend.fontsize": 8,
#     "pdf.fonttype": 42,
#     "ps.fonttype": 42,
# })
#
# # --- Helper functions ---
# def compute_line_formula(x, y):
#     model = LinearRegression()
#     model.fit(np.array(x).reshape(-1, 1), y)
#     return model.coef_[0], model.intercept_
#
# def get_slope_for_all_pairs(data_frame):
#     numerical_columns = data_frame.select_dtypes(include=["float64", "int"]).columns
#     results = {}
#     for i, col1 in enumerate(numerical_columns):
#         for col2 in numerical_columns[i + 1:]:
#             x = data_frame[col1]
#             y = data_frame[col2]
#             if x.isnull().any() or y.isnull().any():
#                 continue
#             slope, intercept = compute_line_formula(x, y)
#             r2 = r2_score(x, y)
#             s_r2 = spearmanr(x, y)
#             results[(col1, col2)] = {"slope": slope, "intercept": intercept, "r2": r2, "spearman_r2": s_r2}
#     return results
#
# def average_entropy_plot(data):
#     # Prepare arrays
#     datasets = data['dataset_name'].tolist()
#     entropy_components = {f"Entropy k{i}": {"orig": [], "byte_shuffle": [], "tdt": []} for i in range(5)}
#
#     for _, row in data.iterrows():
#         for i in range(5):
#             orig = row[f'orig_k{i}']
#             bs = row[f'tdt_k{i}']
#             custom = row[f'custom_k{i}']
#             after_custom = row[f'tdt_after_custom_k0']
#             min_tdt = np.min([custom, after_custom])
#             entropy_components[f"Entropy k{i}"]["orig"].append(orig)
#             entropy_components[f"Entropy k{i}"]["byte_shuffle"].append(bs)
#             entropy_components[f"Entropy k{i}"]["tdt"].append(min_tdt)
#
#     out_path = "/mnt/c/Users/jamalids/Downloads/figs/"
#     for ylabel, values in entropy_components.items():
#         plt.figure(figsize=(10, 6))
#         plt.plot(values["orig"], label="Original")
#         plt.plot(values["byte_shuffle"], label="Byte-Shuffle")
#         plt.plot(values["tdt"], label="TDT")
#         plt.xticks(range(len(datasets)), datasets, rotation=45)
#         plt.gca().spines['top'].set_visible(False)
#         plt.gca().spines['right'].set_visible(False)
#         plt.ylabel(ylabel)
#         plt.tight_layout()
#         plt.legend()
#         plt.savefig(f"{out_path}{ylabel}.pdf")
#         plt.close()
#
# def compute_stats(data, label_x, label_y_list):
#     clean_data = data.dropna(subset=[label_x] + label_y_list)
#     stats = {}
#
#     for idx, label_y in enumerate(label_y_list):
#         p = spearmanr(clean_data[label_x], clean_data[label_y])
#         r2 = r2_score(clean_data[label_x], clean_data[label_y])
#         slope, intercept = compute_line_formula(clean_data[label_x], clean_data[label_y])
#         stats[f"k{idx}"] = {"r2": r2, "slope": slope, "intercept": intercept, "pvalue": p}
#
#     return stats
#
# # --- Load your data ---
# data1 = pd.read_csv("/mnt/c/Users/jamalids/Downloads/figs/results/results-corrolation/merged22 (64).csv")
# data2 = pd.read_csv("/mnt/c/Users/jamalids/Downloads/figs/results/results-corrolation/merged22(32).csv")
# data = pd.merge(data1, data2, how='outer')
#
# # --- Prepare new columns ---
# data['tdt_k0_entropy'] = np.minimum(data['custom_k0'], data['tdt_after_custom_k0'])
# data['tdt_k1_entropy'] = np.minimum(data['custom_k1'], data['tdt_after_custom_k1'])
# data['tdt_k2_entropy'] = np.minimum(data['custom_k2'], data['tdt_after_custom_k2'])
# data['tdt_k3_entropy'] = np.minimum(data['custom_k3'], data['tdt_after_custom_k3'])
# data['tdt_k4_entropy'] = np.minimum(data['custom_k4'], data['tdt_after_custom_k4'])
#
# data['orig_k0 ratio'] = data['orig_k0'] / data['tdt_k0_entropy']
# data['orig_k1 ratio'] = data['orig_k1'] / data['tdt_k1_entropy']
# data['orig_k2 ratio'] = data['orig_k2'] / data['tdt_k2_entropy']
# data['orig_k3 ratio'] = data['orig_k3'] / data['tdt_k3_entropy']
# data['orig_k4 ratio'] = data['orig_k4'] / data['tdt_k4_entropy']
# data['tdt over zstd ratio'] = data['standard zstd ratio'] / data['decomposed zstd row-order (B)']
#
# # --- Plot average entropy ---
# average_entropy_plot(data)
#
# # --- Compute statistics ---
# label_x = 'tdt over zstd ratio'
# label_y_list = ['orig_k0 ratio', 'orig_k1 ratio', 'orig_k2 ratio', 'orig_k3 ratio', 'orig_k4 ratio']
# stats = compute_stats(data, label_x, label_y_list)
# for k, v in stats.items():
#     print(f"K={k}: R²={v['r2']:.3f}, Slope={v['slope']:.3f}, Intercept={v['intercept']:.3f}, p-value={v['pvalue']}")
#
# label_x = 'decomposed zstd row-order (B)'
# label_y_list = ['tdt_k0_entropy', 'tdt_k1_entropy', 'tdt_k2_entropy', 'tdt_k3_entropy', 'tdt_k4_entropy']
# stats = compute_stats(data, label_x, label_y_list)
# for k, v in stats.items():
#     print(f"K={k}: R²={v['r2']:.3f}, Slope={v['slope']:.3f}, Intercept={v['intercept']:.3f}, p-value={v['pvalue']}")
#
# # --- Plot p-values (VLDB style) ---
# plt.figure(figsize=(6.2, 2.5))
# list_k = list(stats.keys())
# pvalues = [v['pvalue'][0] for v in stats.values()]
# plt.plot(list_k, pvalues, marker='o', linestyle='-')
# plt.xticks(list_k)
# plt.gca().spines['top'].set_visible(False)
# plt.gca().spines['right'].set_visible(False)
# plt.xlabel("Entropy Order", labelpad=6)
# plt.ylabel("Spearman Correlation Coefficient", labelpad=6)
# plt.grid(True)
# plt.tight_layout(pad=0.5)
#
# # Save figure
# output_base = "/mnt/c/Users/jamalids/Downloads/figs/spearman_correlation12"
# plt.savefig(f"{output_base}.pdf", bbox_inches="tight", dpi=300)
# plt.savefig(f"{output_base}.png", bbox_inches="tight", dpi=300)
# plt.close()
#
# print("✅ Entropy and correlation plots saved.")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import spearmanr

# 🌟 Set global font and plot style (VLDB style)
mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Latin Modern Roman", "Times", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})
# --- Helper functions ---
def compute_line_formula(x, y):
    model = LinearRegression()
    model.fit(np.array(x).reshape(-1, 1), y)
    return model.coef_[0], model.intercept_

def compute_stats(data, label_x, label_y_list):
    clean_data = data.dropna(subset=[label_x] + label_y_list)
    stats = {}
    for idx, label_y in enumerate(label_y_list):
        p = spearmanr(clean_data[label_x], clean_data[label_y])
        r2 = r2_score(clean_data[label_x], clean_data[label_y])
        slope, intercept = compute_line_formula(clean_data[label_x], clean_data[label_y])
        stats[f"k{idx}"] = {"r2": r2, "slope": slope, "intercept": intercept, "pvalue": p}
    return stats

def average_entropy_plot(data, dataset_ids):
    entropy_components = {f"Entropy k{i}": {"orig": [], "byte_shuffle": [], "tdt": []} for i in range(5)}

    for _, row in data.iterrows():
        for i in range(5):
            orig = row[f'orig_k{i}']
            bs = row[f'tdt_k{i}']
            custom = row[f'custom_k{i}']
            after_custom = row[f'tdt_after_custom_k{i}']
            min_tdt = np.min([custom, after_custom])
            entropy_components[f"Entropy k{i}"]["orig"].append(orig)
            entropy_components[f"Entropy k{i}"]["byte_shuffle"].append(bs)
            entropy_components[f"Entropy k{i}"]["tdt"].append(min_tdt)

    out_path = "/mnt/c/Users/jamalids/Downloads/figs/"
    for ylabel, values in entropy_components.items():
        plt.figure(figsize=(6.2, 2.5))
        plt.plot(values["orig"], label="Original")
       # plt.plot(values["byte_shuffle"], label="Byte-Shuffle")
        plt.plot(values["tdt"], label="TDT")
        plt.xticks(range(len(dataset_ids)), dataset_ids, rotation=45)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.legend()
        plt.savefig(f"{out_path}{ylabel}.pdf")
        plt.close()

# --- Load your data ---
data1 = pd.read_csv("/mnt/c/Users/jamalids/Downloads/figs/results/results-corrolation/merged22 (64).csv")
data2 = pd.read_csv("/mnt/c/Users/jamalids/Downloads/figs/results/results-corrolation/merged22(32).csv")
data = pd.merge(data1, data2, how='outer')
if 'dataset_name' in data.columns:
    data = data.rename(columns={'dataset_name': 'DatasetName'})


# --- Load DatasetIdMapping ---
mapping = pd.read_csv("/mnt/c/Users/jamalids/Downloads/figs/DatasetIdMapping.csv")
# Merge the mapping: now data will have a 'DatasetID' column
data = pd.merge(data, mapping[['DatasetName', 'DatasetID']], on='DatasetName', how='inner')

# --- Sort by DatasetID ---
data = data.sort_values('DatasetID').reset_index(drop=True)

# --- Prepare new columns ---
data['tdt_k0_entropy'] = np.minimum(data['custom_k0'], data['tdt_after_custom_k0'])
data['tdt_k1_entropy'] = np.minimum(data['custom_k1'], data['tdt_after_custom_k1'])
data['tdt_k2_entropy'] = np.minimum(data['custom_k2'], data['tdt_after_custom_k2'])
data['tdt_k3_entropy'] = np.minimum(data['custom_k3'], data['tdt_after_custom_k3'])
data['tdt_k4_entropy'] = np.minimum(data['custom_k4'], data['tdt_after_custom_k4'])

data['orig_k0 ratio'] = data['orig_k0'] / data['tdt_k0_entropy']
data['orig_k1 ratio'] = data['orig_k1'] / data['tdt_k1_entropy']
data['orig_k2 ratio'] = data['orig_k2'] / data['tdt_k2_entropy']
data['orig_k3 ratio'] = data['orig_k3'] / data['tdt_k3_entropy']
data['orig_k4 ratio'] = data['orig_k4'] / data['tdt_k4_entropy']
data['tdt over zstd ratio'] = data['standard zstd ratio'] / data['decomposed zstd row-order (B)']

# --- Plot average entropy (using DatasetID) ---
average_entropy_plot(data, data['DatasetID'])

# --- Compute statistics ---
label_x = 'tdt over zstd ratio'
label_y_list = ['orig_k0 ratio', 'orig_k1 ratio', 'orig_k2 ratio', 'orig_k3 ratio', 'orig_k4 ratio']
stats = compute_stats(data, label_x, label_y_list)
for k, v in stats.items():
    print(f"K={k}: R²={v['r2']:.3f}, Slope={v['slope']:.3f}, Intercept={v['intercept']:.3f}, p-value={v['pvalue']}")

label_x = 'decomposed zstd row-order (B)'
label_y_list = ['tdt_k0_entropy', 'tdt_k1_entropy', 'tdt_k2_entropy', 'tdt_k3_entropy', 'tdt_k4_entropy']
stats = compute_stats(data, label_x, label_y_list)
for k, v in stats.items():
    print(f"K={k}: R²={v['r2']:.3f}, Slope={v['slope']:.3f}, Intercept={v['intercept']:.3f}, p-value={v['pvalue']}")

# --- Plot p-values (VLDB style) ---
plt.figure(figsize=(6.2, 2.5))
list_k = list(stats.keys())
pvalues = [v['pvalue'][0] for v in stats.values()]
plt.plot(list_k, pvalues, marker='o', linestyle='-')
plt.xticks(list_k)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xlabel("Entropy Order", labelpad=6)
plt.ylabel("Spearman Correlation Coefficient", labelpad=6)
plt.grid(True)
plt.tight_layout(pad=0.5)

# Save figure
output_base = "/mnt/c/Users/jamalids/Downloads/figs/spearman_correlation12"
plt.savefig(f"{output_base}.pdf", bbox_inches="tight", dpi=300)
plt.savefig(f"{output_base}.png", bbox_inches="tight", dpi=300)
plt.close()

print("✅ Plots saved with Dataset IDs instead of Dataset names.")
