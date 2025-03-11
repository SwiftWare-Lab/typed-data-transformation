
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import spearmanr

# Compute the line formula for each case
def compute_line_formula(x, y):
    model = LinearRegression()
    model.fit(np.array(x).reshape(-1, 1), y)
    slope = model.coef_[0]
    intercept = model.intercept_
    return slope, intercept


def get_slop_for_all_pair(data_frame):
    numerical_columns = data_frame.select_dtypes(include=["float64", "int"]).columns
    results = {}
    for i, col1 in enumerate(numerical_columns):
        for col2 in numerical_columns[i + 1:]:
            x = data_frame[col1]
            y = data_frame[col2]
            # if x or y has NaN values, continue to the next pair
            if x.isnull().values.any() or y.isnull().values.any():
                continue
            slope, intercept = compute_line_formula(x, y)
            r2 = r2_score(x, y)
            s_r2 = spearmanr(x, y)
            results[(col1, col2)] = {"slope": slope, "intercept": intercept, "r2": r2, "spearman_r2": s_r2}
    return results


def average_entropy_plot(data):
    # for each row in the data, compute the average entropy
    average_bytshuffle_entropy, average_tdt_entropy, original_entropy, tdt_min_entropy, tdt_custom_entropy = [], [], [], [], []
    average_bytshuffle_entropyk1, average_tdt_entropyk1, original_entropyk1, tdt_min_entropyk1, tdt_custom_entropyk1 = [], [], [], [], []
    average_bytshuffle_entropyk2, average_tdt_entropyk2, original_entropyk2, tdt_min_entropyk2, tdt_custom_entropyk2 = [], [], [], [], []
    average_bytshuffle_entropyk3, average_tdt_entropyk3, original_entropyk3, tdt_min_entropyk3, tdt_custom_entropyk3 = [], [], [], [], []
    average_bytshuffle_entropyk4, average_tdt_entropyk4, original_entropyk4, tdt_min_entropyk4, tdt_custom_entropyk4 = [], [], [], [], []
    dataset_names = []
    # for each row in the data, compute the average entropy
    for index, row in data.iterrows():
        oe0, oe1, oe2, oe3, oe4 = row['orig_k0'], row['orig_k1'], row['orig_k2'], row['orig_k3'], row['orig_k4']
        te0, te1, te2, te3, te4 = row['tdt_k0'], row['tdt_k1'], row['tdt_k2'], row['tdt_k3'], row['tdt_k4']
        # bs_b0_e0, bs_b1_e0, bs_b2_e0, bs_b3_e0 = row['tdt_group_entropy_G0'], row['tdt_group_entropy_G1'], row['tdt_group_entropy_G2'], row['tdt_group_entropy_G3']
        # avg_bitshuffle = (bs_b0_e0 + bs_b1_e0 + bs_b2_e0 + bs_b3_e0) / 4
        tdt_k0, tdt_k1, tdt_k2, tdt_k3, tdt_k4 = row['custom_k0'], row['custom_k1'], row['custom_k2'], row['custom_k3'], row['custom_k4']
        tdtc_k0, tdtc_k1, tdtc_k2, tdtc_k3, tdtc_k4 = row['tdt_after_custom_k0'], row['tdt_after_custom_k1'], row['tdt_after_custom_k2'], row['tdt_after_custom_k3'], row['tdt_after_custom_k4']
        we_k0 = row['WE']
        ds_name = row['dataset_name']
        dataset_names.append(ds_name)

        original_entropy.append(oe0)
        average_bytshuffle_entropy.append(te0)
        tdt_custom_entropy.append(tdt_k0)
        average_tdt_entropy.append(tdtc_k0)
        tdt_min_entropy.append(np.min([tdt_k0, tdtc_k0]))
        # k1
        original_entropyk1.append(oe1)
        average_bytshuffle_entropyk1.append(te1)
        tdt_custom_entropyk1.append(tdt_k1)
        average_tdt_entropyk1.append(tdtc_k1)
        tdt_min_entropyk1.append(np.min([tdt_k1, tdtc_k1]))
        # k2
        original_entropyk2.append(oe2)
        average_bytshuffle_entropyk2.append(te2)
        tdt_custom_entropyk2.append(tdt_k2)
        average_tdt_entropyk2.append(tdtc_k2)
        tdt_min_entropyk2.append(np.min([tdt_k2, tdtc_k2]))
        # k3
        original_entropyk3.append(oe3)
        average_bytshuffle_entropyk3.append(te3)
        tdt_custom_entropyk3.append(tdt_k3)
        average_tdt_entropyk3.append(tdtc_k3)
        tdt_min_entropyk3.append(np.min([tdt_k3, tdtc_k3]))
        # k4
        original_entropyk4.append(oe4)
        average_bytshuffle_entropyk4.append(te4)
        tdt_custom_entropyk4.append(tdt_k4)
        average_tdt_entropyk4.append(tdtc_k4)
        tdt_min_entropyk4.append(np.min([tdt_k4, tdtc_k4]))




    out_path="./"
    plot_entropy = lambda entropy, ylabel: (
        plt.figure(figsize=(10, 6)),
        plt.plot(entropy[0], label='Original'),
        plt.plot(entropy[1], label='Byte-Shuffle'),
        plt.plot(entropy[2], label='TDT'),
        plt.xticks(range(len(dataset_names)), dataset_names, rotation=45),
        plt.gca().spines['top'].set_visible(False),
        plt.gca().spines['right'].set_visible(False),
        plt.ylabel(ylabel),
        plt.tight_layout(),
        plt.legend(),
        #plt.show(),
        plt.savefig(f"{out_path}{ylabel}.pdf"),
        plt.close()
    )

    plot_entropy([original_entropy, average_bytshuffle_entropy, tdt_min_entropy], "Entropy k0")
    plot_entropy([original_entropyk1, average_bytshuffle_entropyk1, tdt_min_entropyk1], "Entropy k1")
    plot_entropy([original_entropyk2, average_bytshuffle_entropyk2, tdt_min_entropyk2], "Entropy k2")
    plot_entropy([original_entropyk3, average_bytshuffle_entropyk3, tdt_min_entropyk3], "Entropy k3")
    plot_entropy([original_entropyk4, average_bytshuffle_entropyk4, tdt_min_entropyk4], "Entropy k4")


def compute_stats(data, label_x, label_y):
    # compute coefficient of determination (R^2) for each k
    p_k0 = spearmanr(data[label_x], data[label_y[0]])
    p_k1 = spearmanr(data[label_x], data[label_y[1]])
    p_k2 = spearmanr(data[label_x], data[label_y[2]])
    p_k3 = spearmanr(data[label_x], data[label_y[3]])
    p_k4 = spearmanr(data[label_x], data[label_y[4]])
    r2_k0 = r2_score(data[label_x], data[label_y[0]])
    r2_k1 = r2_score(data[label_x], data[label_y[1]])
    r2_k2 = r2_score(data[label_x], data[label_y[2]])
    r2_k3 = r2_score(data[label_x], data[label_y[3]])
    r2_k4 = r2_score(data[label_x], data[label_y[4]])

    x = np.array(data[label_x])

    slope_k0, intercept_k0 = compute_line_formula(x, data[label_y[0]])
    slope_k1, intercept_k1 = compute_line_formula(x, data[label_y[1]])
    slope_k2, intercept_k2 = compute_line_formula(x, data[label_y[2]])
    slope_k3, intercept_k3 = compute_line_formula(x, data[label_y[3]])
    slope_k4, intercept_k4 = compute_line_formula(x, data[label_y[4]])

    stats = {
        "k0": {"r2": r2_k0, "slope": slope_k0, "intercept": intercept_k0, "pvalue": p_k0},
        "k1": {"r2": r2_k1, "slope": slope_k1, "intercept": intercept_k1, "pvalue": p_k1},
        "k2": {"r2": r2_k2, "slope": slope_k2, "intercept": intercept_k2, "pvalue": p_k2},
        "k3": {"r2": r2_k3, "slope": slope_k3, "intercept": intercept_k3, "pvalue": p_k3},
        "k4": {"r2": r2_k4, "slope": slope_k4, "intercept": intercept_k4, "pvalue": p_k4},
    }

    return stats





# Load the dataset
data = pd.read_csv('/home/jamalids/Documents/results-corrolation/merged22 (64).csv')


# Specify the dataset name for labeling
dataset_name = "Merged Dataset"


# Inspect the data structure and types
#print(data.info())

# Display summary statistics
#print(data.describe())

average_entropy_plot(data)


# add 'orig_k0' / 'tdt_k0', 'orig_k1'/'tdt_k1', 'orig_k2'/'tdt_k2' to the data dataframe

data['tdt_k0_entropy'] = np.minimum(data['custom_k0'], data['tdt_after_custom_k0'])
data['tdt_k1_entropy'] = np.minimum(data['custom_k1'], data['tdt_after_custom_k1'])
data['tdt_k2_entropy'] = np.minimum(data['custom_k2'], data['tdt_after_custom_k2'])
data['tdt_k3_entropy'] = np.minimum(data['custom_k3'], data['tdt_after_custom_k3'])
data['tdt_k4_entropy'] = np.minimum(data['custom_k4'], data['tdt_after_custom_k4'])
data['orig_k0 ratio'] = data['orig_k0'] / data['tdt_k0_entropy'] # higher is better
data['orig_k1 ratio'] = data['orig_k1'] / data['tdt_k1_entropy']
data['orig_k2 ratio'] = data['orig_k2'] / data['tdt_k2_entropy']
data['orig_k3 ratio'] = data['orig_k3'] / data['tdt_k3_entropy']
data['orig_k4 ratio'] = data['orig_k4'] / data['tdt_k4_entropy']
data['tdt over zstd ratio'] = data['standard zstd ratio'] / data['decomposed zstd row-order (B)'] # tdt zstd size / zstd size -> lower is better


# slopes = get_slop_for_all_pair(data)
# for pair, values in slopes.items():
#     # if values['r2'] > 0.8: #and values['r2'] < 2:
#     #     print(f"Pair: {pair}, Slope: {values['slope']}, R^2: {values['r2']}")
#     if 'comp ratio ratio' in pair[0] or 'comp ratio ratio' in pair[1]:
#         print(f"Pair: {pair}, Slope: {values['slope']}, R^2: {values['r2']}, Spearman R^2: {values['spearman_r2']}")

# Example visualizations:
# Note: Adjust column names to fit your dataset's actual structure.

# make correlation plot between 'reordered zstd row-order (B)' and 'tdt_after_custom_k1'

label_x = 'tdt over zstd ratio'
label_y = ['orig_k0 ratio', 'orig_k1 ratio', 'orig_k2 ratio', 'orig_k3 ratio', 'orig_k4 ratio']
stats = compute_stats(data, label_x, label_y)
for k, v in stats.items():
    print(f"K={k}: R^2={v['r2']}, Slope={v['slope']}, Intercept={v['intercept']}, p-value={v['pvalue']}")
    print()

label_x = 'decomposed zstd row-order (B)'
label_y = ['tdt_k0_entropy', 'tdt_k1_entropy', 'tdt_k2_entropy', 'tdt_k3_entropy', 'tdt_k4_entropy']
stats = compute_stats(data, label_x, label_y)
for k, v in stats.items():
    print(f"K={k}: R^2={v['r2']}, Slope={v['slope']}, Intercept={v['intercept']}, p-value={v['pvalue']}")
    print()

# plot k vs v['pvalue'][0] for each k
plt.figure(figsize=(10, 6))
list_k = list(stats.keys())
plt.plot(list_k, [v['pvalue'][0] for v in stats.values()], label='p-value', marker='o')
plt.xticks(list_k)
# remove top and right spines
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xlabel("Entropy Order")
plt.ylabel("Spearman Correlation Coefficient")
plt.grid()
plt.savefig("/home/jamalids/Documents/spearman_correlation.pdf")


# plt.xlabel("Entropy Order")
# plt.title(f"Scatter Plot: reordered zstd row-order (B) vs tdt_after_custom")
# plt.show()


# plot tdt_k1_entropy vs 'decomposed zstd row-order ratio'



# 1. Line Plot (for trends if a time-series or sequential column exists):
if "date" in data.columns:  # Replace 'date' with your date/time column.
    data["date"] = pd.to_datetime(data["date"])
    data.sort_values("date", inplace=True)
    plt.figure(figsize=(10, 6))
    plt.plot(data["date"], data.iloc[:, 1], label=data.columns[1])  # Replace with meaningful columns.
    plt.title("Line Plot for Time Series")
    plt.xlabel("Date")
    plt.ylabel(data.columns[1])
    plt.legend()
    plt.grid()
    plt.savefig("/home/jamalids/Documents/line_plot")

# 2. Histogram (for distribution of a numerical column):
numerical_columns = data.select_dtypes(include=["float64", "int"]).columns
if not numerical_columns.empty:
    plt.figure(figsize=(10, 6))
    for col in numerical_columns[:3]:  # Limit to first 3 numerical columns if many.
        sns.histplot(data[col], kde=True, label=col, bins=20)
    plt.title("Histogram of Numerical Columns")
    plt.legend()
    plt.savefig("/home/jamalids/Documents/histogram.pdf")

# 3. Scatter Plot with regression line (for relationships between two numerical columns):
if len(numerical_columns) >= 2:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=numerical_columns[0], y=numerical_columns[1], data=data, alpha=0.6)
    sns.regplot(x=numerical_columns[0], y=numerical_columns[1], data=data, scatter=False, line_kws={"color": "red"})
    plt.title(f"Scatter Plot: {numerical_columns[0]} vs {numerical_columns[1]}")
    plt.savefig("/home/jamalids/Documents/Scatter Plot")

# 4. Box Plot (to analyze spread and outliers for a category):
categorical_columns = data.select_dtypes(include=["object"]).columns
if not categorical_columns.empty:
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=categorical_columns[0], y=numerical_columns[0], data=data)  # Replace with meaningful columns.
    plt.title(f"Box Plot of {numerical_columns[0]} by {categorical_columns[0]}")
    plt.xticks(rotation=45)
    plt.savefig(f"box-plot-{numerical_columns[0]}")

# 5. Heatmap (for correlations among numerical columns):
if len(numerical_columns) > 1:
    corr_matrix = data[numerical_columns].corr(method='pearson')
    plt.figure(figsize=(80, 64))

    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    # make x-axis and y-axis labels bigger
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig("/home/jamalids/Documents/correlation_heatmap.png")




