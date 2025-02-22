
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


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
            results[(col1, col2)] = {"slope": slope, "intercept": intercept, "r2": r2}
    return results

# Load the dataset
data = pd.read_csv("scripts/merged.csv")


# Specify the dataset name for labeling
dataset_name = "Merged Dataset"


# Inspect the data structure and types
#print(data.info())

# Display summary statistics
#print(data.describe())



# add 'orig_k0' / 'tdt_k0', 'orig_k1'/'tdt_k1', 'orig_k2'/'tdt_k2' to the data dataframe
data['orig_k0 ratio'] = data['orig_k0'] / data['tdt_k0']
data['orig_k1 ratio'] = data['orig_k1'] / data['tdt_k1']
data['orig_k2 ratio'] = data['orig_k2'] / data['tdt_k2']
data['orig_k3 ratio'] = data['orig_k3'] / data['tdt_k3']
data['orig_k4 ratio'] = data['orig_k4'] / data['tdt_k4']


slopes = get_slop_for_all_pair(data)
for pair, values in slopes.items():
    if values['r2'] > 0.5 and values['r2'] < 2:
        print(f"Pair: {pair}, Slope: {values['slope']}, R^2: {values['r2']}")
# Example visualizations:
# Note: Adjust column names to fit your dataset's actual structure.

# make correlation plot between 'reordered zstd row-order (B)' and 'tdt_after_custom_k1'
plt.figure(figsize=(10, 6))


sns.scatterplot(x='reordered zstd row-order (B)', y='tdt_after_custom_k0', data=data, alpha=0.6)
sns.regplot(x='reordered zstd row-order (B)', y='tdt_after_custom_k0', data=data, scatter=False, line_kws={"color": "green"})
sns.scatterplot(x='reordered zstd row-order (B)', y='tdt_after_custom_k1', data=data, alpha=0.6)
sns.regplot(x='reordered zstd row-order (B)', y='tdt_after_custom_k1', data=data, scatter=False, line_kws={"color": "red"})
sns.scatterplot(x='reordered zstd row-order (B)', y='tdt_after_custom_k2', data=data, alpha=0.6)
sns.regplot(x='reordered zstd row-order (B)', y='tdt_after_custom_k2', data=data, scatter=False, line_kws={"color": "blue"})
sns.scatterplot(x='reordered zstd row-order (B)', y='tdt_after_custom_k3', data=data, alpha=0.6)
sns.regplot(x='reordered zstd row-order (B)', y='tdt_after_custom_k3', data=data, scatter=False, line_kws={"color": "yellow"})
sns.scatterplot(x='reordered zstd row-order (B)', y='tdt_after_custom_k4', data=data, alpha=0.6)
sns.regplot(x='reordered zstd row-order (B)', y='tdt_after_custom_k4', data=data, scatter=False, line_kws={"color": "purple"})
# add legend
plt.legend(['k0', 'k1', 'k2', 'k3', 'k4'])
# compute coefficient of determination (R^2) for each k
r2_k0 = r2_score(data['reordered zstd row-order (B)'], data['tdt_after_custom_k0'])
r2_k1 = r2_score(data['reordered zstd row-order (B)'], data['tdt_after_custom_k1'])
r2_k2 = r2_score(data['reordered zstd row-order (B)'], data['tdt_after_custom_k2'])
r2_k3 = r2_score(data['reordered zstd row-order (B)'], data['tdt_after_custom_k3'])
r2_k4 = r2_score(data['reordered zstd row-order (B)'], data['tdt_after_custom_k4'])
x = np.array(data['reordered zstd row-order (B)'])

slope_k0, intercept_k0 = compute_line_formula(x, data['tdt_after_custom_k0'])
slope_k1, intercept_k1 = compute_line_formula(x, data['tdt_after_custom_k1'])
slope_k2, intercept_k2 = compute_line_formula(x, data['tdt_after_custom_k2'])
slope_k3, intercept_k3 = compute_line_formula(x, data['tdt_after_custom_k3'])
slope_k4, intercept_k4 = compute_line_formula(x, data['tdt_after_custom_k4'])

print(f"R^2 for k0: {r2_k0}, Line: y = {slope_k0}x + {intercept_k0}")
print(f"R^2 for k1: {r2_k1}, Line: y = {slope_k1}x + {intercept_k1}")
print(f"R^2 for k2: {r2_k2}, Line: y = {slope_k2}x + {intercept_k2}")
print(f"R^2 for k3: {r2_k3}, Line: y = {slope_k3}x + {intercept_k3}")
print(f"R^2 for k4: {r2_k4}, Line: y = {slope_k4}x + {intercept_k4}")



plt.xlabel("reordered zstd row-order (B)")
plt.title(f"Scatter Plot: reordered zstd row-order (B) vs tdt_after_custom")
plt.show()



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
    plt.show()

# 2. Histogram (for distribution of a numerical column):
numerical_columns = data.select_dtypes(include=["float64", "int"]).columns
if not numerical_columns.empty:
    plt.figure(figsize=(10, 6))
    for col in numerical_columns[:3]:  # Limit to first 3 numerical columns if many.
        sns.histplot(data[col], kde=True, label=col, bins=20)
    plt.title("Histogram of Numerical Columns")
    plt.legend()
    plt.show()

# 3. Scatter Plot with regression line (for relationships between two numerical columns):
if len(numerical_columns) >= 2:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=numerical_columns[0], y=numerical_columns[1], data=data, alpha=0.6)
    sns.regplot(x=numerical_columns[0], y=numerical_columns[1], data=data, scatter=False, line_kws={"color": "red"})
    plt.title(f"Scatter Plot: {numerical_columns[0]} vs {numerical_columns[1]}")
    plt.show()

# 4. Box Plot (to analyze spread and outliers for a category):
categorical_columns = data.select_dtypes(include=["object"]).columns
if not categorical_columns.empty:
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=categorical_columns[0], y=numerical_columns[0], data=data)  # Replace with meaningful columns.
    plt.title(f"Box Plot of {numerical_columns[0]} by {categorical_columns[0]}")
    plt.xticks(rotation=45)
    plt.show()

# 5. Heatmap (for correlations among numerical columns):
if len(numerical_columns) > 1:
    corr_matrix = data[numerical_columns].corr()
    plt.figure(figsize=(80, 64))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    # make x-axis and y-axis labels bigger
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()


