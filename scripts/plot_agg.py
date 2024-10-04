import pandas as pd
import matplotlib.pyplot as plt

file_path = "/home/jamalids/Documents/compression-part3/big-data-compression/modeling/final/H_E_agg.csv"
low_entropy_data1 = pd.read_csv(file_path)
# Sort data based on 'entropy_float'
low_entropy_data= low_entropy_data1.sort_values(by='entropy_float')

# Create a plot comparing comp_ratio_zstd_default, comp_ratio_l22, and comp_ratio_gzip
# against max_Decom+zstd_22_com_ratio, max_Decom+zstd_com_ratio, max_Decom+gzip_com_ratio

plt.figure(figsize=(10, 6))

# Plot compression ratios
plt.plot(low_entropy_data['dataset_name'], low_entropy_data['comp_ratio_zstd_default'], label='comp_ratio_zstd_default', marker='o')
plt.plot(low_entropy_data['dataset_name'], low_entropy_data['comp_ratio_l22'], label='comp_ratio_l22', marker='o')
plt.plot(low_entropy_data['dataset_name'], low_entropy_data['comp_ratio_gzip'], label='comp_ratio_gzip', marker='o')

# Plot max decompression ratios for comparison
plt.plot(low_entropy_data['dataset_name'], low_entropy_data['max_Decom+zstd_22_com_ratio'], label='Decom+zstd_22_com_ratio', linestyle='--')
plt.plot(low_entropy_data['dataset_name'], low_entropy_data['max_Decom+zstd_com_ratio'], label='Decom+zstd_com_ratio', linestyle='--')
plt.plot(low_entropy_data['dataset_name'], low_entropy_data['max_Decom+gzip_com_ratio'], label='Decom+gzip_com_ratio', linestyle='--')

plt.xticks(rotation=90)  # Rotate dataset names for better readability
plt.xlabel('Dataset Name')
plt.ylabel('Compression Ratio')
plt.title('Comparison of Compression Ratios for High Entropy Float Data with and without Decomposition')
plt.legend()
plt.tight_layout()
plt.savefig('High Entropy.png')
# Display the plot
plt.show()
#########################
# Let's fix the missing import for numpy and re-run the code
import numpy as np

plt.figure(figsize=(10, 6))
data = low_entropy_data

# Plot compression ratios only with points (no lines connecting them)
plt.scatter(data['entropy_float'], data['comp_ratio_zstd_default'], label='comp_ratio_zstd_default', color='orange')
plt.scatter(data['entropy_float'], data['comp_ratio_l22'], label='comp_ratio_l22', color='red')
plt.scatter(data['entropy_float'], data['comp_ratio_gzip'], label='comp_ratio_gzip', color='green')

# Customize the plot for better presentation
plt.xlabel('Entropy (Float)', fontsize=12)
plt.ylabel('Compression Ratio', fontsize=12)
plt.title('Impact of Entropy on Compression Ratios', fontsize=14)
plt.legend(title="Compression Ratios", fontsize=10)

# Set x-axis to use entropy as continuous float numbers
plt.xticks(np.arange(0, data['entropy_float'].max() + 1, step=1))

plt.grid(True, linestyle='--', alpha=0.7)

# Display the plot with points only and adjusted x-axis as float numbers
plt.tight_layout()
plt.show()
# Let's remove the trend lines and display only the points
# To show the correlation line between entropy and compression ratios, let's fit a linear regression line for each compression ratio.
from sklearn.linear_model import LinearRegression
import numpy as np

plt.figure(figsize=(10, 6))

# Prepare data for linear regression
X = data['entropy_float'].values.reshape(-1, 1)

# Fit linear regression models for each compression ratio
for ratio_col, color, label in zip(['comp_ratio_zstd_default', 'comp_ratio_l22', 'comp_ratio_gzip'],
                                   ['orange', 'red', 'green'],
                                   ['comp_ratio_zstd_default', 'comp_ratio_zstd_22', 'comp_ratio_gzip']):
    y = data[ratio_col].values
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)

    # Scatter plot
    plt.scatter(data['entropy_float'], y, label=label, color=color)

    # Plot the regression line
    plt.plot(data['entropy_float'], y_pred, color=color, linestyle='--')

# Customize the plot for better presentation
plt.xlabel('Entropy ', fontsize=12)
plt.ylabel('Compression Ratio', fontsize=12)
plt.title('Correlation between Entropy and Compression Ratios with Trend Lines', fontsize=14)
plt.legend(title="Compression Ratios", fontsize=10)

# Set x-axis to use entropy as continuous float numbers
plt.xticks(np.arange(0, data['entropy_float'].max() + 1, step=1))

plt.grid(True, linestyle='--', alpha=0.7)

# Display the plot with scatter points and trend lines
plt.tight_layout()
plt.savefig('h_comp_ratios.png', dpi=300)
plt.show()
