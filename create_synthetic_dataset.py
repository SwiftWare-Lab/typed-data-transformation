
#generating synthetic data based on the distribution and features of the real dataset.
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np

# Load the real data from the CSV file
#data = pd.read_csv('/home/jamalids/Downloads/OC_Blood_Routine (1).csv')
csv_file_path = input("Please enter the path to your CSV file: ")
data = pd.read_csv(csv_file_path)
# Ask the user for a number
size_data = int(input("Please enter a size of data: "))

#Analyzing Data Distributions
data_description = data.describe()
data.hist(figsize=(12, 10))
plt.show()
synthetic_data = pd.DataFrame()
for column in data.columns:
    if data[column].dtype in ['float64', 'int64']:
        # Fit a normal distribution and sample data
        mean, std = norm.fit(data[column].dropna())
        synthetic_data[column] = np.random.normal(mean, std, size=size_data )
#Categorical Data:
for column in data.columns:
    if data[column].dtype == 'object':
        # Sample based on the frequency of categories
        frequencies = data[column].value_counts(normalize=True)
        synthetic_data[column] = np.random.choice(frequencies.index, size=size_data , p=frequencies.values)

#Finalizing the Synthetic Dataset
for column in data.columns:
    synthetic_data[column] = synthetic_data[column].astype(data[column].dtype)

# Save or use the synthetic data
csv_save_path = input("Please enter the path where you want to save the results CSV file (including filename and extension): ")
synthetic_data.to_csv(csv_save_path, index=False)
#synthetic_data.to_csv('/home/jamalids/Downloads/Synthetic_OC_Blood_Routine_50k1.csv', index=False)





