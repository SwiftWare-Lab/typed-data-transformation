import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
import os
import sys

if len(sys.argv) != 3:
    print("Usage: python Synthetic_Dataset.py <dataset_name> <size_data>")
    sys.exit(1)

# Load the dataset from the command-line argument
dataset_path = sys.argv[1]
data = pd.read_csv(dataset_path)  

# Convert size_data to an integer
size_data = int(sys.argv[2])  

# Analyzing Data Distributions
data_description = data.describe()
data.hist(figsize=(12, 10))
plt.show()

synthetic_data = pd.DataFrame()
for column in data.columns:
    if data[column].dtype in ['float64', 'int64']:
        # Fit a normal distribution and sample data
        mean, std = norm.fit(data[column].dropna())
        synthetic_data[column] = np.random.normal(mean, std, size=size_data)  

# Categorical Data:
for column in data.columns:
    if data[column].dtype == 'object':
        frequencies = data[column].value_counts(normalize=True)
        synthetic_data[column] = np.random.choice(frequencies.index, size=size_data, p=frequencies.values)  
# Finalizing the Synthetic Dataset
for column in data.columns:
    synthetic_data[column] = synthetic_data[column].astype(data[column].dtype)

if not os.path.exists('./data'):
    os.makedirs('./data')
    
filename = f'Synthetic-Dataset_{size_data}'   
csv_save_path = os.path.join('./data', filename)
# Save the DataFrame to the specified path
synthetic_data.to_csv(csv_save_path)
csv_save_path = os.path.join('./data', filename)

