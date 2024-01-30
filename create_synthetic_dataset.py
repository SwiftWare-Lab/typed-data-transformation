
#generating synthetic data based on the distribution and features of the real dataset.
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np

# Load the real data from the CSV file
data = pd.read_csv('./data/OC_Blood_Routine.csv')
#Analyzing Data Distributions
data_description = data.describe()
data.hist(figsize=(12, 10))
plt.show()
synthetic_data = pd.DataFrame()
for column in data.columns:
    if data[column].dtype in ['float64', 'int64']:
        # Fit a normal distribution and sample data
        mean, std = norm.fit(data[column].dropna())
        synthetic_data[column] = np.random.normal(mean, std, size=50000)
#Categorical Data:
for column in data.columns:
    if data[column].dtype == 'object':
        # Sample based on the frequency of categories
        frequencies = data[column].value_counts(normalize=True)
        synthetic_data[column] = np.random.choice(frequencies.index, size=50000, p=frequencies.values)

#Finalizing the Synthetic Dataset
for column in data.columns:
    synthetic_data[column] = synthetic_data[column].astype(data[column].dtype)

# Save or use the synthetic data
# TODO store it in the data folder
synthetic_data.to_csv('./data/Synthetic_OC_Blood_Routine_50k1.csv', index=False)





