#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss

# Load the data from the CSV file
data = pd.read_csv('/home/jamalids/Downloads/OC_Blood_Routine (1).csv')


# In[ ]:


#Analyzing Data Distributions


# In[14]:


# Get basic statistics
data_description = data.describe()

data.hist(figsize=(12, 10))
plt.show()


from scipy.stats import norm
import numpy as np

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





# Adjust data types if needed
for column in data.columns:
    synthetic_data[column] = synthetic_data[column].astype(data[column].dtype)

# Save or use the synthetic data
synthetic_data.to_csv('/home/jamalids/Downloads/Synthetic_OC_Blood_Routine_50k.csv', index=False)





