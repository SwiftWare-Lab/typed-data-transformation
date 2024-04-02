#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from tslearn.datasets import UCR_UEA_datasets

from tslearn.datasets import UCR_UEA_datasets
UCR_UEA_datasets_list = UCR_UEA_datasets().list_multivariate_datasets()
ts_list, y_true, _, _ = UCR_UEA_datasets().load_dataset('LSST') 

tensor = ts_list
print(tensor.shape)
print(tensor.shape[1])

# Number of features in the tensor
num_features = tensor.shape[2]
print(num_features)

# Creating a time series index for plotting
timesteps = pd.date_range(start='00:00:00', periods=tensor.shape[1], freq='min')

# Setting up the plot
plt.figure(figsize=(14, 8))

# Plotting the trend for each feature
for feature in range(num_features):
    # Averaging over the samples for each timestep for the current feature
    feature_average = tensor[:, :, feature].mean(axis=0)
    
    # Plotting the averaged values over time
    plt.plot(timesteps, feature_average, label=f'Feature {feature + 1} Trend', marker='o')

# Customizing the plot
plt.title('Trend of Each Feature Over Time')
plt.xlabel('Time')
plt.ylabel('Average Value')
plt.grid(True)
plt.legend()
plt.show()


# In[ ]:





# In[7]:


import numpy as np
import matplotlib.pyplot as plt
from tslearn.datasets import UCR_UEA_datasets

# Loading the dataset
UCR_UEA_datasets_list = UCR_UEA_datasets().list_multivariate_datasets()
ts_list, y_true, _, _ = UCR_UEA_datasets().load_dataset('LSST')

# Assigning the loaded data to the tensor variable
tensor = ts_list
print(tensor.shape)

# Number of features in the tensor
num_features = tensor.shape[2]
print(num_features)

# The range of timesteps is used directly for plotting
timesteps = range(tensor.shape[1])

# Setting up the plot
plt.figure(figsize=(14, 8))

# Plotting the trend for each feature
for feature in range(num_features):
    # Averaging over the samples for each timestep for the current feature
    feature_average = tensor[:, :, feature].mean(axis=0)
    
    # Plotting the averaged values over time, using timesteps as the x-axis
    plt.plot(timesteps, feature_average, label=f'Feature {feature + 1} Trend', marker='o')

# Customizing the plot
plt.title('Trend of Each Feature Over Time')
plt.xlabel('Timestep')
plt.ylabel('Average Value')
plt.grid(True)
plt.legend()
plt.show()


# In[47]:


import numpy as np
import matplotlib.pyplot as plt
from tslearn.datasets import UCR_UEA_datasets

# Loading the dataset
UCR_UEA_datasets_list = UCR_UEA_datasets().list_multivariate_datasets()
ts_list, y_true, _, _ = UCR_UEA_datasets().load_dataset('UWaveGestureLibrary')

# Assigning the loaded data to the tensor variable
tensor = ts_list
print("Tensor shape:", tensor.shape)

# Number of features in the tensor
num_features = tensor.shape[2]
print("Number of features:", num_features)

# The range of timesteps is used directly for plotting
timesteps = np.arange(1, tensor.shape[1] + 1)  # Starting from 1 to the number of timesteps

# Setting up the plot
plt.figure(figsize=(14, 8))

# Plotting the trend for each feature
for feature in range(num_features):
    # Averaging over the samples for each timestep for the current feature
    feature_average = tensor[:, :, feature].mean(axis=0)
    
    # Plotting the averaged values over time, using timesteps as the x-axis
    plt.plot(timesteps, feature_average, label=f'Feature {feature + 1} Trend', marker='o')

# Customizing the plot
plt.title('Trend of Each Feature Over Time')
plt.xlabel('Timestep')
plt.ylabel('Average Value')
plt.xticks(timesteps)  # Ensure every timestep is marked on the x-axis
plt.grid(True)
plt.legend()
plt.show()


# In[20]:


import numpy as np
from tslearn.datasets import UCR_UEA_datasets
from statsmodels.tsa.stattools import acf
import matplotlib.pyplot as plt

# Loading the dataset
ts_list, y_true, _, _ = UCR_UEA_datasets().load_dataset('ArticularyWordRecognition')

# Assigning the loaded data to the tensor variable
tensor = ts_list

# Number of features in the tensor
num_features = tensor.shape[2]

# Setting up the plot
plt.figure(figsize=(14, num_features * 4))

for feature in range(num_features):
    # Compute the autocorrelation for each feature
    # We average the autocorrelation across all samples for simplicity
    autocorrelations = np.mean([acf(tensor[sample, :, feature], nlags=20, fft=True) for sample in range(tensor.shape[0])], axis=0)
    
    # Plotting the autocorrelation for the feature
    plt.subplot(num_features, 1, feature + 1)
    plt.stem(range(len(autocorrelations)), autocorrelations)
    plt.title(f'Autocorrelation for Feature {feature + 1}')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.tight_layout()

plt.show()


# In[21]:


import numpy as np
import matplotlib.pyplot as plt

# Hypothetical tensor with shape [samples, timesteps, features]
# For example, let's assume we have 50 samples, 100 timesteps, and 3 features
np.random.seed(0)  # For reproducibility
tensor = np.random.rand(50, 100, 3)

num_samples, num_timesteps, num_features = tensor.shape

# Function to calculate autocorrelation for a single feature across all samples
def autocorrelation(feature_values):
    """
    Calculate the autocorrelation for a single feature across all timesteps.
    feature_values: NumPy array of shape [samples, timesteps]
    Returns: NumPy array of autocorrelation values.
    """
    feature_mean = feature_values.mean()
    n = feature_values.size
    autocorr = np.correlate(feature_values - feature_mean, feature_values - feature_mean, mode='full')[-n:]
    return autocorr[:n // 2] / autocorr[n // 2]  # Normalize by zero lag

# Initialize a dictionary to store max autocorrelation and corresponding lag for each feature
feature_autocorr_info = {}

for feature in range(num_features):
    feature_values = tensor[:, :, feature].flatten()  # Flatten to treat all samples as a continuous series
    autocorr_values = autocorrelation(feature_values)
    max_lag = np.argmax(autocorr_values[1:]) + 1  # Skip the zero lag
    max_autocorr = autocorr_values[max_lag]
    feature_autocorr_info[feature] = (max_lag, max_autocorr)

# Identify the feature with the highest autocorrelation
best_feature, (best_lag, best_autocorr) = max(feature_autocorr_info.items(), key=lambda item: item[1][1])

print(f"Feature with the highest autocorrelation: Feature {best_feature}")
print(f"Highest autocorrelation: {best_autocorr} at lag {best_lag}")

# Optional: Visualize the autocorrelation for the best feature
plt.figure(figsize=(10, 6))
plt.plot(autocorrelation(tensor[:, :, best_feature].flatten()))
plt.title(f"Autocorrelation for Feature {best_feature}")
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.show()


# In[22]:


import numpy as np

# Hypothetical tensor with shape [samples, timesteps, features]
# For example, let's assume we have 50 samples, 100 timesteps, and 3 features
np.random.seed(0)  # For reproducibility
tensor = np.random.rand(50, 100, 3)

num_samples, num_timesteps, num_features = tensor.shape

# Reshape tensor to combine samples and timesteps into one dimension, resulting in a 2D array
data = tensor.reshape(-1, num_features)

# Function to calculate Pearson correlation coefficient matrix
def pearson_correlation(data):
    """
    Calculate the Pearson correlation coefficient matrix for a dataset.
    data: 2D NumPy array of shape [observations, features]
    Returns: 2D NumPy array representing the correlation matrix.
    """
    mean_centered_data = data - np.mean(data, axis=0)
    std_dev = np.std(data, axis=0)
    correlation_matrix = np.dot(mean_centered_data.T, mean_centered_data) / (std_dev[:, None] * std_dev[None, :] * (len(data) - 1))
    return correlation_matrix

# Calculate correlation matrix
correlation_matrix = pearson_correlation(data)

# Mask the diagonal and upper triangle to avoid duplicate pairs and self-correlation
np.fill_diagonal(correlation_matrix, 0)
lower_triangle_indices = np.tril_indices_from(correlation_matrix, k=-1)

# Find the pair of features with the highest absolute correlation
max_corr_value = np.max(np.abs(correlation_matrix[lower_triangle_indices]))
max_corr_indices = np.argwhere(correlation_matrix == max_corr_value)[0]

print(f"Pair of features with the highest correlation: Feature {max_corr_indices[0]} and Feature {max_corr_indices[1]}")
print(f"Correlation value: {max_corr_value}")

# Optional: Print the correlation matrix
print("Correlation matrix:\n", correlation_matrix)


# In[27]:


correlation_matrix 


# In[31]:


import numpy as np
import pandas as pd
from tslearn.datasets import UCR_UEA_datasets

# Loading the dataset
ts_list, y_true, _, _ = UCR_UEA_datasets().load_dataset('ArticularyWordRecognition')

# Assigning the loaded data to the tensor variable
tensor = ts_list
print("Tensor shape:", tensor.shape)

# Number of features in the tensor is determined by the last dimension of the tensor
num_features = tensor.shape[2]
print("Number of features:", num_features)

# Flatten the tensor across samples for this example, focusing on correlation between features across all timesteps
data_flattened = tensor.reshape(-1, num_features)

# Calculate the Pearson correlation coefficient matrix between features
correlation_matrix = np.corrcoef(data_flattened, rowvar=False)  # False to compute between columns (features)

# Create a DataFrame from the correlation matrix for easier manipulation and visualization
corr_df = pd.DataFrame(correlation_matrix, 
                       columns=[f"Feature {i+1}" for i in range(num_features)],
                       index=[f"Feature {i+1}" for i in range(num_features)])

# Find pairs of features with high correlation
# We exclude the diagonal and upper triangle to avoid repeating pairs and self-comparison
high_corr_pairs = corr_df.where(np.tril(np.ones(corr_df.shape), k=-1).astype(bool))
high_corr_threshold = 0.5  # Define threshold for high correlation
high_corr_pairs = high_corr_pairs.stack().reset_index()
high_corr_pairs.columns = ['Feature 1', 'Feature 2', 'Correlation']
high_corr_pairs = high_corr_pairs[high_corr_pairs['Correlation'] > high_corr_threshold]

print("Pairs of features with high correlation:")
print(high_corr_pairs)


# In[30]:


corr_df


# In[32]:


import pandas as pd

# Initialize a list to hold correlation matrices for each timestep
correlation_matrices = []

# Calculate correlation for each timestep
for timestep in range(num_timesteps):
    timestep_data = tensor[:, timestep, :]  # Data for all samples at this timestep
    corr_matrix = np.corrcoef(timestep_data, rowvar=False)  # Calculate correlation matrix for this timestep
    correlation_matrices.append(corr_matrix)

# Optionally, average these matrices to get an overall sense of feature correlation across all timesteps
avg_corr_matrix = np.mean(correlation_matrices, axis=0)
avg_corr_df = pd.DataFrame(avg_corr_matrix, columns=[f"Feature {i+1}" for i in range(num_features)],
                           index=[f"Feature {i+1}" for i in range(num_features)])

print("Average Correlation Matrix Between Features Across All Timesteps:")
print(avg_corr_df)


# In[33]:


avg_corr_df


# In[34]:


correlation_matrices


# In[35]:


from statsmodels.tsa.stattools import acf

# Assuming tensor is your dataset with shape [samples, timesteps, features]
num_samples, num_timesteps, num_features = tensor.shape

# Calculate and plot autocorrelation for each feature
for feature in range(num_features):
    feature_data = tensor[:, :, feature].flatten()  # Flatten across samples and timesteps
    autocorr = acf(feature_data, nlags=num_timesteps-1, fft=True)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(autocorr, marker='o', linestyle='--')
    plt.title(f'Autocorrelation for Feature {feature + 1}')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.show()


# In[36]:


autocorr


# In[38]:


average_corr_df


# In[39]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'tensor' is your dataset with shape [samples, timesteps, features]
# And you have already calculated 'correlation_matrices' as shown in the previous steps

# Select timesteps to visualize. For example, the first, middle, and last timestep
selected_timesteps = [0, tensor.shape[1] // 2, tensor.shape[1] - 1]
feature_labels = [f"Feature {i+1}" for i in range(tensor.shape[2])]

# Plot correlation matrices for the selected timesteps
for i, timestep in enumerate(selected_timesteps):
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrices[timestep], annot=True, fmt=".2f", cmap="coolwarm",
                xticklabels=feature_labels, yticklabels=feature_labels)
    plt.title(f'Correlation Between Features at Timestep {timestep + 1}')
    plt.show()


# In[40]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'tensor' is your dataset with shape [samples, timesteps, features]

# Calculate the correlation matrix for features at each timestep
num_timesteps = tensor.shape[1]
correlation_matrices = [np.corrcoef(tensor[:, t, :], rowvar=False) for t in range(num_timesteps)]

# Select a subset of timesteps to visualize to avoid too many plots, for example, 10 evenly spaced timesteps
selected_timesteps = np.linspace(0, num_timesteps - 1, 10, dtype=int)

# Plot correlation matrices for the selected timesteps
for timestep in selected_timesteps:
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrices[timestep], annot=True, fmt=".2f", cmap="coolwarm",
                cbar=True, square=True, xticklabels=True, yticklabels=True)
    plt.title(f'Feature Correlation at Timestep {timestep + 1}')
    plt.xlabel('Features')
    plt.ylabel('Features')
    plt.show()


# In[41]:


import numpy as np
import matplotlib.pyplot as plt

# Assuming 'tensor' is your dataset with shape [samples, timesteps, features]

num_timesteps = tensor.shape[1]
correlations_between_timesteps = []

# Calculate correlation between all features at each timestep and the next
for t in range(num_timesteps - 1):
    # Flatten the feature sets for the current and next timestep across all samples
    current_features = tensor[:, t, :].flatten()
    next_features = tensor[:, t+1, :].flatten()
    
    # Calculate the correlation between these two sets of features
    correlation = np.corrcoef(current_features, next_features)[0, 1]
    correlations_between_timesteps.append(correlation)

# Plot the correlations between consecutive timesteps
plt.figure(figsize=(10, 6))
plt.plot(correlations_between_timesteps, marker='o', linestyle='-', color='b')
plt.title('Correlation Between Consecutive Timesteps Across All Features')
plt.xlabel('Timestep')
plt.ylabel('Correlation')
plt.grid(True)
plt.show()


# In[ ]:




