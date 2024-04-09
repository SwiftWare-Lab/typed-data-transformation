#!/usr/bin/env python
# coding: utf-8

# In[52]:


import numpy as np
import pandas as pd
import networkx as nx
import fpzip 
from scipy.stats import entropy
from tslearn.datasets import UCR_UEA_datasets

def identify_feature_groups(correlation_matrix, threshold=0.7):
    # Create a graph based on the correlation matrix
    G = nx.Graph()
    for i in correlation_matrix.columns:
        for j in correlation_matrix.columns:
            if i != j and np.abs(correlation_matrix.loc[i, j]) > threshold:
                G.add_edge(i, j)
    
    # Find connected components in the graph as feature groups
    feature_groups = list(nx.connected_components(G))
    
    # Identify individual features (those not in any group)
    all_features = set(correlation_matrix.columns)
    grouped_features = set([item for group in feature_groups for item in group])
    individual_features = list(all_features - grouped_features)
    print(feature_groups)
    print(individual_features)
    return feature_groups, individual_features

def compress_feature(feature_data):
    # Compress individual feature data
    feature_data = feature_data.astype(np.float32)
    compressed_data = fpzip.compress(feature_data)
    compressed_size = len(compressed_data)
    original_size = feature_data.nbytes
    compression_ratio = original_size / max(compressed_size, 1)
    print(original_size)
    return compressed_size, compression_ratio

def compress_feature_group(group_data):
    # Compress group of features
    
    group_data = group_data.astype(np.float32)
    
    compressed_data = fpzip.compress(group_data)
    compressed_size = len(compressed_data)
    original_size = group_data.nbytes
   # original_size =len(group_data)  # calculate original size correctly
    compression_ratio = original_size / max(compressed_size, 1)
    print(compressed_size ) 
    print(original_size)
    return compressed_size, compression_ratio

def calculate_metrics_and_select_compression(ts_data, dataset_name, dataset_type):
    n_samples, n_timesteps, n_features = ts_data.shape
    flat_data = ts_data.reshape(n_samples * n_timesteps, n_features)
    df = pd.DataFrame(flat_data)
    correlation_matrix = df.corr()

    feature_groups, individual_features = identify_feature_groups(correlation_matrix)
    results = []
    all_group_data = None  # Initialize a variable to hold all grouped data
    
    for group_index, group in enumerate(feature_groups):
        indices = [int(index) for index in group]  # Convert to integers if necessary
        group_data = ts_data[:, :, indices]  # Extract the group data without reshaping

        print(group_data.shape)
        
    all_group_data=group_data
    n_samples, n_timesteps, n_features =  all_group_data.shape
    #flattened_group_data = all_group_data.reshape(n_samples , n_timesteps* n_features)
    flattened_group_data = all_group_data.reshape(-1)
    compressed_size, compression_ratio = compress_feature_group(flattened_group_data)
    results.append({
        'Group Index': group_index,  # Use the group index to identify the group
        'Feature Indices': ', '.join(map(str, indices)),  # List of feature indices in the group
        'Compressed Size (bytes)': compressed_size,
        'Compression Ratio': compression_ratio,
        'Grouped': True,
        'Number of Features in Group': len(indices)  # Number of features in this group
       })

    # Handle individual features
    for index in individual_features:
        print("gggggg")
        print(ts_data[:, :, int(index)].shape)
        #feature_data = ts_data[:, :, int(index)].reshape(n_samples * n_timesteps, -1)
        feature_data =ts_data[:, :, int(index)].reshape(-1)
        
        compressed_size, compression_ratio = compress_feature(feature_data)
        results.append({
            'Feature Indices': int(index),
            'Compressed Size (bytes)': compressed_size,
            'Compression Ratio': compression_ratio,
            'Grouped': False
        })

    return pd.DataFrame(results)

# Example usage
ts_data, y_true, _, _ = UCR_UEA_datasets().load_dataset('LSST')
print(ts_data.shape)
results_df = calculate_metrics_and_select_compression(ts_data, 'LSST', 'UCR_UEA')


# In[53]:


results_df


# In[193]:


import numpy as np
import pandas as pd
import networkx as nx
import fpzip 
from scipy.stats import entropy
from tslearn.datasets import UCR_UEA_datasets

def identify_feature_groups(correlation_matrix, threshold=0.8):
    # Create a graph based on the correlation matrix
    G = nx.Graph()
    for i in correlation_matrix.columns:
        for j in correlation_matrix.columns:
            if i != j and np.abs(correlation_matrix.loc[i, j]) > threshold:
                G.add_edge(i, j)
    
    # Find connected components in the graph as feature groups
    feature_groups = list(nx.connected_components(G))
    
    # Identify individual features (those not in any group)
    all_features = set(correlation_matrix.columns)
    grouped_features = set([item for group in feature_groups for item in group])
    individual_features = list(all_features - grouped_features)
    
    
    return feature_groups, individual_features

def compress_feature(feature_data):
    # Compress individual feature data
    feature_data = feature_data.astype(np.float32)
    compressed_data = fpzip.compress(feature_data ,precision=32)
    compressed_size = len(compressed_data)
    original_size = feature_data.nbytes
    compression_ratio = original_size / max(compressed_size, 1)
    
    return compressed_size, compression_ratio,original_size

def compress_feature_group(group_data):
    # Compress group of features
    
    group_data = group_data.astype(np.float32)
    
    compressed_data = fpzip.compress(group_data , precision=32)
    compressed_size = len(compressed_data)
    original_size = group_data.nbytes
   # original_size =len(group_data)  # calculate original size correctly
    compression_ratio = original_size / max(compressed_size, 1)
     
    
    return compressed_size, compression_ratio,original_size

def calculate_metrics_and_select_compression(ts_data, dataset_name, dataset_type):
    n_samples, n_timesteps, n_features = ts_data.shape
    flat_data = ts_data.reshape(n_samples * n_timesteps, n_features)
    df = pd.DataFrame(flat_data)
    correlation_matrix = df.corr()

    feature_groups, individual_features = identify_feature_groups(correlation_matrix)
    feature_results = []  # Store individual feature results
    group_results = []  # Store group results

    # Initialize dictionaries to track total sizes per feature and group
    total_original_sizes = {}
    total_compressed_sizes = {}

    # Handle grouped features
    for group_index, group in enumerate(feature_groups):
        indices = [int(index) for index in group]
        group_data = ts_data[:, :, indices].reshape(-1, len(indices))

        compressed_size, compression_ratio, original_size = compress_feature_group(group_data)
        # Update total sizes for each feature in the group
        for index in indices:
            if index not in total_original_sizes:
                total_original_sizes[index] = original_size / len(indices)  # Assume evenly distributed original size
                total_compressed_sizes[index] = compressed_size / len(indices)  # Evenly distributed compressed size
            else:
                total_original_sizes[index] += original_size / len(indices)
                total_compressed_sizes[index] += compressed_size / len(indices)

        group_results.append({
            'Group Index': group_index,
            'Feature Indices': ', '.join(map(str, indices)),
            'Compressed Size (bytes)': compressed_size,
            'Compression Ratio': compression_ratio,
            'Original Size (bytes)': original_size
        })
        
 
       # Handle individual features
    for index in range(n_features):
        feature_data = ts_data[:, :, index].reshape(-1)
        compressed_size, compression_ratio, original_size = compress_feature(feature_data)
        total_original_sizes[index] = original_size
        total_compressed_sizes[index] = compressed_size
    
        feature_results.append({
        'Feature Index': index,
        'Total Compressed Size (bytes)': compressed_size,
        'Overall Compression Ratio': original_size / compressed_size,
        'Total Original Size (bytes)': original_size
    })

     # Create a summary DataFrame for individual features
    summary_data = [{
        'Feature Index': index,
        'Total Original Size (bytes)': total_original_sizes[index],
        'Total Compressed Size (bytes)': total_compressed_sizes[index],
        'Compression Ratio': total_original_sizes[index] / total_compressed_sizes[index]
    } for index in sorted(total_original_sizes.keys())]

    return pd.DataFrame(feature_results), pd.DataFrame(group_results), pd.DataFrame(summary_data)

# Example usage
#ts_data, y_true, _, _ = UCR_UEA_datasets().load_dataset('LSST')

results_df,group_results,summary = calculate_metrics_and_select_compression(ts_data, 'LSST', 'UCR_UEA')


# In[187]:


import numpy as np
import pandas as pd

# Generate random data for the tensor
n_samples = 1000  # Number of samples
n_timesteps = 10  # Number of timesteps
n_features = 6  # Number of features

# Generate random data for the tensor
tensor_data = np.random.randn(n_samples, n_timesteps, n_features)

# Generate perfect correlation between feature 0 and feature 2
feature0_index = 0
feature2_index = 2

# Generate random data for feature 0
feature0_data = np.random.randn(n_samples, n_timesteps)
tensor_data[:, :, feature0_index] = feature0_data

# Copy feature 0 data to feature 2
tensor_data[:, :, feature2_index] = feature0_data

# Display the shape of the generated tensor
print("Shape of the generated tensor:", tensor_data.shape)

# Convert tensor to dataframe and calculate correlation matrix
flat_data = tensor_data.reshape(n_samples * n_timesteps, n_features)
df = pd.DataFrame(flat_data)
correlation_matrix = df.corr()

# Print correlation matrix
print("\nCorrelation matrix:")
print(correlation_matrix)


# In[189]:


correlation_matrix 


# In[190]:


results_df


# In[191]:


group_results


# In[113]:


summary


# In[95]:


results_df


# In[96]:


summary


# In[148]:


ts_data, y_true, _, _ = UCR_UEA_datasets().load_dataset('LSST')

results_df,group_results,summary = calculate_metrics_and_select_compression(ts_data, 'LSST', 'UCR_UEA')

def compress_feature(feature_data):
    #print(feature_data.shape)
    feature_data = feature_data.astype(np.float32)
    original_size = feature_data.nbytes
    
    data_bytes = feature_data.tobytes()
   
    # Compress individual feature data
    
    compressed_data = fpzip.compress(feature_data , precision=32)
    compressed_size = len(compressed_data)
    original_size = feature_data.nbytes
    
    data_bytes = feature_data.tobytes()
   
    compression_ratio = original_size / max(compressed_size, 1)
    
    return compressed_size, compression_ratio,original_size
n_samples, n_timesteps, n_features = ts_data.shape
for timestep_index in range(n_timesteps):
        
            feature_data = ts_data[:, timestep_index, :].reshape(-1)
            feature_data = feature_data.astype(np.float32)
           
            data_bytes = feature_data.tobytes()
            
            original_size = feature_data.nbytes
           
            compressed_size, compression_ratio,original_size=compress_feature(feature_data)
            print(timestep_index,compressed_size, compression_ratio,original_size)
        


# In[150]:


ts_data, y_true, _, _ = UCR_UEA_datasets().load_dataset('LSST')

results_df,group_results,summary = calculate_metrics_and_select_compression(ts_data, 'LSST', 'UCR_UEA')

def compress_feature(feature_data):
    #print(feature_data.shape)
    feature_data = feature_data.astype(np.float32)
    original_size = feature_data.nbytes
    
    data_bytes = feature_data.tobytes()
   
    # Compress individual feature data
    
    compressed_data = fpzip.compress(feature_data , precision=32)
    compressed_size = len(compressed_data)
    original_size = feature_data.nbytes
    
    data_bytes = feature_data.tobytes()
   
    compression_ratio = original_size / max(compressed_size, 1)
n_samples, n_timesteps, n_features = ts_data.shape    
for features_index in range(n_features):
        
            feature_data = ts_data[:, :, features_index].reshape(-1)
            feature_data = feature_data.astype(np.float32)
           
            data_bytes = feature_data.tobytes()
            
            original_size = feature_data.nbytes
           
            compressed_size, compression_ratio,original_size=compress_feature(feature_data)
            print(timestep_index,compressed_size, compression_ratio,original_size)
        


# In[152]:


import numpy as np
from tslearn.datasets import UCR_UEA_datasets
import fpzip

def compress_feature(feature_data):
    # Compress individual feature data
    compressed_data = fpzip.compress(feature_data, precision=32)
    compressed_size = len(compressed_data)
    original_size = feature_data.nbytes
    compression_ratio = original_size / max(compressed_size, 1)
    return compressed_size, compression_ratio, original_size

# Load dataset
ts_data, y_true, _, _ = UCR_UEA_datasets().load_dataset('LSST')

n_samples, n_timesteps, n_features = ts_data.shape

# Loop over features
for features_index in range(n_features):
    print(f"Feature {features_index + 1}:")
    
    # Loop over time steps
    for timestep_index in range(0, n_timesteps, 10):
        # Extract feature data for 5 time steps
        feature_data = ts_data[:, timestep_index:timestep_index+10, features_index].reshape(-1)
        feature_data = feature_data.astype(np.float32)
        
        # Compress feature data
        compressed_size, compression_ratio, original_size = compress_feature(feature_data)
        
        # Print compression results
        print(f"Time steps {timestep_index}-{timestep_index+4}:")
        print(f"Compressed size: {compressed_size} bytes")
        print(f"Compression ratio: {compression_ratio}")
        print(f"Original size: {original_size} bytes")
        print()


# In[ ]:


# Handle individual features
    for index in individual_features:
        total_feature_original_size = 0
        total_feature_compressed_size = 0
        for timestep_index in range(n_timesteps):
            feature_data = ts_data[:, timestep_index, int(index)].reshape(-1)
            compressed_size, compression_ratio, original_size = compress_feature(feature_data)
            total_feature_original_size += original_size
            total_feature_compressed_size += compressed_size
        
        total_original_sizes[index] = total_feature_original_size
        total_compressed_sizes[index] = total_feature_compressed_size

        feature_results.append({
            'Feature Index': index,
            'Total Compressed Size (bytes)': total_feature_compressed_size,
            'Overall Compression Ratio': total_feature_original_size / total_feature_compressed_size,
            'Total Original Size (bytes)': total_feature_original_size
        })

    # Create a summary DataFrame for individual features
    summary_data = [{
        'Feature Index': index,
        'Total Original Size (bytes)': total_original_sizes[index],
        'Total Compressed Size (bytes)': total_compressed_sizes[index],
        'Compression Ratio': total_original_sizes[index] / total_compressed_sizes[index]
    } for index in sorted(total_original_sizes.keys())]

    return pd.DataFrame(feature_results), pd.DataFrame(group_results), pd.DataFrame(summary_data)


# In[201]:


import numpy as np
import pandas as pd
import networkx as nx
import fpzip 
from scipy.stats import entropy
from tslearn.datasets import UCR_UEA_datasets

def identify_feature_groups(correlation_matrix, threshold=0.3):
    # Create a graph based on the correlation matrix
    G = nx.Graph()
    for i in correlation_matrix.columns:
        for j in correlation_matrix.columns:
            if i != j and np.abs(correlation_matrix.loc[i, j]) > threshold:
                G.add_edge(i, j)
    
    # Find connected components in the graph as feature groups
    feature_groups = list(nx.connected_components(G))
    
    # Identify individual features (those not in any group)
    all_features = set(correlation_matrix.columns)
    grouped_features = set([item for group in feature_groups for item in group])
    individual_features = list(all_features - grouped_features)
    
    
    return feature_groups, individual_features

def compress_feature(feature_data):
    # Compress individual feature data
    feature_data = feature_data.astype(np.float32)
    compressed_data = fpzip.compress(feature_data ,precision=32)
    compressed_size = len(compressed_data)
    original_size = feature_data.nbytes
    compression_ratio = original_size / max(compressed_size, 1)
    
    return compressed_size, compression_ratio,original_size

def compress_feature_group(group_data):
    # Compress group of features
    
    group_data = group_data.astype(np.float32)
    
    compressed_data = fpzip.compress(group_data , precision=32)
    compressed_size = len(compressed_data)
    original_size = group_data.nbytes
   # original_size =len(group_data)  # calculate original size correctly
    compression_ratio = original_size / max(compressed_size, 1)
     
    
    return compressed_size, compression_ratio,original_size

def calculate_metrics_and_select_compression(ts_data, dataset_name, dataset_type):
    n_samples, n_timesteps, n_features = ts_data.shape
    flat_data = ts_data.reshape(n_samples * n_timesteps, n_features)
    df = pd.DataFrame(flat_data)
    correlation_matrix = df.corr()

    feature_groups, individual_features = identify_feature_groups(correlation_matrix)
    feature_results = []  # Store individual feature results
    group_results = []  # Store group results

    # Initialize dictionaries to track total sizes per feature and group
    total_original_sizes = {}
    total_compressed_sizes = {}

    # Handle grouped features
    for group_index, group in enumerate(feature_groups):
        indices = [int(index) for index in group]
        group_data = ts_data[:, :, indices].reshape(-1)

        # Compress all data at once for the group
        print(group_data.shape)
        compressed_size, compression_ratio, original_size = compress_feature_group(group_data)
        
        # Update total sizes for each feature in the group
        for index in indices:
            if index not in total_original_sizes:
                total_original_sizes[index] = original_size / len(indices)  # Assume evenly distributed original size
                total_compressed_sizes[index] = compressed_size / len(indices)  # Evenly distributed compressed size
            else:
                total_original_sizes[index] += original_size / len(indices)
                total_compressed_sizes[index] += compressed_size / len(indices)

        group_results.append({
            'Group Index': group_index,
            'Feature Indices': ', '.join(map(str, indices)),
            'Compressed Size (bytes)': compressed_size,
            'Compression Ratio': compression_ratio,
            'Original Size (bytes)': original_size
        })
        
    # Handle individual features
    for index in range(n_features):
        feature_data = ts_data[:, :, index].reshape(-1)
        compressed_size, compression_ratio, original_size = compress_feature(feature_data)
        total_original_sizes[index] = original_size
        total_compressed_sizes[index] = compressed_size
    
        feature_results.append({
            'Feature Index': index,
            'Total Compressed Size (bytes)': compressed_size,
            'Overall Compression Ratio': original_size / compressed_size,
            'Total Original Size (bytes)': original_size
        })

    # Create a summary DataFrame for individual features
    summary_data = [{
        'Feature Index': index,
        'Total Original Size (bytes)': total_original_sizes[index],
        'Total Compressed Size (bytes)': total_compressed_sizes[index],
        'Compression Ratio': total_original_sizes[index] / total_compressed_sizes[index]
    } for index in sorted(total_original_sizes.keys())]

    return pd.DataFrame(feature_results), pd.DataFrame(group_results), pd.DataFrame(summary_data)

# Example usage
ts_data, y_true, _, _ = UCR_UEA_datasets().load_dataset('LSST')
results_df,group_results,summary = calculate_metrics_and_select_compression(ts_data, 'LSST', 'UCR_UEA')


# In[202]:


results_df


# In[203]:


group_results


# In[ ]:


(442620,)

