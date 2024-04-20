#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from tslearn.datasets import UCR_UEA_datasets

UCR_UEA_datasets_list = UCR_UEA_datasets().list_multivariate_datasets()
ts_list, y_true, _, _ = UCR_UEA_datasets().load_dataset('Cricket')

tensor1 = ts_list[0:1,0:1000, :]
tensor=tensor1


num_features = tensor.shape[2]
print(num_features)

timesteps = range(tensor.shape[1])

plt.figure(figsize=(14, 8))

for feature in range(num_features):
    feature_average = tensor[:, :, feature].mean(axis=0)
    
    plt.plot(timesteps, feature_average, label=f'Feature {feature + 1} Trend', marker='o')

plt.title('Trend of Each Feature Over Time')
plt.xlabel('Timestep')
#plt.ylabel('Average Value')
#plt.xticks(timesteps)
#plt.grid(True)
plt.legend()
plt.show()

