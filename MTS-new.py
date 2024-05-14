#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import sys
import os
import pandas as pd
from tslearn.datasets import UCR_UEA_datasets
# Predefined list of datasets
dataset_names = [
     'FingerMovements', 'HandMovementDirection', 'Handwriting',
    'Heartbeat', 'InsectWingbeat', 'JapaneseVowels', 'Libras', 'LSST', 'MotorImagery',
    'NATOPS', 'PenDigits', 'PEMS-SF', 'Phoneme', 'RacketSports', 'SelfRegulationSCP1',
    'SelfRegulationSCP2', 'SpokenArabicDigits', 'StandWalkJump', 'UWaveGestureLibrary'
]

def save_datasets(output_dir):
    UCR_UEA_datasets_list = UCR_UEA_datasets().list_multivariate_datasets()
    UCR_UEA_datasets_list.remove('InsectWingbeat')
    UCR_UEA_datasets_list.remove('AtrialFibrillation')

    for dataset_name in UCR_UEA_datasets_list:
        try:
            ts_data, _, _, _ = UCR_UEA_datasets().load_dataset(dataset_name)
            num_instances, num_timepoints, num_variables = ts_data.shape

            # Initialize a list to hold dictionaries representing each instance
            data_dicts = []
            # Loop through variables to populate data_dicts
            for var_index in range(num_variables):
                for instance_index in range(num_instances):
                    instance_data = {'InstanceIndex': instance_index + 1, 'VariableIndex': var_index + 1}
                    for time_index in range(num_timepoints):
                        col_name = f"Feature_Time{time_index + 1}"
                        instance_data[col_name] = ts_data[instance_index, time_index, var_index]
                    data_dicts.append(instance_data.copy())  # Append instance_data for each instance

            # Convert data_dicts to DataFrame
            df = pd.DataFrame(data_dicts)
            
            # Save DataFrame to a TSV file
            file_path = os.path.join(output_dir, f"{dataset_name}.tsv")
            df.to_csv(file_path, sep='\t', index=False)
            
        except Exception as e:
            print(f"Failed to process {dataset_name}. Error: {e}")

file_path = '/home/jamalids/Documents/dataucr'
save_datasets(file_path)


# In[4]:


from tslearn.datasets import UCR_UEA_datasets 
ts_data, _, _, _ = UCR_UEA_datasets().load_dataset('BasicMotions')
num_instances, num_timepoints, num_variables = ts_data.shape


# In[5]:


print(num_instances, num_timepoints, num_variables)


# In[ ]:


dataset_names = [
    'ArticularyWordRecognition', 'AtrialFibrillation', 'BasicMotions', 'CharacterTrajectories',
    'Cricket', 'DuckDuckGeese', 'EigenWorms', 'Epilepsy', 'EthanolConcentration', 'ERing',
    'FaceDetection', 'FingerMovements', 'HandMovementDirection', 'Handwriting',
    'Heartbeat', 'InsectWingbeat', 'JapaneseVowels', 'Libras', 'LSST', 'MotorImagery',
    'NATOPS', 'PenDigits', 'PEMS-SF', 'Phoneme', 'RacketSports', 'SelfRegulationSCP1',
    'SelfRegulationSCP2', 'SpokenArabicDigits', 'StandWalkJump', 'UWaveGestureLibrary'
]

