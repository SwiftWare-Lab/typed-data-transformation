import sys
import os
import pandas as pd
from tslearn.datasets import UCR_UEA_datasets


# Predefined list of datasets
dataset_names = [
    'ArticularyWordRecognition', 'AtrialFibrillation', 'BasicMotions', 'CharacterTrajectories',
    'Cricket', 'DuckDuckGeese', 'EigenWorms', 'Epilepsy', 'EthanolConcentration', 'ERing',
    'FaceDetection', 'FingerMovements', 'HandMovementDirection', 'Handwriting',
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
            data_dicts = []

            for instance_index in range(num_instances):
                instance_data = {}
                for var_index in range(num_variables):
                    for time_index in range(num_timepoints):
                        col_name = f"Variable{var_index + 1}_Time{time_index + 1}"
                        instance_data[col_name] = ts_data[instance_index, time_index, var_index]
                data_dicts.append(instance_data)

            df = pd.DataFrame(data_dicts)
            file_path = os.path.join(output_dir, f"{dataset_name}.csv")
            df.to_csv(file_path, index=False)
            
        
        except Exception as e:
            print(f"Failed to process {dataset_name}. Error: {e}")

def read_and_describe_datasets(directory_path):
    for dataset_name in dataset_names:
        file_path = os.path.join(directory_path, f"{dataset_name}.csv")
        
        if not os.path.exists(file_path):
            print(f"File for {dataset_name} not found.")
            continue

        try:
            df = pd.read_csv(file_path)
            # Display basic information
            print(f"Dataset: {dataset_name}")
            print(f"Number of instances: {len(df)}")
            print(f"Number of features: {len(df.columns)}")
            print("First few rows:")
            print(df.head())
            print("Basic statistical description:")
            print(df.describe())
            print("\n\n")  
        except Exception as e:
            print(f"Failed to read {dataset_name}. Error: {e}")

if __name__ == "__main__":
    
    if len(sys.argv) != 3:
        print("Usage: python dataset.py <mode> <directory_path>")
        sys.exit(1)

    mode = sys.argv[1]
    directory_path = sys.argv[2]

    if mode == 'save':
        save_datasets(directory_path)
    elif mode == 'read':
        read_and_describe_datasets(directory_path)
    else:
        print("Invalid mode. Use 'save' or 'read'.")
        sys.exit(1)



