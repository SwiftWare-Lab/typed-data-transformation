import sys
import os
import pandas as pd

def read_and_describe_dataset(dataset_path):
    if not os.path.exists(dataset_path):
        print(f"File not found: {dataset_path}")
        return

    try:
        df = pd.read_csv(dataset_path)
        # Display basic information about the dataset
        dataset_name = os.path.basename(dataset_path).replace('.csv', '')
        print(f"Dataset: {dataset_name}")
        print(f"Number of instances: {len(df)}")
        print(f"Number of features: {len(df.columns)}")
        print("First few rows:")
        print(df.head())
        print("Basic statistical description:")
        print(df.describe())
        print("\n\n")  
    except Exception as e:
        print(f"Failed to read the dataset. Error: {e}")

if __name__ == "__main__":
    
    if len(sys.argv) != 2:
        print("Usage: python dataset.py <dataset_path>")
        sys.exit(1)

    dataset_path = sys.argv[1]
    read_and_describe_dataset(dataset_path)

