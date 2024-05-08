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
        #print(f"Dataset: {dataset_name}")
        print(f"Number of instances: {len(df)}")
        print(f"Number of features: {len(df.columns)}")
        #print("First few rows:")
        #print(df.head())
        #print("Basic statistical description:")
        #print(df.describe())
        print("\n\n")
    except Exception as e:
        print(f"Failed to read the dataset. Error: {e}")


# TODO: please add all other compression tools

def arg_parser():
    import argparse
    parser = argparse.ArgumentParser(description='compress one dataset and store the log.')
    parser.add_argument('--dataset', dest='dataset_path', help='Path to the UCR dataset tsv/csv.')
    parser.add_argument('--variant', dest='variant', default="dictionary", help='Variant of the algorithm.')
    parser.add_argument('--pattern', dest='pattern', default="10*16", help='Pattern to match the files.')
    # TODO: make sure to append to the CSV file
    parser.add_argument('--outcsv', dest='log_file', default="./log_out.csv", help='Output directory for the sbatch scripts.')
    parser.add_argument('--nthreads', dest='num_threads', default=1, type=int, help='Number of threads to use.')
    return parser


if __name__ == "__main__":
    parser = arg_parser()
    args = parser.parse_args()
    dataset_path = args.dataset_path
    comp_variant = args.variant
    pattern = args.pattern
    log_file = args.log_file
    num_threads = args.num_threads

    print(f"Compressing dataset: {dataset_path} with variant: {comp_variant} and pattern: {pattern}..., log file: {log_file}..., num_threads: {num_threads}...")


    read_and_describe_dataset(dataset_path)

