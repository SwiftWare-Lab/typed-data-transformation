import math
import os
import sys
import zstandard as zstd
import pandas as pd
import numpy as np
#from matplotlib import pyplot as plt
import gzip
def run_and_collect_data(dataset_path):
    dataset_path = "/home/jamalids/Documents/2D/data1/TS/"
    datasets = [os.path.join(dp, f) for dp, dn, filenames in os.walk(dataset_path) for f in filenames if f.endswith('.tsv')]
    results = []
    #datasets = [dataset_path]

    for dataset_path in datasets:
        result_row = {}

        ts_data1 = pd.read_csv(dataset_path, delimiter='\t', header=None)
        dataset_name = os.path.basename(dataset_path).replace('.tsv', '')
        print(f"Processing dataset: {dataset_name}")

        group_f = ts_data1.drop(ts_data1.columns[0], axis=1)
        group_f = group_f.iloc[0:2000000, :].T
        group_f = group_f.astype(np.float32).to_numpy().reshape(-1)
        entropy_float_all = calculate_entropy_float(group_f)