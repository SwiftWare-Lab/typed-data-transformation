import os
import numpy as np
import pandas as pd

# Load the datasets
dataset_path = "/home/jamalids/Documents/2D/data1/"
#datasets = [dataset_path]
datasets = [os.path.join(dp, f) for dp, dn, filenames in os.walk(dataset_path) for f in filenames if f.endswith('.tsv')]
results = []


def count_consecutive_identical_values(data):
    if len(data) == 0:
        return {}

    counts = {}
    current_value = data[0]
    current_length = 1

    for value in data[1:]:
        if value == current_value:
            current_length += 1
        else:
            # Store counts of each unique (value, length) pair
            key = (current_value, current_length)
            counts[key] = counts.get(key, 0) + 1
            current_value = value
            current_length = 1

    # Include the last sequence in the count
    key = (current_value, current_length)
    counts[key] = counts.get(key, 0) + 1

    return counts


for dataset_path in datasets:
    ts_data1 = pd.read_csv(dataset_path, delimiter='\t', header=None)
    ts_data1 = ts_data1.drop(ts_data1.columns[0], axis=1)  # Assume the first column is not needed
    ts_data1 = ts_data1.T  # Transpose if necessary
    ts_data1 = ts_data1.astype(np.float64).to_numpy().reshape(-1)  # Flatten the array
    dataset_name = os.path.basename(dataset_path).replace('.tsv', '')

    consecutive_counts = count_consecutive_identical_values(ts_data1)

    # Sort the dictionary by the length of sequences, then by frequency of those lengths
    sorted_consecutive_counts = sorted(consecutive_counts.items(), key=lambda x: (x[0][1], x[1]), reverse=True)[:5]

    # Prepare the data for export
   # statistics = {
     #   "dataset_name": dataset_name,
      #  'Top Five Consecutive Values': [val[0] for val, _ in sorted_consecutive_counts],
       # 'Top Five Consecutive Lengths': [val[1] for val, _ in sorted_consecutive_counts],
      #  'Top Five Frequencies of Lengths': [freq for _, freq in sorted_consecutive_counts]
   # }
    # Prepare the data for export by structuring it into separate fields for each of the top five entries
    if len(sorted_consecutive_counts) >= 5:
        statistics = {
            "dataset_name": dataset_name,
            'Top1_Value': sorted_consecutive_counts[0][0][0],
            'Top1_Length': sorted_consecutive_counts[0][0][1],
            'Top1_Frequency': sorted_consecutive_counts[0][1],
            'Top2_Value': sorted_consecutive_counts[1][0][0],
            'Top2_Length': sorted_consecutive_counts[1][0][1],
            'Top2_Frequency': sorted_consecutive_counts[1][1],
            'Top3_Value': sorted_consecutive_counts[2][0][0],
            'Top3_Length': sorted_consecutive_counts[2][0][1],
            'Top3_Frequency': sorted_consecutive_counts[2][1],
            'Top4_Value': sorted_consecutive_counts[3][0][0],
            'Top4_Length': sorted_consecutive_counts[3][0][1],
            'Top4_Frequency': sorted_consecutive_counts[3][1],
            'Top5_Value': sorted_consecutive_counts[4][0][0],
            'Top5_Length': sorted_consecutive_counts[4][0][1],
            'Top5_Frequency': sorted_consecutive_counts[4][1]
        }
    else:
        # In case there are less than 5 entries, fill remaining with None
        statistics = {"dataset_name": dataset_name}
        for i in range(1, 6):
            if i - 1 < len(sorted_consecutive_counts):
                statistics[f'Top{i}_Value'] = sorted_consecutive_counts[i - 1][0][0]
                statistics[f'Top{i}_Length'] = sorted_consecutive_counts[i - 1][0][1]
                statistics[f'Top{i}_Frequency'] = sorted_consecutive_counts[i - 1][1]
            else:
                statistics[f'Top{i}_Value'] = None
                statistics[f'Top{i}_Length'] = None
                statistics[f'Top{i}_Frequency'] = None

    results.append(statistics)

   # results.append(statistics)

# Convert to DataFrame and save to CSV
statistics_df = pd.DataFrame(results)
statistics_df.to_csv("top_longest_consecutive_statistics1.csv", index=False)
print("Statistics saved to top_longest_consecutive_statistics.csv")