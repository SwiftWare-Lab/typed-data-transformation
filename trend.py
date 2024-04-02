
import numpy as np
import pandas as pd
from scipy.stats import linregress
from statsmodels.tsa.stattools import adfuller
from scipy.fftpack import fft
from scipy.signal import find_peaks
from tslearn.datasets import UCR_UEA_datasets

def identify_trend(data):
    if not data.size:
        return "Empty"
    result = adfuller(data)
    if result[1] < 0.05:
        return "Stationary"
    x = np.arange(len(data))
    slope, _, r_value, _, _ = linregress(x, data)
    if abs(r_value) > 0.8:
        return "Linear" if slope != 0 else "Stationary"
    if np.all(data > 0):
        log_data = np.log(data)
        _, _, r_value_log, _, _ = linregress(x, log_data)
        if abs(r_value_log) > 0.8:
            return "Exponential"
    fft_data = fft(data - np.mean(data))
    frequencies = np.abs(fft_data)
    peaks, _ = find_peaks(frequencies[1:len(frequencies)//2], height=np.max(frequencies)/4)
    if len(peaks) > 1:
        return "Cyclical"
    elif len(peaks) == 1:
        return "Seasonal"
    return "Unclear"

def process_dataset(ts_list, dataset_name):
    n_samples, n_timesteps, n_features = ts_list.shape
    results = []
    for features_index in range(n_features):
        feature_data = ts_list[:, :, features_index].reshape(-1)
        trend = identify_trend(feature_data)
        results.append({'Dataset': dataset_name, 'Feature': features_index, 'Trend': trend})
    return results

def process_and_save_datasets_to_one_csv(datasets):
    all_results = []
    for dataset in datasets:
        dataset_name = dataset['name']
        try:
            ts_list, _, _, _ = UCR_UEA_datasets().load_dataset(dataset_name)
            if ts_list is not None:
                dataset_results = process_dataset(ts_list, dataset_name)
                all_results.extend(dataset_results)
            else:
                print(f"Dataset {dataset_name} could not be loaded. Skipping.")
        except Exception as e:
            print(f"Failed to process dataset {dataset_name} due to an error: {e}. Skipping.")
    df = pd.DataFrame(all_results)
    return df

def summarize_trends(dataframe):
    if dataframe.empty:
        return [{"Dataset": "N/A", "Summary": "DataFrame is empty.", "Trend": "N/A"}]

    summaries = []
    grouped = dataframe.groupby('Dataset')
    for dataset_name, group in grouped:
        last_trend = None
        trend_changes = []
        for i, row in group.iterrows():
            current_trend = row['Trend']
            if current_trend != last_trend:
                trend_changes.append(current_trend)
                last_trend = current_trend
        summary_text = ", ".join([f"{trend}" for trend in trend_changes])
        summaries.append({"Dataset": dataset_name, "Trend Changes": summary_text})
    return summaries

UCR_UEA_datasets_list = UCR_UEA_datasets().list_multivariate_datasets()
datasets = [{'name': 'Libras'},
 {'name': 'LSST'},
 {'name': 'MotorImagery'},
 {'name': 'NATOPS'},
 {'name': 'PenDigits'},
 {'name': 'PEMS-SF'},
 {'name': 'Phoneme'},]
UCR_UEA_datasets_list.remove('InsectWingbeat')
UCR_UEA_datasets_list.remove('AtrialFibrillation')
UCR_UEA_datasets_list.remove('EigenWorms')
UCR_UEA_datasets_list.remove('EthanolConcentration')
#datasets = [{'name': name} for name in UCR_UEA_datasets_list]
df_results = process_and_save_datasets_to_one_csv(datasets)
summary_list = summarize_trends(df_results)
summary_df = pd.DataFrame(summary_list)

print("All dataset results saved to combined_dataset_trends.csv")
print("Summary of dataset trends saved to summary_dataset_trends.csv")


df_results2=df_results

