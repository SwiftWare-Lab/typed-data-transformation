import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import time
from kneed import KneeLocator
from aeon.datasets import load_classification
from typing import Optional, Literal
from tslearn.datasets import UCR_UEA_datasets
from aeon.datasets.tsc_data_lists import univariate2015, univariate, multivariate, univariate_equal_length, multivariate_equal_length
import os


def pipeline(dataset_type: Literal['UCR_UEA', 'AEON'], dataset_name: Optional[str] = None) -> (np.ndarray, np.ndarray):
    print(f'Processing dataset: {dataset_name}')
    try:
        if dataset_type == 'UCR_UEA':
            X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset(dataset_name)
        elif dataset_type == 'AEON':
            X_train, y_train = load_classification(dataset_name)
        else:
            raise ValueError(f'Invalid dataset type: {dataset_type}')
        return X_train, y_train
    except Exception as e:
        print(f"Failed to load dataset {dataset_name} of type {dataset_type}. Error: {e}")
        return None, None


class CNNAutoencoder(nn.Module):
    def __init__(self, sequence_length, num_channels):
        super(CNNAutoencoder, self).__init__()
        self.sequence_length = sequence_length
        self.num_channels = num_channels
        self.encoder = nn.Sequential(
            nn.Conv1d(num_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool1d(2, stride=2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(8, 16, kernel_size=2, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose1d(16, num_channels, kernel_size=2, stride=2),
            nn.ReLU(True),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        # Resize the output to ensure it matches the input size exactly(if automatic adjustments are insufficient)
        x = nn.functional.interpolate(x, size=self.sequence_length)
        return x


def encode_data(model, dataloader):
    model.eval()
    encoded_features = []
    with torch.no_grad():
        for inputs, _ in dataloader:
            encoded = model.encoder(inputs)
            encoded_features.append(encoded)
    return torch.cat(encoded_features, dim=0)

def find_optimal_clusters(data, max_k):
    iters = range(2, max_k + 1)
    sse = []
    for k in iters:
        kmeans = KMeans(n_clusters=k, random_state=42).fit(data)
        sse.append(kmeans.inertia_)
    kneedle = KneeLocator(iters, sse, curve='convex', direction='decreasing')
    optimal_k = kneedle.elbow
    return optimal_k if optimal_k is not None else 2  

def silhouette_analysis(data, max_k):
    silhouette_scores = []
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42).fit(data)
        score = silhouette_score(data, kmeans.labels_)
        silhouette_scores.append(score)
    optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2 
    return optimal_k, silhouette_scores

# Main processing loop
UCR_UEA_datasets_list =  UCR_UEA_datasets().list_multivariate_datasets()
UCR_UEA_datasets_list.remove('InsectWingbeat')
UCR_UEA_datasets_list.remove('Epilepsy')#Given input size: (8x1x1). Calculated output size: (8x1x0). Output size is too small
UCR_UEA_datasets_list.remove('EthanolConcentration')#Output size is too small
UCR_UEA_datasets_list.remove('Handwriting')
UCR_UEA_datasets_list.remove('Libras')
AEON_datasets_list = list(multivariate)
AEON_datasets_list.remove('InsectWingbeat')
AEON_datasets_list.remove('CharacterTrajectories')
AEON_datasets_list.remove('SpokenArabicDigits')
AEON_datasets_list.remove('JapaneseVowels')
    

# Prepare datasets 
datasets = [{'name': name, 'type': 'UCR_UEA'} for name in UCR_UEA_datasets_list] +            [{'name': name, 'type': 'AEON'} for name in AEON_datasets_list]

results = []
def train_autoencoder(dataloader, autoencoder, criterion, optimizer):
    for epoch in range(10):  # training loop for demonstration
        for data in dataloader:
            try:
                optimizer.zero_grad()
                X_tensor, _ = data  
                outputs = autoencoder(X_tensor)
                loss = criterion(outputs, X_tensor)
                loss.backward()
                optimizer.step()
            except RuntimeError as e:
                if "Output size is too small" in str(e):
                    print(f"Skipping dataset due to RuntimeError: {e}")
                    return False  #  skip the rest of the dataset processing
                else:
                    raise  
    return True  

for dataset in datasets:
    
    start_time = time.time()
    X_train, y_train = pipeline(dataset_type=dataset['type'], dataset_name=dataset['name'])
    load_time = time.time() - start_time
    if X_train is None or y_train is None:
        print(f"Skipping dataset {dataset['name']} due to loading issues.")
        continue 
    
    num_samples, channels, sequence_length = X_train.shape
    X_train_reshaped = X_train.reshape(num_samples * channels, sequence_length)
    scaler = StandardScaler()
    X_scaled_reshaped = scaler.fit_transform(X_train_reshaped)
    X_scaled = X_scaled_reshaped.reshape(num_samples, channels, sequence_length)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    autoencoder = CNNAutoencoder(sequence_length, channels)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
    dataloader = DataLoader(TensorDataset(X_tensor, X_tensor), batch_size=32, shuffle=True)
    success = train_autoencoder(dataloader, autoencoder, criterion, optimizer)
    if not success:
        print(f"Skipped {dataset['name']} due to critical error during training.")
        continue  # Move to the next dataset 
    
    
    train_time = time.time() - start_time
    
    # Data encoding
    start_time = time.time()
    #encoded_features_np = encode_data(autoencoder, dataloader).view(-1, autoencoder.encoder[3].out_channels * 64).detach().numpy()
    encoded_features_np = encode_data(autoencoder, dataloader).flatten(start_dim=1).detach().numpy()
    encode_time = time.time() - start_time
    
    # Clustering
    start_time = time.time()
    optimal_k = find_optimal_clusters(encoded_features_np, 10)
    kmeans_time = time.time() - start_time
    
    start_time = time.time()
    silhouette_avg = silhouette_score(encoded_features_np, KMeans(n_clusters=optimal_k, random_state=42).fit_predict(encoded_features_np))
    silhouette_time = time.time() - start_time
    
    results.append({
        'Dataset Name': dataset['name'],
        'Type': dataset['type'],
        'Optimal Clusters': optimal_k,
        'Silhouette Score': silhouette_avg,
        'Load Time (s)': load_time,
        'Training Time (s)': train_time,
        'Encoding Time (s)': encode_time,
        'Finding K Time (s)': kmeans_time,
        'Silhouette Calc Time (s)': silhouette_time
    })

results_df = pd.DataFrame(results)
print(results_df)
# Save DataFrame to CSV
if not os.path.exists('./output'):
           os.makedirs('./output') 
filename = "results.csv"
csv_save_path = os.path.join('./output', filename)
results_df.to_csv(csv_save_path, index=False)







