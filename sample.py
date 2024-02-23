#!/usr/bin/env python
# coding: utf-8

# In[13]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import silhouette_score
import numpy as np
import time
from aeon.datasets import load_classification

start_time = time.time()

# Load a dataset
X_train, y_train = load_classification('Libras')  # Assuming  returns 3D numpy arrays

# Flatten the last two dimensions for StandardScaler
num_samples, channels, features = X_train.shape
X_train_flattened = X_train.reshape(num_samples, channels * features)

# Standardize the dataset
scaler = StandardScaler()
X_scaled_flattened = scaler.fit_transform(X_train_flattened)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X_scaled_flattened, dtype=torch.float32)

print(f"Data loading and preprocessing time: {time.time() - start_time:.2f} seconds")

#  Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, num_features):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, num_features),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

autoencoder = Autoencoder(channels * features)
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

# DataLoader for the dataset
dataset = TensorDataset(X_tensor, X_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Train the Autoencoder
train_start_time = time.time()
num_epochs = 20
for epoch in range(num_epochs):
    for inputs, _ in dataloader:
        optimizer.zero_grad()
        outputs = autoencoder(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
print(f"Autoencoder training time: {time.time() - train_start_time:.2f} seconds")

#  encode_data function
def encode_data(model, dataloader):
    model.eval()
    encoded_features = []
    with torch.no_grad():
        for inputs, _ in dataloader:
            encoded = model.encoder(inputs)
            encoded_features.append(encoded)
    return torch.cat(encoded_features, dim=0)

# Extract features
encode_start_time = time.time()
encoded_features = encode_data(autoencoder, dataloader)
print(f"Feature encoding time: {time.time() - encode_start_time:.2f} seconds")

encoded_features_np = encoded_features.numpy()

# Clustering number(adding code to finding best number of cluster )
n_clusters = 3

# Apply KMeans Clustering
kmeans_start_time = time.time()
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters_kmeans = kmeans.fit_predict(encoded_features_np)
print(f"KMeans clustering time: {time.time() - kmeans_start_time:.2f} seconds")

# Apply Hierarchical Clustering
hierarchical_start_time = time.time()
hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
clusters_hierarchical = hierarchical.fit_predict(encoded_features_np)
print(f"Hierarchical clustering time: {time.time() - hierarchical_start_time:.2f} seconds")

# Apply Spectral Clustering
spectral_start_time = time.time()
spectral = SpectralClustering(n_clusters=n_clusters, assign_labels='discretize', random_state=42)
clusters_spectral = spectral.fit_predict(encoded_features_np)
print(f"Spectral clustering time: {time.time() - spectral_start_time:.2f} seconds")

#silhouette scores for KMeans
print("Silhouette Score for KMeans:", silhouette_score(encoded_features_np, clusters_kmeans))
#silhouette Hierarchical Clustering
print("Silhouette Score for hierarchical:", silhouette_score(encoded_features_np, clusters_hierarchical))

# Total execution time
print(f"Total execution time: {time.time() - start_time:.2f} seconds")





# In[ ]:




