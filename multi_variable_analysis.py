import os
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load data
if len(sys.argv) != 2:
    print("Usage: python analyze_dataset.py <dataset_name>")
    sys.exit(1)
dataset_name = sys.argv[1]
dataset_name_striped = dataset_name.split('/')[-1].split('.')[0]
data_real = pd.read_csv(dataset_name)
# remove the second type column
data_real = data_real.drop(columns=['TYPE STR'])
# data_real.info()
t = data_real.describe()
t = data_real.isna().any()
# print(t)
correlation_matrix = data_real.corr()
fig = plt.figure(figsize=(20, 18))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
if not os.path.exists('./plot'):
    os.makedirs('./plot')
# make output plot name from datasetname
# strip dataset name from postfix and path
out_plot = os.path.join('plot', 'correlation_matrix_plot.png')
plt.savefig(out_plot)
plt.show()


## use principal component analysis to reduce the dimensionality of the dataset
# Standardize the Data
scaler = StandardScaler()
scaler.fit(data_real)
scaled_data = scaler.transform(data_real)
features = data_real.columns
target = 'TYPE'
# exclude the target column
features = features.drop(target)

# Separating out the features
x = data_real.loc[:, features].values

# Separating out the target
y = data_real.loc[:, [target]].values

# Standardizing the features
x = StandardScaler().fit_transform(x)
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data=principalComponents
                           , columns=['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, data_real[[target]]], axis=1)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Principal Component 1', fontsize=15)
ax.set_ylabel('Principal Component 2', fontsize=15)
ax.set_title('Two component PCA', fontsize=20)

targets = [0, 1]
colors = ['r', 'g']
for targett, color in zip(targets, colors):
    indicesToKeep = finalDf[target] == targett
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c=color
               , s=50)
ax.legend(['BOT', 'OC'])
out_plot = os.path.join('plot', 'PCA2.png')
plt.savefig(out_plot)
plt.show()
