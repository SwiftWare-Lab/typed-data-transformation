#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
from memory_profiler import profile
import time

# Load the data
data = pd.read_csv('/home/jamalids/Downloads/Synthetic_OC_Blood_Routine_50k.csv')


# Process y based on TYPE.1 values
for i in range(len(data)):
    if data['TYPE'].iloc[i] != 0 and data['TYPE'].iloc[i] != 1:
        data.iloc[i] = 1 if data['TYPE.1'].iloc[i] == 'OC' else 0


data_frame = data.iloc[:, :-1]  # Assuming the last column is not part of the features

@profile
def train_and_evaluate(data_frame):
    start_time = time.time()

    # Define features and target
    feature_column_names = data_frame.columns[:-1]  # Excluding the last column for featuresprint
    print(feature_column_names)
    predicted_class_name = 'TYPE'
    #type_1_column = 'TYPE.1'  # Adjust this based on your DataFrame structure

    X = data_frame[feature_column_names].values
    y = data_frame[predicted_class_name].values

    
    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardizing the data
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Training the Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train, y_train.ravel())

    # Predictions and metrics
    y_train_pred = model.predict(X_train)
    training_accuracy = accuracy_score(y_train, y_train_pred)
    training_precision = precision_score(y_train, y_train_pred)
    y_train_prob = model.predict_proba(X_train)
    training_log_loss = log_loss(y_train, y_train_prob)
    training_recall = recall_score(y_train, y_train_pred)

    # Printing training results
    print("Training Accuracy:", training_accuracy)
    print("Training Precision:", training_precision)
    print("Training Log Loss:", training_log_loss)
    print("Training Recall:", training_recall)

    # Testing metrics
    y_pred = model.predict(X_test)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    logloss = log_loss(y_test, y_pred)

    # Printing testing metrics
    print("Precision: {0:.4f}".format(precision))
    print("Recall: {0:.4f}".format(recall))
    print("F1: {0:.4f}".format(f1))
    print("Log Loss: {0:.4f}".format(logloss))

    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds")

# Execute the function
train_and_evaluate(data_frame)

