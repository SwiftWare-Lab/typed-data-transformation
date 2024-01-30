
import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from memory_profiler import memory_usage

# Function to train SVM and measure memory usage
def train_svm(X_train, y_train):
    start_time = time.time()
    clf = SVC(kernel='linear')
    clf.fit(X_train, y_train)
    end_time = time.time()
    return clf, end_time - start_time

# Function to train Logistic Regression and measure memory usage
def train_logistic_regression(X_train, y_train):
    start_time = time.time()
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    end_time = time.time()
    return clf, end_time - start_time

# Function to evaluate a classifier and return metrics
def evaluate_classifier(classifier, X_train, y_train, X_test, y_test):
    train_accuracy = accuracy_score(y_train, classifier.predict(X_train))
    test_accuracy = accuracy_score(y_test, classifier.predict(X_test))
    return train_accuracy, test_accuracy

# Define the dataset 
data_50k = pd.read_csv('/home/jamalids/Downloads/Synthetic_OC_Blood_Routine_50k.csv')
data_1000k = pd.read_csv('/home/jamalids/Downloads/Synthetic_OC_Blood_Routine_1000k.csv')
data_real = pd.read_csv('/home/jamalids/Downloads/OC_Blood_Routine (2).csv')


datasets = [
    {"name": "Real Data", "data": data_real},
    {"name": "Synthetic Data (50k)", "data": data_50k},
    {"name": "Synthetic Data (1000k)", "data": data_1000k}
]

# Initialize results dictionary
results = {
    "Dataset": [],
    "SVM Train Accuracy": [],
    "SVM Test Accuracy": [],
    "LR Train Accuracy": [],
    "LR Test Accuracy": [],
    "SVM Training Time (seconds)": [],
    "LR Training Time (seconds)": [],
    "SVM Memory Usage (MiB)": [],
    "LR Memory Usage (MiB)": [],
    "SVM Time Consumption (seconds)": [],  
    "LR Time Consumption (seconds)": []    
}

# Loop over datasets
for dataset_info in datasets:
    dataset_name = dataset_info["name"]
    data_frame = dataset_info["data"]
    
    # Preprocessing
    for i in range(len(data_frame)):
        if data_frame['TYPE'].iloc[i] != 0 and data_frame['TYPE'].iloc[i] != 1:
            data_frame.iloc[i] = 1 if data_frame['TYPE.1'].iloc[i] == 'OC' else 0
    
    feature_column_names = data_frame.columns[:-1]
    predicted_class_name = 'TYPE'
    data_frame = data_frame.iloc[:, :-1]

    feature_column_names = data_frame.columns[:-1]
    X = data_frame[feature_column_names].values
    y = data_frame['TYPE'].values

    sm = SMOTE(random_state=2)
    X, y = sm.fit_resample(X, y)

    # Divide the data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_validation, X_test, y_validation, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_validation = sc.transform(X_validation)
    X_test = sc.transform(X_test)

    # Train SVM and measure memory and time
    svm_memory_used, m_best_svm = memory_usage((train_svm, (X_train, y_train)), max_usage=True, retval=True)
    svm_training_time = m_best_svm[1]

    # Train Logistic Regression and measure memory and time
    lr_memory_used, m_best_lr = memory_usage((train_logistic_regression, (X_train, y_train)), max_usage=True, retval=True)
    lr_training_time = m_best_lr[1]

    # Model evaluation
    best_lr_estimator = m_best_lr[0]  
    y_pred_lr = best_lr_estimator.predict(X_test)
    best_svm_estimator = m_best_svm[0]
    y_pred_svm = best_svm_estimator.predict(X_test)

    # Measure time consumption
    svm_time_consumption = svm_training_time
    lr_time_consumption = lr_training_time

    # Evaluate classifiers
    svm_train_accuracy, svm_test_accuracy = evaluate_classifier(best_svm_estimator, X_train, y_train, X_test, y_test)
    lr_train_accuracy, lr_test_accuracy = evaluate_classifier(best_lr_estimator, X_train, y_train, X_test, y_test)
    

    # Store results in the dictionary
    results["Dataset"].append(dataset_name)
    results["SVM Train Accuracy"].append(svm_train_accuracy)
    results["SVM Test Accuracy"].append(svm_test_accuracy)
    results["LR Train Accuracy"].append(lr_train_accuracy)
    results["LR Test Accuracy"].append(lr_test_accuracy)
    results["SVM Training Time (seconds)"].append(svm_training_time)
    results["LR Training Time (seconds)"].append(lr_training_time)
    results["SVM Memory Usage (MiB)"].append(svm_memory_used)
    results["LR Memory Usage (MiB)"].append(lr_memory_used)
    results["SVM Time Consumption (seconds)"].append(svm_time_consumption)  
    results["LR Time Consumption (seconds)"].append(lr_time_consumption)   

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Print and save results
print(results_df)
results_df.to_csv('/home/jamalids/Downloads/data.csv', index=False)

# Plot results (line plots)
plt.figure(figsize=(16, 6))
plt.subplot(1, 3, 1)
plt.plot(results_df["Dataset"], results_df["SVM Test Accuracy"], label="SVM Test Accuracy")
plt.plot(results_df["Dataset"], results_df["LR Test Accuracy"], label="LR Test Accuracy")
plt.xlabel("Dataset")
plt.ylabel("Accuracy")
plt.title("Test Accuracy Comparison")
plt.xticks(rotation=45)
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(results_df["Dataset"], results_df["SVM Memory Usage (MiB)"], label="SVM Memory Usage (MiB)")
plt.plot(results_df["Dataset"], results_df["LR Memory Usage (MiB)"], label="LR Memory Usage (MiB)")
plt.xlabel("Dataset")
plt.ylabel("Memory Usage (MiB)")
plt.title("Memory Usage Comparison")
plt.xticks(rotation=45)
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(results_df["Dataset"], results_df["SVM Time Consumption (seconds)"], label="SVM Time Consumption (seconds)")
plt.plot(results_df["Dataset"], results_df["LR Time Consumption (seconds)"], label="LR Time Consumption (seconds)")
plt.xlabel("Dataset")
plt.ylabel("Time Consumption (seconds)")
plt.title("Time Consumption Comparison")
plt.xticks(rotation=45)
plt.legend()

plt.tight_layout()
plt.savefig('/home/jamalids/Downloads/plot.jpg')
plt.show()




