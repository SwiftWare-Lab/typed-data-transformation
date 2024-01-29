
import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE
from memory_profiler import memory_usage
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

# Function to train SVM and measure memory usage
def train_svm(X_train, y_train, param_grid):
    start_time = time.time()
    clf = SVC(kernel='linear')
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=1)
    grid_search.fit(X_train, y_train)
    clf = grid_search.best_estimator_
    clf.fit(X_train, y_train)
    end_time = time.time()
    return clf, end_time - start_time

# Function to train Logistic Regression and measure memory usage
def train_logistic_regression(X_train, y_train, grid_values):
    start_time = time.time()
    clf = LogisticRegression()
    grid_search = GridSearchCV(estimator=clf, param_grid=grid_values, scoring='accuracy', cv=5, n_jobs=1)
    grid_search.fit(X_train, y_train)
    clf = grid_search.best_estimator_
    clf.fit(X_train, y_train)
    end_time = time.time()
    return clf, end_time - start_time

# Additional functions for new classifiers
def train_decision_tree(X_train, y_train, param_dict):
    start_time = time.time()
    clf = DecisionTreeClassifier()
    grid_search = GridSearchCV(estimator=clf, param_grid=param_dict, cv=5, scoring='accuracy', n_jobs=1)
    grid_search.fit(X_train, y_train)
    clf = grid_search.best_estimator_
    clf.fit(X_train, y_train)
    end_time = time.time()
    return clf, end_time - start_time

def train_xgboost(X_train, y_train, parameters):
    start_time = time.time()
    clf = XGBClassifier(**parameters)
    grid_search = GridSearchCV(estimator=clf, param_grid=parameters, cv=5, scoring='accuracy', n_jobs=1)
    grid_search.fit(X_train, y_train)
    clf = grid_search.best_estimator_
    clf.fit(X_train, y_train)
    end_time = time.time()
    return clf, end_time - start_time

def train_random_forest(X_train, y_train, param_grid):
    start_time = time.time()
    clf = RandomForestClassifier()
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=1)
    grid_search.fit(X_train, y_train)
    clf = grid_search.best_estimator_
    clf.fit(X_train, y_train)
    end_time = time.time()
    return clf, end_time - start_time

def train_lightgbm(X_train, y_train, params):
    start_time = time.time()
    clf = LGBMClassifier(**params)
    clf.fit(X_train, y_train)
    end_time = time.time()
    return clf, end_time - start_time

# Function to evaluate a classifier and return metrics
def evaluate_classifier(classifier, X_train, y_train, X_test, y_test):
    train_accuracy = accuracy_score(y_train, classifier.predict(X_train))
    test_accuracy = accuracy_score(y_test, classifier.predict(X_test))
    precision = precision_score(y_test, classifier.predict(X_test))
    recall = recall_score(y_test, classifier.predict(X_test))
    return train_accuracy, test_accuracy, precision, recall

# Load data
data_real = pd.read_csv('/home/jamalids/Downloads/OC_Blood_Routine (2).csv')
datasets = [{"name": "Real Data", "data": data_real}]

# Initialize results dictionary
results = {
    "Dataset": [],
    "Method": [],
    "Model": [],
    "Accuracy": [],
    "Precision": [],
    "Recall": [],
    "Training Time (seconds)": [],
    "Memory Usage (MiB)": []
}

# Hyperparameters for GridSearchCV
param_grid_svm = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['linear']}
grid_values_lr = {'penalty': ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
param_dict_dt = {"criterion": ['gini', 'entropy'], "max_depth": [150, 155, 160], "min_samples_split": range(1, 10), "min_samples_leaf": range(1, 5)}
parameters_xgb = {'objective': ['binary:logistic'],'nthread': [4],'seed': [42],'max_depth': range(2, 10, 1),'n_estimators': range(60, 220, 40),
    'learning_rate': [0.1, 0.01, 0.05]}
param_grid_rf = {'n_estimators': [200, 500], 'max_features': ['auto', 'sqrt', 'log2'], 'max_depth': [4, 5, 6, 7, 8], 'criterion': ['gini', 'entropy']}
params_lgbm = {'application': 'binary', 'boosting': 'gbdt', 'num_iterations': 100, 'learning_rate': 0.05, 'num_leaves': 62, 'device': 'cpu', 'max_depth': -1, 'max_bin': 510, 'lambda_l1': 5, 'lambda_l2': 10, 'metric': 'binary_error', 'subsample_for_bin': 200, 'subsample': 1, 'colsample_bytree': 0.8, 'min_split_gain': 0.5, 'min_child_weight': 1, 'min_child_samples': 5}

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

    #sm = SMOTE(random_state=2)
    #X, y = sm.fit_resample(X, y)
    # Divide the data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_validation, X_test, y_validation, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_validation = sc.transform(X_validation)
    X_test = sc.transform(X_test)
    # Train and evaluate models
    models = [
        ("SVM", train_svm, param_grid_svm),
        ("Logistic Regression", train_logistic_regression, grid_values_lr),
        ("Decision Tree", train_decision_tree, param_dict_dt),
        ("XGBoost", train_xgboost, parameters_xgb),
        ("Random Forest", train_random_forest, param_grid_rf),
        ("LightGBM", train_lightgbm, params_lgbm)
    ]

    for model_name, train_func, params in models:
        memory_used, trained_model = memory_usage((train_func, (X_train, y_train, params)), max_usage=True, retval=True)
        training_time = trained_model[1]
        best_estimator = trained_model[0]
        train_accuracy, test_accuracy, precision, recall = evaluate_classifier(best_estimator, X_train, y_train, X_test, y_test)

        # Store results
        results["Dataset"].append(dataset_name)
        results["Method"].append("With Cross-Validation")
        results["Model"].append(model_name)
        results["Accuracy"].append(test_accuracy)
        results["Precision"].append(precision)
        results["Recall"].append(recall)
        results["Training Time (seconds)"].append(training_time)
        results["Memory Usage (MiB)"].append(memory_used)

results_df = pd.DataFrame(results)

# Print and save results
print(results_df)
results_df.to_csv('/home/jamalids/Downloads/result_Crossvalidation_all.csv', index=False)
# Group the results by Model and calculate the mean for each metric
grouped_results = results_df.groupby('Model').mean()
# Extract accuracy, training time, and memory usage
accuracy = grouped_results['Accuracy']
training_time = grouped_results['Training Time (seconds)']
memory_usage = grouped_results['Memory Usage (MiB)']
# Models for the x-axis
models = grouped_results.index
# Create line charts
plt.figure(figsize=(15, 10))
# Plot Accuracy
plt.subplot(3, 1, 1)
plt.plot(models, accuracy, marker='o', color='b', label='Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xticks(models)
plt.legend()
# Plot Training Time
plt.subplot(3, 1, 2)
plt.plot(models, training_time, marker='o', color='g', label='Training Time (seconds)')
plt.title('Model Training Time')
plt.ylabel('Time (seconds)')
plt.xticks(models)
plt.legend()
# Plot Memory Usage
plt.subplot(3, 1, 3)
plt.plot(models, memory_usage, marker='o', color='r', label='Memory Usage (MiB)')
plt.title('Model Memory Usage')
plt.ylabel('Memory (MiB)')
plt.xticks(models)
plt.legend()
# Adjust layout and show plot
plt.tight_layout()
plt.savefig('/home/jamalids/Downloads/plot_all.jpg')
plt.show()




