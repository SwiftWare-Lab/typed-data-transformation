
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
import warnings
from sklearn.exceptions import FitFailedWarning
import re
from scipy.stats import uniform
from scipy.stats import expon
from sklearn.model_selection import RandomizedSearchCV



# Function to train SVM and measure memory usage
def train_svm(X_train, y_train, param_distributions, n_iter=10):
    print("Training SVM model, please wait...")
    start_time = time.time()
    clf = SVC(kernel='linear')
    random_search = RandomizedSearchCV(estimator=clf, param_distributions=param_distributions, 
                                       n_iter=n_iter, scoring='accuracy', cv=5, n_jobs=-1)
    random_search.fit(X_train, y_train)
    clf = random_search.best_estimator_
    clf.fit(X_train, y_train)
    end_time = time.time()
    print("SVM model training completed.")
    return clf, end_time - start_time




# Function to train Logistic Regression and measure memory usage
def train_logistic_regression(X_train, y_train, param_distributions, n_iter=10):
    print("Training Logistic Regression model...")
    start_time = time.time()
    clf = LogisticRegression(solver='saga', max_iter=1000)
    random_search = RandomizedSearchCV(clf, param_distributions, n_iter=n_iter, scoring='accuracy', cv=5, n_jobs=-1, random_state=42)
    random_search.fit(X_train, y_train)
    clf = random_search.best_estimator_
    clf.fit(X_train, y_train)
    end_time = time.time()
    print("Logistic Regression training completed.")
    return clf, end_time - start_time

# Additional functions for new classifiers
def train_decision_tree(X_train, y_train, param_distributions, n_iter=10):
    print("Training Decision Tree model...")
    start_time = time.time()
    clf = DecisionTreeClassifier()
    random_search = RandomizedSearchCV(clf, param_distributions, n_iter=n_iter, scoring='accuracy', cv=5, n_jobs=-1, random_state=42)
    random_search.fit(X_train, y_train)
    clf = random_search.best_estimator_
    clf.fit(X_train, y_train)
    end_time = time.time()
    print("Decision Tree training completed.")
    return clf, end_time - start_time


def train_xgboost(X_train, y_train, parameters):
    print("Training XGBoost  model, please wait...")
    start_time = time.time()
    clf = XGBClassifier(**parameters)
    grid_search = GridSearchCV(estimator=clf, param_grid=parameters, cv=5, scoring='accuracy', n_jobs=1)
    grid_search.fit(X_train, y_train)
    clf = grid_search.best_estimator_
    clf.fit(X_train, y_train)
    end_time = time.time()
    print("logistic XGBoost model training completed.")
    return clf, end_time - start_time


def train_random_forest(X_train, y_train, param_distributions, n_iter=10):
    print("Training Random Forest model...")
    start_time = time.time()
    clf = RandomForestClassifier()
    random_search = RandomizedSearchCV(clf, param_distributions, n_iter=n_iter, scoring='accuracy', cv=5, n_jobs=-1, random_state=42)
    random_search.fit(X_train, y_train)
    clf = random_search.best_estimator_
    clf.fit(X_train, y_train)
    end_time = time.time()
    print("Random Forest training completed.")
    return clf, end_time - start_time


def train_lightgbm(X_train, y_train, params):
    print("Training LightGBM  model, please wait...")
    # Suppress specific LightGBM warnings
    warnings.filterwarnings('ignore', message='boosting is set=gbdt, boosting_type=gbdt will be ignored. Current value: boosting=gbdt')
    warnings.filterwarnings('ignore', message='lambda_l1 is set=5, reg_alpha=0.0 will be ignored. Current value: lambda_l1=5')
    warnings.filterwarnings('ignore', message='lambda_l2 is set=10, reg_lambda=0.0 will be ignored. Current value: lambda_l2=10')
    warnings.filterwarnings('ignore', message='No further splits with positive gain, best gain: -inf')
    start_time = time.time()
    clf = LGBMClassifier(**params)
    clf.fit(X_train, y_train)
    end_time = time.time()
    print("logistic LightGBM model training completed.")
    return clf, end_time - start_time


# Function to evaluate a classifier and return metrics
def evaluate_classifier(classifier, X_train, y_train, X_test, y_test):
    train_accuracy = accuracy_score(y_train, classifier.predict(X_train))
    test_accuracy = accuracy_score(y_test, classifier.predict(X_test))
    precision = precision_score(y_test, classifier.predict(X_test))
    recall = recall_score(y_test, classifier.predict(X_test))
    return train_accuracy, test_accuracy, precision, recall


# Load data
csv_file_path = input("Please enter the path to your CSV file: ")
data_real = pd.read_csv(csv_file_path)
datasets = [{"name": "Real Data", "data": data_real}]

param_distributions_dt = {
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': uniform(0.1, 1.0),
    'min_samples_leaf': uniform(0.1, 0.5)
}
param_distributions_rf = {
    'n_estimators': range(100, 1001, 100),
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': uniform(0.1, 1.0),
    'min_samples_leaf': uniform(0.1, 0.5)
}
parameters_xgb = {'objective': ['binary:logistic'],'nthread': [4],'seed': [42],'max_depth': range(2, 10, 1),'n_estimators': range(60, 220, 40),
    'learning_rate': [0.1, 0.01, 0.05]}
params_lgbm = {'application': 'binary', 'boosting': 'gbdt', 'num_iterations': 100, 'learning_rate': 0.05, 'num_leaves': 62, 'device': 'cpu', 'max_depth': -1, 'max_bin': 510, 'lambda_l1': 5, 'lambda_l2': 10, 'metric': 'binary_error', 'subsample_for_bin': 200, 'subsample': 1, 'colsample_bytree': 0.8, 'min_split_gain': 0.5, 'min_child_weight': 1, 'min_child_samples': 5}

param_distributions_svm = {
    'C': expon(scale=100), 
    'gamma': expon(scale=.1), 
    'kernel': ['linear']
}

# Initialize results dictionary
results = {
    "Dataset": [],
     "Model": [],
    "Accuracy": [],
    "Precision": [],
    "Recall": [],
    "Training Time (seconds)": [],
    "Memory Usage (MiB)": []
}

# Hyperparameters for GridSearchCV
# Use a smaller number of iterations for RandomizedSearchCV
n_iter = 10
param_distributions_svm = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}
param_distributions_lr = {
    'C': uniform(loc=0, scale=4),
    'penalty': ['l2', 'none']
}

warnings.filterwarnings('ignore', category=FitFailedWarning)
warnings.filterwarnings('ignore', message="No further splits with positive gain")
warnings.filterwarnings('ignore', module='lightgbm')
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
        #("SVM", train_svm, param_distributions_svm),
        ("Logistic Regression", train_logistic_regression, param_distributions_lr),
        ("Decision Tree", train_decision_tree, param_distributions_dt),
        ("XGBoost", train_xgboost, parameters_xgb),
        ("Random Forest", train_random_forest, param_distributions_rf),
        ("LightGBM", train_lightgbm, params_lgbm)
    ]

    for model_name, train_func, params in models:
        
        memory_used, trained_model = memory_usage((train_func, (X_train, y_train, params)), max_usage=True, retval=True)
        training_time = trained_model[1]
        best_estimator = trained_model[0]
        train_accuracy, test_accuracy, precision, recall = evaluate_classifier(best_estimator, X_train, y_train, X_test, y_test)

        # Store results
        results["Dataset"].append(dataset_name)
        results["Model"].append(model_name)
        results["Accuracy"].append(test_accuracy)
        results["Precision"].append(precision)
        results["Recall"].append(recall)
        results["Training Time (seconds)"].append(training_time)
        results["Memory Usage (MiB)"].append(memory_used)

results_df = pd.DataFrame(results)

# Print and save results
print(results_df)
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
csv_save_path = input("Please enter the path where you want to save the results CSV file (including filename and extension): ")
results_df.to_csv(csv_save_path, index=False)
save_plot_path = input("Please enter the path where you want to save the plot (including filename and extension): ")
plt.savefig(save_plot_path )
plt.show()


# In[ ]:




