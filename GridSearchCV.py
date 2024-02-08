
import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score, roc_auc_score, log_loss,confusion_matrix
from imblearn.over_sampling import SMOTE
from memory_profiler import memory_usage
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
import warnings
from sklearn.exceptions import FitFailedWarning
import re
from scipy.stats import uniform,expon
import seaborn as sns
import ast
import numpy as np
import os 
import sys
from sklearn.svm import SVC

# Function to train SVM
def train_svm(X_train, y_train, param_grid):
    print("Training SVM  model, please wait...")
    start_time = time.time()
    clf = SVC(kernel='linear')
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=1)
    grid_search.fit(X_train, y_train)
    clf = grid_search.best_estimator_
    clf.fit(X_train, y_train)
    end_time = time.time()
    print("SVM model training completed.")
    return clf, end_time - start_time


# Function to train Logistic Regression 
def train_logistic_regression(X_train, y_train, grid_values):
    print("Training logistic Regression  model, please wait...")
    start_time = time.time()
    clf = LogisticRegression(random_state=42)
    grid_search = GridSearchCV(estimator=clf, param_grid=grid_values, scoring='accuracy', cv=5, n_jobs=1)
    grid_search.fit(X_train, y_train)
    clf = grid_search.best_estimator_
    clf.fit(X_train, y_train)
    end_time = time.time()
    print("logistic Regression model training completed.")
    return clf, end_time - start_time


# Additional functions for other classifiers
def train_decision_tree(X_train, y_train, param_dict):
    print("Training Decision Tree  model, please wait...")
    start_time = time.time()
    clf = DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=clf, param_grid=param_dict, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    clf = grid_search.best_estimator_
    clf.fit(X_train, y_train)
    end_time = time.time()
    print("logistic Decision Tree model training completed.")
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


def train_random_forest(X_train, y_train, param_grid):
    print("Training Random Forest  model, please wait...")
    start_time = time.time()
    clf = RandomForestClassifier()
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=1)
    grid_search.fit(X_train, y_train)
    clf = grid_search.best_estimator_
    clf.fit(X_train, y_train)
    end_time = time.time()
    print("logistic Random Forest model training completed.")
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
    # Predictions
    train_preds = classifier.predict(X_train)
    test_preds = classifier.predict(X_test)
    test_probs = None  # Initialize probability estimates
    # Check if classifier supports probability estimates directly
    if hasattr(classifier, "predict_proba"):
        test_probs = classifier.predict_proba(X_test)[:, 1]
    elif hasattr(classifier, "decision_function"):  
        test_probs = classifier.decision_function(X_test)
    if test_probs is not None:
        auc = roc_auc_score(y_test, test_probs)
        logloss = log_loss(y_test, test_probs)
    else:
        auc = None
        logloss = None

    # Calculating other metrics
    train_accuracy = accuracy_score(y_train, train_preds)
    test_accuracy = accuracy_score(y_test, test_preds)
    precision = precision_score(y_test, test_preds, zero_division=0)
    recall = recall_score(y_test, test_preds, zero_division=0)
    f1 = f1_score(y_test, test_preds, zero_division=0)
    feature_importances = None
    if hasattr(classifier, 'feature_importances_'):
        feature_importances = classifier.feature_importances_
    elif isinstance(classifier, (LogisticRegression)) :
        feature_importances = classifier.coef_[0]
    elif isinstance(classifier, SVC) and classifier.kernel == 'linear':
        # Check for SVC with a linear kernel
        feature_importances = classifier.coef_[0]    
    conf_matrix = confusion_matrix(y_test, test_preds)
    
    return train_accuracy, test_accuracy, precision, recall, f1, auc, logloss, conf_matrix, feature_importances


# Function to parse the confusion matrix from the string format in the CSV
def parse_conf_matrix(conf_matrix):
    if isinstance(conf_matrix, str):
        # If conf_matrix is a string, parse it
        conf_matrix_str = conf_matrix.replace('[', '').replace(']', '').split('\n')
        conf_matrix = [list(map(int, row.strip().split())) for row in conf_matrix_str]
        return np.array(conf_matrix)
    elif isinstance(conf_matrix, np.ndarray):
        # If conf_matrix is already a numpy array, return it directly
        return conf_matrix
    else:
        # Handle other types as needed or raise an error
        raise ValueError("Unexpected data type for confusion matrix")


# Function to plot confusion matrix
def plot_conf_matrix(conf_matrix, model_name, ax):
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(f'Confusion Matrix for {model_name}')
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.xaxis.set_ticklabels(['False', 'True'])
    ax.yaxis.set_ticklabels(['False', 'True'])
  
   
# Function to plot feature importance
def plot_feature_importance(feature_importances, feature_names, model_name, ax=None):
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
    plt.title(f'Feature Importance for {model_name}')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.show()


def plot_decision_tree(decision_tree, feature_names, model_name, ax=None, save_path=None):
    # Import necessary libraries for plotting decision tree
    from sklearn.tree import plot_tree
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    plot_tree(decision_tree, filled=True, feature_names=feature_names, class_names=['0', '1'], ax=ax)
    ax.set_title(f'Decision Tree for {model_name}')
    if save_path:
        fig.savefig(save_path)  
        plt.close(fig)  # Close the figure to release resources
    elif ax is None:
        plt.show()


def plot_xgboost_feature_importance(feature_importances, feature_names, model_name, ax=None):
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
    plt.title(f'Feature Importance for {model_name}')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    #plt.show()
    if ax is None:
        plt.show()
   
   
    
def plot_feature_importance(feature_importances, feature_names, model_name, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
    ax.set_title(f'Feature Importance for {model_name}')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    # handling for SVM and Logistic Regression
    if 'SVM' in model_name:
        plt.bar(range(len(feature_importances)), np.abs(feature_importances))
        plt.title(f'Feature Importance for {model_name}')
        plt.xlabel('Feature')
        plt.ylabel('Importance')
    if ax is None:
        plt.show()
    
# Hyperparameters for GridSearchCV
param_distributions_dt = {"criterion": ['gini', 'entropy'], "max_depth": [150, 155, 160], "min_samples_split": range(1, 10), "min_samples_leaf": range(1, 5)}

param_distributions_rf= {'n_estimators': [200, 500],
                          'max_features': ['auto', 'sqrt', 'log2'],
                          'max_depth': [4, 5, 6, 7, 8], 
                          'criterion': ['gini', 'entropy']}
parameters_xgb = {'objective': ['binary:logistic'],'nthread': [4],'seed': [42],'max_depth': range(2, 10, 1),'n_estimators': range(60, 220, 40),
    'learning_rate': [0.1, 0.01, 0.05]}

params_lgbm = {'application': 'binary', 'boosting': 'gbdt', 'num_iterations': 100, 'learning_rate': 0.05, 'num_leaves': 62, 'device': 'cpu', 'max_depth': -1, 'max_bin': 510, 'lambda_l1': 5, 'lambda_l2': 10, 'metric': 'binary_error', 'subsample_for_bin': 200, 'subsample': 1, 'colsample_bytree': 0.8, 'min_split_gain': 0.5, 'min_child_weight': 1, 'min_child_samples': 5}

param_distributions_svm = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['linear']}  

param_distributions_lr = {'penalty': ['l1','l2'], 'C': [0.001,0.01,0.1,1,10,100,1000]}

#Running command-line
if len(sys.argv) != 2:
    print("Usage: python GridSearchCV.py <dataset_name> ")
    sys.exit(1)

# Load the dataset from the command-line argument
dataset_path = sys.argv[1]
data_real= pd.read_csv(dataset_path)  
datasets = [{"name": "Real Data", "data": data_real}]

# Initialize results dictionary
results = {
    "Dataset": [],
     "Model": [],
    "Accuracy": [],
    "Precision": [],
    "Recall": [],
    "F1-Score":[],
    "AUC":[],
    "LogLoss":[],
    "Training Time (seconds)": [],
    "Memory Usage (MiB)": [],
    "Conf_Matrix":[]
  
}
n_iter = 10
warnings.filterwarnings('ignore', category=FitFailedWarning)
warnings.filterwarnings('ignore', message="No further splits with positive gain")
warnings.filterwarnings('ignore', module='lightgbm')
figures = []
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
    print(feature_column_names)
    X = data_frame[feature_column_names].values
    y = data_frame['TYPE'].values
    #SMOTE Function
    sm = SMOTE(random_state=2)
    X, y = sm.fit_resample(X, y)
    # Divide the data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_validation, X_test, y_validation, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_validation = sc.transform(X_validation)
    X_test = sc.transform(X_test)
    # Train and evaluate models
    models = [
        ("SVM", train_svm, param_distributions_svm),
        ("Logistic Regression", train_logistic_regression, param_distributions_lr),
        ("Decision Tree", train_decision_tree, param_distributions_dt),
        ("XGBoost", train_xgboost, parameters_xgb),
        ("Random Forest", train_random_forest, param_distributions_rf),
        ("LightGBM", train_lightgbm, params_lgbm)
    ]

if not os.path.exists('./feature_importance_plots'):
               os.makedirs('./feature_importance_plots')
for model_name, train_func, params in models:
    memory_used, trained_model = memory_usage((train_func, (X_train, y_train, params)), max_usage=True, retval=True)
    training_time = trained_model[1]
    best_estimator = trained_model[0]
    train_accuracy, test_accuracy, precision, recall, f1, auc, logloss, conf_matrix, feature_importances = evaluate_classifier(best_estimator, X_train, y_train, X_test, y_test)
    print(model_name)
    print(feature_importances)
    # Store results
    results["Dataset"].append(dataset_name)
    results["Model"].append(model_name)
    results["Accuracy"].append(test_accuracy)
    results["Precision"].append(precision)
    results["Recall"].append(recall)
    results["F1-Score"].append(f1)
    results["AUC"].append(auc)
    results["LogLoss"].append(logloss)
    results["Training Time (seconds)"].append(training_time)
    results["Memory Usage (MiB)"].append(memory_used)
    results["Conf_Matrix"].append(conf_matrix)
    
    
    if feature_importances is not None:
        # Define the figure for plotting
        fig, ax = plt.subplots(figsize=(10, 6))
        # Call the appropriate plotting function based on the model
        if isinstance(best_estimator, (RandomForestClassifier, DecisionTreeClassifier, XGBClassifier, LGBMClassifier)):
            plot_feature_importance(feature_importances, feature_column_names, model_name, ax)
        elif isinstance(best_estimator, (SVC)) and best_estimator.kernel == 'linear':
            plot_feature_importance(np.abs(feature_importances), feature_column_names, model_name, ax)
        elif isinstance(best_estimator, LogisticRegression):
            plot_feature_importance(np.abs(feature_importances), feature_column_names, model_name, ax)
        # Save the plot to the specified directory
        out_plot_feature = os.path.join('feature_importance_plots', f'{model_name}_feature_importance1.png')
        # Save the feature importance plot
        fig.savefig(out_plot_feature)
        
    if feature_importances is not None:
        
         if isinstance(best_estimator, DecisionTreeClassifier):
            # Create a new figure and axis for each plot
            fig, ax = plt.subplots(figsize=(12, 8))
            plot_decision_tree(best_estimator, feature_column_names, model_name, ax)
            # Save the plot as an image file
            out_plot_feature = os.path.join('plot', f'{model_name}_feature_importance.png')
            # Save the feature importance plot
            fig.savefig(out_plot_feature)
            # release resources
            plt.close(fig)  
        
# Adjust layout
plt.tight_layout()
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
plt.tight_layout()
if not os.path.exists('./output'):
    os.makedirs('./output')

# Specify the filename within the 'Output' directory
filename = "results.csv"
csv_save_path = os.path.join('./output', filename)
# Save the DataFrame to the specified path
results_df.to_csv(csv_save_path, index=False)
conf_matrices_corrected = results_df['Conf_Matrix'].apply(parse_conf_matrix)
print(conf_matrices_corrected)
# Plotting with the corrected confusion matrices
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Confusion Matrices for Different Models', fontsize=16)
axes = axes.flatten()
for i, (model_name, conf_matrix) in enumerate(zip(results_df['Model'], conf_matrices_corrected)):
    if i < len(axes):  # Check to avoid index out of range error
        plot_conf_matrix(conf_matrix, model_name, axes[i])
      
# Hide any unused subplots
for ax in axes[len(results_df['Model']):]:
    ax.axis('off')
# Adjust layout
plt.tight_layout()
plt.subplots_adjust(top=0.9)
out_plot = os.path.join('plot', 'confusion_matrix_plot.png')
plt.savefig(out_plot)
plt.show()





