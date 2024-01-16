#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
from scipy import stats
def calculate_tests_and_ci(df1, df2):
   
    # Create an empty DataFrame to store the results
    result_df = pd.DataFrame(index=df1.columns)

    # Calculate the Mann-Whitney U test, t-test, and confidence intervals for each column
    for column in df1.columns:
        # Calculate the Mann-Whitney U test
        u_stat, mw_p_value = stats.mannwhitneyu(df1[column], df2[column], alternative='two-sided')

        # Calculate the t-test
        t_stat, t_p_value = stats.ttest_ind(df1[column], df2[column])

        # Calculate the means and standard deviations for both DataFrames
        mean1 = df1[column].mean()
        mean2 = df2[column].mean()
        std1 = df1[column].std()
        std2 = df2[column].std()

        # Calculate the pooled standard error
        pooled_std = np.sqrt((std1**2 / len(df1)) + (std2**2 / len(df2)))

        # Calculate the degrees of freedom
        df = len(df1) + len(df2) - 2

        # Calculate the critical value for a 95% confidence interval (two-tailed)
        alpha = 0.05
        critical_value = stats.t.ppf(1 - alpha / 2, df=df)

        # Calculate the margin of error
        margin_of_error = critical_value * pooled_std

        # Calculate the confidence interval
        lower_bound = (mean1 - mean2) - margin_of_error
        upper_bound = (mean1 - mean2) + margin_of_error

        # Store the results in the DataFrame
        result_df.loc[column, "Mann-Whitney U Test Statistic"] = u_stat
        result_df.loc[column, "Mann-Whitney U Test p-value"] = mw_p_value
        result_df.loc[column, "t-Test Statistic"] = t_stat
        result_df.loc[column, "t-Test p-value"] = t_p_value
        result_df.loc[column, "95% CI Lower Bound"] = lower_bound
        result_df.loc[column, "95% CI Upper Bound"] = upper_bound

    return result_df

data_frame = pd.read_csv('/home/jamalids/Downloads/OC_Blood_Routine.csv')
data_frame = data_frame.iloc[: , :-1]
#data_frame.head
#bot
df_bot=data_frame[data_frame["TYPE"]==0]
n = len(df_bot.columns)
df_bot = df_bot.iloc[:, 1:n-1]
#OC
df_OC=data_frame[data_frame["TYPE"]==1]
n = len(df_OC.columns)
df_OC = df_OC.iloc[:, 1:n-1]

results = calculate_tests_and_ci(df_bot, df_OC)


# In[7]:


results .head(100)


# # logistic regression

# In[10]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


# In[38]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.metrics import confusion_matrix
data_frame = pd.read_csv('/home/jamalids/Downloads/OC_Blood_Routine.csv')
data_frame = data_frame.iloc[: , :-1]
data_frame.head



# In[43]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Load your dataset and split it into features (X) and target variable (y)
predicted_class_name = ['TYPE']

X = data_frame[feature_column_names].values
y = data_frame[predicted_class_name].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Create a Logistic Regression model
model = LogisticRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
y_train_pred = model.predict(X_train)
training_accuracy = accuracy_score(y_train, y_train_pred)

# Calculate training precision
training_precision = precision_score(y_train, y_train_pred)

# Calculate training log loss (requires probability estimates)
y_train_prob = model.predict_proba(X_train)
training_log_loss = log_loss(y_train, y_train_prob)

# Calculate training recall
training_recall = recall_score(y_train, y_train_pred)

# Print the results
print("Training Accuracy:", training_accuracy)
print("Training Precision:", training_precision)
print("Training Log Loss:", training_log_loss)
print("Training Recall:", training_recall)



precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)
logloss = log_loss(y_test, y_pred)


print ("Precision: {0:.4f}".format(precision))
print ("Recall: {0:.4f}".format(recall))
print ("F1: {0:.4f}".format(f1))
print ("AUC: {0:.4f}".format(auc))
print ("Log Loss: {0:.4f}".format(logloss))


# # SVM

# In[51]:


from sklearn import svm

clf = svm.SVC(kernel='linear',probability=True)

# Train the SVM classifier
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)
# Probabilities for log loss
y_prob = clf.predict_proba(X_test)[:, 1]  
# Calculate test accuracy
accuracy = accuracy_score(y_test, y_pred)

#Calculate precision
precision = precision_score(y_test, y_pred)

# Calculate recall
recall = recall_score(y_test, y_pred)

# Calculate F1-score
f1 = f1_score(y_test, y_pred)

# Calculate AUC (ROC AUC)
auc = roc_auc_score(y_test, y_prob)

# Calculate Log Loss
logloss = log_loss(y_test, y_prob)

# Print the results
print("Test Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
print("AUC (ROC AUC):", auc)
print("Log Loss:", logloss)


# # Decision Trees 

# In[54]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss


# Create and train a Decision Tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]  # Probabilities for log loss

# Calculate Test Accuracy
test_accuracy = accuracy_score(y_test, y_pred)

# Calculate Precision
precision = precision_score(y_test, y_pred)

# Calculate Recall
recall = recall_score(y_test, y_pred)

# Calculate F1-Score
f1 = f1_score(y_test, y_pred)

# Calculate AUC (ROC AUC)
roc_auc = roc_auc_score(y_test, y_prob)

# Calculate Log Loss
logloss = log_loss(y_test, y_prob)
# Print the results
print("Test Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
print("AUC (ROC AUC):", auc)
print("Log Loss:", logloss)


# # LightGBM 

# In[58]:



import lightgbm as lgb
params = {
    'application': 'binary', 
    'boosting': 'gbdt', 
    'num_iterations': 100, 
    'learning_rate': 0.05,
    'num_leaves': 62,
    'device': 'cpu', 
    'max_depth': -1, 
    'max_bin': 510, 
    'lambda_l1': 5, 
    'lambda_l2': 10, 
    'metric' : 'binary_error',
    'subsample_for_bin': 200, 
    'subsample': 1,
    'colsample_bytree': 0.8,
    'min_split_gain': 0.5, 
    'min_child_weight': 1, 
    'min_child_samples': 5
}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a LightGBM classifier with the specified parameters
clf = lgb.LGBMClassifier(**params)

# Train the LightGBM classifier
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]  # Probabilities for log loss

# Calculate Test Accuracy
test_accuracy = accuracy_score(y_test, y_pred)

# Calculate Precision
precision = precision_score(y_test, y_pred)

# Calculate Recall
recall = recall_score(y_test, y_pred)

# Calculate F1-Score
f1 = f1_score(y_test, y_pred)

# Calculate AUC (ROC AUC)
roc_auc = roc_auc_score(y_test, y_prob)

# Calculate Log Loss
logloss = log_loss(y_test, y_prob)

# Print the results
print("Test Accuracy:", test_accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
print("AUC (ROC AUC):", roc_auc)
print("Log Loss:", logloss)


# In[ ]:




