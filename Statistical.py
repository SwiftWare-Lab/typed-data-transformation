
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
results.head(40)
print(results.head(30))




