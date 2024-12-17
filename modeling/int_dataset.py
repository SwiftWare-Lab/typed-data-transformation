
import pandas as pd

import kagglehub

# Download latest version
# path = kagglehub.dataset_download("yashkmd/credit-profile-two-wheeler-loan-dataset")
#
# print("Path to dataset files:", path)

# Download latest version
# path = kagglehub.dataset_download("simiotic/ethereum-nfts")
#
# print("Path to dataset files:", path)
import sqlite3

# Connect to the database
conn = sqlite3.connect('data/nfts.sqlite')

# reading one column from the database
df = pd.read_sql_query("SELECT * FROM nfts", conn)


print("jjj")
# read csv file
# df = pd.read_csv("data/BankChurners.csv")
# # print first 5 rows
# print(df.head())