import pandas as pd

# Load the CSV file
df = pd.read_csv("/home/jamalids/Documents/compression-part4/new1/num_brain_f64.csv")

# Ensure the column names are correct
print(df.columns)

# Group by 'base type' and calculate median for each numeric column
# Group by 'base type' and calculate the median for 'compressed_time' and 'decompressed_time'
result = df.groupby('Type')[['TotalTimeCompressed', 'TotalTimeDecompressed']].median()


# Save the result to a new CSV file
result.to_csv("/home/jamalids/Documents/compression-part4/new1/msg_bt_f64.csv")

print(result)
