import pandas as pd
import matplotlib.pyplot as plt

# Define file paths (adjust these paths if needed)
row_file = r"C:\Users\jamalids\Downloads\row-col-order\row-col-order\fastlz\max_fastlz-row.csv"
col_file = r"C:\Users\jamalids\Downloads\row-col-order\row-col-order\fastlz\max_fastlz-col.csv"

# Load CSV files.
df_row = pd.read_csv(row_file)
df_col = pd.read_csv(col_file)

# Remove extra whitespace from column names
df_row.columns = df_row.columns.str.strip()
df_col.columns = df_col.columns.str.strip()

# Ensure that the key column is named 'DatasetName'
if 'DatasetName' not in df_row.columns:
    df_row.rename(columns={df_row.columns[0]: 'DatasetName'}, inplace=True)
if 'DatasetName' not in df_col.columns:
    df_col.rename(columns={df_col.columns[0]: 'DatasetName'}, inplace=True)

# Filter rows where RunType is equal to 'Decompose_Block_Parallel'
df_row = df_row[df_row['RunType'] == 'Decompose_Block_Parallel']
df_col = df_col[df_col['RunType'] == 'Decompose_Block_Parallel']

# Rename the metric columns for clarity.
# Here we assume that the metric to compare is "CompressionRatio".
if 'CompressionRatio' in df_row.columns:
    df_row.rename(columns={'CompressionRatio': 'value_row'}, inplace=True)
if 'CompressionRatio' in df_col.columns:
    df_col.rename(columns={'CompressionRatio': 'value_col'}, inplace=True)

# Merge the DataFrames on 'DatasetName'
try:
    df = pd.merge(df_row, df_col, on='DatasetName', suffixes=('_row', '_col'))
except KeyError as e:
    print(f"Merge failed: {e}")
    exit(1)

# ------------------------------
# Check whether ConfigString_row matches ConfigString_col
# ------------------------------
# Strip whitespace from config strings
df['ConfigString_row'] = df['ConfigString_row'].astype(str).str.strip()
df['ConfigString_col'] = df['ConfigString_col'].astype(str).str.strip()

# Identify rows where the ConfigString values differ
mismatch_mask = df['ConfigString_row'] != df['ConfigString_col']
df_mismatch = df[mismatch_mask]

# Print mismatches (if any)
if not df_mismatch.empty:
    print("Datasets with differing ConfigString:")
    for idx, row_data in df_mismatch.iterrows():
        print(f"  DatasetName: {row_data['DatasetName']}")
        print(f"    ConfigString_row: {row_data['ConfigString_row']}")
        print(f"    ConfigString_col: {row_data['ConfigString_col']}\n")

# ------------------------------
# Plot 1: Matching ConfigString
# ------------------------------
df_match = df[~mismatch_mask]

plt.figure(figsize=(10, 6))
plt.plot(df_match['DatasetName'], df_match['value_row'], marker='o', label='TDT-Row')
plt.plot(df_match['DatasetName'], df_match['value_col'], marker='o', label='TDT-Col')
plt.xlabel('Dataset Name', ha='left', x=0.0)
plt.ylabel('CompressionRatio')
# Using symlog to better reveal slight differences (adjust linthresh as needed)
#plt.yscale('symlog', linthresh=1e-3)
#plt.ylim(0.1, 5)
plt.title('Comparison of TDT-Row vs TDT-Col Metrics (Matching ConfigString)')
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('fastlz.png')
plt.show()

# ------------------------------
# Plot 2: Mismatched ConfigString (if any)
# ------------------------------
if not df_mismatch.empty:
    # Create a composite label including both config strings for clarity
    df_mismatch['Label'] = df_mismatch.apply(
        lambda x: f"{x['DatasetName']}\nrow: {x['ConfigString_row']}\ncol: {x['ConfigString_col']}",
        axis=1
    )
    plt.figure(figsize=(10, 6))
    plt.plot(df_mismatch['Label'], df_mismatch['value_row'], marker='o', label='TDT-Row')
    plt.plot(df_mismatch['Label'], df_mismatch['value_col'], marker='o', label='TDT-Col')
    plt.xlabel('Dataset Name / ConfigString (Mismatch)', ha='left', x=0.0)
    plt.ylabel('CompressionRatio')
    plt.yscale('symlog', linthresh=1e-3)
    plt.title('Comparison of TDT-Row vs TDT-Col Metrics (Mismatched ConfigString)')
    plt.legend()
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('fastlz_mismatch.png')
    plt.show()

# ------------------------------
# Additional Comparison Plots (for datasets with matching ConfigString)
# ------------------------------
# Work on a copy of df_match
#df_comp = df_match.copy()
df_comp=df_match
# Compute additional comparison metrics
df_comp['diff'] = df_comp['value_row'] - df_comp['value_col']
df_comp['ratio'] = df_comp['value_row'] / df_comp['value_col']

# Plot 3: Absolute Difference
plt.figure(figsize=(10, 6))
plt.plot(df_comp['DatasetName'], df_comp['diff'], marker='o', label='Difference (Row - Col)')
plt.xlabel('Dataset Name', ha='left', x=0.0)
plt.ylabel('Absolute Difference')
plt.title('Absolute Difference between TDT-Row and TDT-Col')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('fastlz_difference.png')
plt.show()

# Plot 4: Ratio of Row to Column
plt.figure(figsize=(10, 6))
plt.plot(df_comp['DatasetName'], df_comp['ratio'], marker='o', label='Ratio (Row/Col)')
plt.xlabel('Dataset Name', ha='left', x=0.0)
plt.ylabel('Ratio (comp ratio Row/ comp ratio Col)')
plt.title('Ratio of TDT-Row to TDT-Col')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('fastlz_ratio.png')
plt.show()

# Plot 5: Scatter Plot (Row vs. Col) with y=x line
plt.figure(figsize=(8, 8))
plt.scatter(df_comp['value_row'], df_comp['value_col'], label='Data Points')
max_val = max(df_comp['value_row'].max(), df_comp['value_col'].max())
min_val = min(df_comp['value_row'].min(), df_comp['value_col'].min())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y = x')
plt.xlabel('TDT-Row CompressionRatio')
plt.ylabel('TDT-Col CompressionRatio')
plt.title('Scatter Plot of TDT-Row vs TDT-Col')
plt.legend()
plt.tight_layout()
plt.savefig('fastlz_scatter.png')
plt.show()
