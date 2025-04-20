import pandas as pd
import matplotlib.pyplot as plt

# Load CSV file (update path as needed)
csv_file = "/home/jamalids/Documents/combine-com-through/joined_compression_stats1.csv"   # or use the full path if necessary
df = pd.read_csv(csv_file)

# (Optional) Print the columns to check
print("Columns in CSV:", df.columns.tolist())

# Rename the columns so that the legend displays the desired names.
# Here we assume:
#   - "raw_zstd_ratio" will be plotted as "TDT_zstd"
#   - "bitshuffle_zstd_ratio" as "bitshuffle_zstd"
#   - "raw_zstd_byteshuffle_ratio" as "byteshuffle_zstd"
df = df.rename(columns={

    "bitshuffle_zstd_ratio": "bitshuffle_zstd",
    "blosc_zstd_byteshuffle_ratio": "byteshuffle_zstd",
    "raw_zstd_ratio_y":"standard_zstd"

})

# Optionally, sort by dataset name to maintain a consistent ordering on the x-axis.
df = df.sort_values("DatasetName")

# Create the plot.
plt.figure(figsize=(12, 6))
plt.plot(df["DatasetName"], df["TDT_zstd"], marker='o', label="TDT_zstd")
plt.plot(df["DatasetName"], df["bitshuffle_zstd"], marker='s', label="bitshuffle_zstd")
#plt.plot(df["DatasetName"], df["byteshuffle_zstd"], marker='^', label="byteshuffle_zstd")
# --- Added lines for raw_zstd_ratio and standard_zstd ---
#plt.plot(df["DatasetName"], df["raw_zstd_ratio"], marker='D', label="raw_zstd_ratio")
plt.plot(df["DatasetName"], df["standard_zstd"], marker='x', label="standard_zstd")
plt.yscale("log")
plt.xlabel("DatasetName")
plt.ylabel("log(Compression Ratio)")
plt.title("Compression Ratio by Dataset for ZSTD Variants")
plt.xticks(rotation=45, ha="right")
plt.yscale("log")  # if you also want a logarithmic y-axis, otherwise remove this line
plt.legend()
plt.tight_layout()
plt.savefig("/home/jamalids/Documents/transformation-plot3.png")

