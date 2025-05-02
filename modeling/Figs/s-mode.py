import pandas as pd
from scipy.stats import gmean
from pathlib import Path

def combine_and_compute_gmean(directory_path, col_name, output_filename):
    directory = Path(directory_path)
    dataframes = []

    for file in directory.glob("*.csv"):
        try:
            df = pd.read_csv(file)
            df.fillna(0, inplace=True)

            # Standardize Huffman column names
            if "decomposed huffman_compress col-order ratio" in df.columns:
                df.rename(columns={'decomposed huffman_compress col-order ratio': 'decomposed huffman compression ratio'}, inplace=True)
            if "standard huffman_compress ratio" in df.columns:
                df.rename(columns={'standard huffman_compress ratio': 'standard huffman compression ratio'}, inplace=True)

            # Standardize XOR column names
            if "decomposed xor col-order ratio" in df.columns:
                df.rename(columns={'decomposed xor col-order ratio': 'decomposed xor compression ratio'}, inplace=True)
            if "standard xor ratio" in df.columns:
                df.rename(columns={'standard xor ratio': 'standard xor compression ratio'}, inplace=True)

            # Add fallback DatasetName
            if "DatasetName" not in df.columns and "dataset name" in df.columns:
                df.rename(columns={"dataset name": "DatasetName"}, inplace=True)
            if "DatasetName" not in df.columns:
                df["DatasetName"] = file.stem

            dataframes.append(df)
        except Exception as e:
            print(f"❌ Error reading {file}: {e}")

    if not dataframes:
        print(f"⚠️ No CSV files found in {directory_path}")
        return None, None

    # Combine and save the DataFrame
    combined_df = pd.concat(dataframes, ignore_index=True)
    output_path = f"/mnt/c/Users/jamalids/Downloads/figs/combined_{col_name}.csv"
    combined_df.to_csv(output_path, index=False)
    print(f"✅ Combined CSV saved to: {output_path}")

    # Extract relevant columns
    decomposed_col = f"decomposed {col_name} compression ratio"
    standard_col = f"standard {col_name} compression ratio"

    if decomposed_col not in combined_df.columns or standard_col not in combined_df.columns:
        print(f"⚠️ Missing expected columns in {output_filename}")
        return None, None

    gmean_decomposed = gmean(combined_df[decomposed_col][combined_df[decomposed_col] > 0])
    gmean_standard = gmean(combined_df[standard_col][combined_df[standard_col] > 0])
    return gmean_decomposed, gmean_standard


# === Run for Huffman ===
huff_path = "/mnt/c/Users/jamalids/Downloads/figs/results/huffman"
huff_decomp, huff_std = combine_and_compute_gmean(huff_path, "huffman", "combined_huffman.csv")

# === Run for XOR ===
xor_path = "/mnt/c/Users/jamalids/Downloads/figs/results/xor1"
xor_decomp, xor_std = combine_and_compute_gmean(xor_path, "xor", "combined_xor.csv")


# === Print geometric mean results ===
print("\n🔍 Geometric Mean Compression Ratios:")
print("→ Huffman:")
print(f"   Decomposed: {huff_decomp:.4f}" if huff_decomp else "   Decomposed: Not found")
print(f"   Standard  : {huff_std:.4f}" if huff_std else "   Standard  : Not found")
print(f"   Ratio     : {huff_decomp/huff_std:.4f}" if huff_decomp and huff_std else "   Ratio     : N/A")

print("\n→ XOR:")
print(f"   Decomposed: {xor_decomp:.4f}" if xor_decomp else "   Decomposed: Not found")
print(f"   Standard  : {xor_std:.4f}" if xor_std else "   Standard  : Not found")
print(f"   Ratio     : {xor_decomp/xor_std:.4f}" if xor_decomp and xor_std else "   Ratio     : N/A")
