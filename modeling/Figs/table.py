import os
import pandas as pd
import re

# --- Step 1: Define dataset-to-domain mapping ---
domain_mapping = {
    "msg_bt": "HPC", "num_brain": "HPC", "num_control": "HPC", "rsim": "HPC",
    "astro_mhd": "HPC", "astro_pt": "HPC", "miranda3d": "HPC", "turbulence": "HPC",
    "wave": "HPC", "citytemp": "TS",
    "wesad_chest": "TS",  "solar_wind": "TS",
    "acs_wht": "OBS", "hdr_night": "OBS", "hdr_palermo": "OBS", "hst_wfc3_uvis": "OBS",
    "hst_wfc3_ir": "OBS", "spitzer_irac": "OBS", "g24_78_usb": "OBS", "jw_mirimage": "OBS",
    "tpcH_order": "DB", "tpcxbb_store": "DB", "tpcxbb_web": "DB", "tpcds_catalog": "DB",
    "tpcds_store": "DB", "tpch_order": "DB", "tpcds_web": "DB","LLama" : "ML"
}

# --- Step 2: Normalize dataset names ---
def remove_trailing_pattern(s):
    return re.sub(r'(_?[fF]\d*)$', '', s)

# --- Step 3: Load CSV ---
input_path = "/mnt/c/Users/jamalids/Downloads/combine.csv"
df = pd.read_csv(input_path)

df['NormalizedDatasetName'] = df['DatasetName'].apply(remove_trailing_pattern)
df['Domain'] = df['NormalizedDatasetName'].map(domain_mapping)

# --- Step 4: Combine RunType and tool ---
if "RunType" in df.columns:
    df["runytpe_compression_tool"] = df["RunType"].astype(str) + "_" + df["compression_tool"].astype(str)
else:
    df["runytpe_compression_tool"] = df["compression_tool"]

# --- Step 5: Pivot ---
pivot_df = df.pivot_table(index=["NormalizedDatasetName", "Domain"],
                          columns="runytpe_compression_tool",
                          values="CompressionRatio",
                          aggfunc="first")
pivot_df = pivot_df.fillna("")
pivot_df.reset_index(inplace=True)

# --- Step 6: Save LaTeX table manually ---
output_tex = "/mnt/c/Users/jamalids/Downloads/compression_table1.tex"
with open(output_tex, "w") as f:
    f.write("\\begin{table*}[!ht]\n")
    f.write("\\centering\n")
    f.write("\\resizebox{\\linewidth}{!}{%\n")
    f.write("\\begin{tabular}{" + "l" * len(pivot_df.columns) + "}\n")
    f.write("\\toprule\n")

    # Header row
    f.write(" & ".join([f"\\textbf{{{col}}}" for col in pivot_df.columns]) + " \\\\\n")
    f.write("\\midrule\n")

    # Data rows
    for _, row in pivot_df.iterrows():
        row_line = " & ".join(str(x) if x != "" else "\\phantom{0}" for x in row)
        f.write(row_line + " \\\\\n")

    f.write("\\bottomrule\n")
    f.write("\\end{tabular}%\n")
    f.write("}\n")
    f.write("\\caption{Compression ratio comparison (TDT vs. standard)}\n")
    f.write("\\label{tab:CompressionRatio}\n")
    f.write("\\end{table*}\n")

print(f"LaTeX table saved to {output_tex}")
