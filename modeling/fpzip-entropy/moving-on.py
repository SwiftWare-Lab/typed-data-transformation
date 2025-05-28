# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # ── 1) Read your DatasetIdMapping (entropy) ────────────────────────
# df_entropy = pd.read_csv(
#     "/home/jamalids/Documents/fpzip-zstd/DatasetIdMapping.csv",
#     usecols=["DatasetName","Entropy"]
# )
#
# # ── 2) Read each compressor’s ratios ───────────────────────────────
# df_fpzip = (
#     pd.read_csv("/home/jamalids/Documents/fpzip-zstd/fpzip_ratios.csv")
#       .rename(columns={"Dataset":"DatasetName","FPZIP_Ratio":"CompressionRatio"})
#       .assign(tool="fpzip")
# )
#
# df_zstd = (
#     pd.read_csv("/home/jamalids/Documents/fpzip-zstd/zstd.csv")
#       .assign(RunType=lambda d: d["RunType"].replace({
#           "Chunked_Decompose_Parallel":"TDT",
#           "Chunk-decompose_Parallel":"TDT",
#           "Decompose_Chunk_Parallel":"TDT",
#           "Component":"TDT",
#           "Whole":"standard","Full":"standard","Chunked_parallel":"standard"
#       }))
#       .query("RunType in ['standard','TDT']")
#       .assign(tool=lambda d: d["RunType"].map({"standard":"zstd","TDT":"TDT+zstd"}))
#       [["DatasetName","CompressionRatio","tool"]]
# )
#
# df_snappy = (
#     pd.read_csv("/home/jamalids/Documents/fpzip-zstd/snappy.csv")
#       .assign(RunType=lambda d: d["RunType"].replace({
#           "Chunked_Decompose_Parallel":"TDT",
#           "Chunk-decompose_Parallel":"TDT",
#           "Decompose_Chunk_Parallel":"TDT",
#           "Component":"TDT",
#           "Whole":"standard","Full":"standard","Chunked_parallel":"standard"
#       }))
#       .query("RunType in ['standard','TDT']")
#       .assign(tool=lambda d: d["RunType"].map({"standard":"snappy","TDT":"TDT+snappy"}))
#       [["DatasetName","CompressionRatio","tool"]]
# )
#
# # ── 3) Merge all with entropy ───────────────────────────────────────
# df = (
#     pd.concat([df_fpzip, df_zstd, df_snappy], ignore_index=True)
#       .merge(df_entropy, on="DatasetName")
# )
#
# # ── 4) Compute a 3-point moving average for each tool ───────────────
# window = 3
# ma_list = []
# for tool, sub in df.groupby("tool"):
#     sub = sub.sort_values("Entropy").reset_index(drop=True)
#     sub["MA"] = sub["CompressionRatio"].rolling(window, min_periods=1).mean()
#     ma_list.append(sub)
# ma_df = pd.concat(ma_list, ignore_index=True)
#
# # ── 5) Plot raw points + moving average curves ─────────────────────
# sns.set_style("whitegrid")
# plt.figure(figsize=(8,5))
#
# palette = {
#     "fpzip":  "#e377c2",
#     "zstd":   sns.light_palette("#1f77b4", n_colors=6)[3],
#     "TDT+zstd":"#1f77b4",
#     "snappy": "#6a3d9a",
#     "TDT+snappy":"#6a3d9a"
# }
# markers = {
#     "fpzip":  "X",
#     "zstd":   "o",
#     "TDT+zstd":"s",
#     "snappy": "^",
#     "TDT+snappy":"D"
# }
#
# # scatter
# for tool, sub in df.groupby("tool"):
#     plt.scatter(
#         sub["Entropy"], sub["CompressionRatio"],
#         c=palette[tool], marker=markers[tool],
#         edgecolor="k", s=60, alpha=0.6, label=f"{tool} pts"
#     )
#
# # moving‐average lines
# for tool, sub in ma_df.groupby("tool"):
#     plt.plot(
#         sub["Entropy"], sub["MA"],
#         c=palette[tool], lw=2, label=f"{tool} {window}-pt MA"
#     )
#
# plt.xlabel("Dataset entropy (bits)")
# plt.ylabel("Compression ratio")
# plt.title("Compression ratio vs. entropy\nwith simple moving averages")
# plt.legend(bbox_to_anchor=(1.05,1), loc="upper left")
# plt.tight_layout()
# plt.savefig("/home/jamalids/Documents/entropy_ma.png", dpi=300)
#
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1) Read in your entropy mapping
df_entropy = pd.read_csv(
    "/home/jamalids/Documents/fpzip-zstd/DatasetIdMapping.csv",
    usecols=["DatasetName","Entropy"]
)

# 2) Read each tool’s ratios and tag them
df_fpzip = (
    pd.read_csv("/home/jamalids/Documents/fpzip-zstd/fpzip_ratios.csv")
      .rename(columns={"Dataset":"DatasetName","FPZIP_Ratio":"CompressionRatio"})
      .assign(tool="fpzip")
)
df_zstd = (
    pd.read_csv("/home/jamalids/Documents/fpzip-zstd/zstd.csv")
      .assign(RunType=lambda d: d["RunType"].replace({
          "Chunked_Decompose_Parallel":"TDT",
          "Chunk-decompose_Parallel":"TDT",
          "Decompose_Chunk_Parallel":"TDT",
          "Component":"TDT",
          "Whole":"standard","Full":"standard","Chunked_parallel":"standard"
      }))
      .query("RunType in ['standard','TDT']")
      .assign(tool=lambda d: d["RunType"].map({"standard":"zstd","TDT":"TDT+zstd"}))
      [["DatasetName","CompressionRatio","tool"]]
)
df_snappy = (
    pd.read_csv("/home/jamalids/Documents/fpzip-zstd/snappy.csv")
      .assign(RunType=lambda d: d["RunType"].replace({
          "Chunked_Decompose_Parallel":"TDT",
          "Chunk-decompose_Parallel":"TDT",
          "Decompose_Chunk_Parallel":"TDT",
          "Component":"TDT",
          "Whole":"standard","Full":"standard","Chunked_parallel":"standard"
      }))
      .query("RunType in ['standard','TDT']")
      .assign(tool=lambda d: d["RunType"].map({"standard":"snappy","TDT":"TDT+snappy"}))
      [["DatasetName","CompressionRatio","tool"]]
)

# 3) Merge all with entropy
df = pd.concat([df_fpzip, df_zstd, df_snappy], ignore_index=True) \
       .merge(df_entropy, on="DatasetName")

# 4) Compute a 3-point moving average for each tool
window = 3
ma_dfs = []
for tool, grp in df.groupby("tool"):
    grp = grp.sort_values("Entropy").reset_index(drop=True)
    grp["MA"] = grp["CompressionRatio"].rolling(window, min_periods=1).mean()
    grp["tool"] = tool
    ma_dfs.append(grp)
ma_df = pd.concat(ma_dfs, ignore_index=True)

# 5) Plot only the MA lines
sns.set_style("whitegrid")
plt.figure(figsize=(8,5))

palette = {
    "fpzip":  "#e377c2",
    "zstd":   sns.light_palette("#1f77b4", n_colors=6)[3],
    "TDT+zstd":"#1f77b4",
    "snappy": "#6a3d9a",
    "TDT+snappy":"#6a3d9a"
}

for tool, grp in ma_df.groupby("tool"):
    plt.plot(
        grp["Entropy"],
        grp["MA"],
        color=palette[tool],
        linewidth=2.5,
        label=f"{tool} {window}-pt MA"
    )

plt.xlabel("Dataset entropy (bits)")
plt.ylabel("Compression ratio (geom. mean)")
plt.title("Moving Averages of Compression Ratio vs. Entropy")
plt.legend(bbox_to_anchor=(1.05,1), loc="upper left")
plt.tight_layout()
plt.savefig("/home/jamalids/Documents/entropy_ma_only.png", dpi=300)

##########################################float entropy #############################
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

# ── 1) Metadata table ──────────────────────────────────────────────
meta_csv = """ID,Application,Type,Size,Entropy,Domain
D1,astro-mhd,D,548M,0.97,HPC
D2,rsim,S,94M,18.50,HPC
D3,turbulence,S,67M,23.73,HPC
D4,wave,S,537M,25.27,HPC
D5,num-brain,D,142M,23.97,HPC
D6,num-control,D,160M,24.14,HPC
D7,astro-pt,D,671M,26.32,HPC
D8,msg-bt,D,266M,23.67,HPC
D9,citytemp,S,12M,9.43,TS
D10,wesad-chest,D,272M,13.85,TS
D11,solar-wind,S,424M,14.06,TS
D12,hdr-night,S,537M,9.03,OBS
D13,hdr-palermo,S,843M,9.34,OBS
D14,hst-wfc3-ir,S,24M,15.04,OBS
D15,hst-wfc3-uvis,S,109M,15.61,OBS
D16,acs-wht,S,225M,20.13,OBS
D17,spitzer-irac,S,165M,20.54,OBS
D18,g24-78-usb,S,1.3G,26.02,OBS
D19,jws-mirimage,S,169M,23.16,OBS
D20,tpcxBB-store,D,790M,16.73,DB
D21,tpcxBB-web,D,987M,17.64,DB
D22,tpcH-order,D,120M,23.40,DB
D23,tpcDS-catalog,S,173M,17.34,DB
D24,tpcDS-store,S,277M,15.17,DB
D25,tpcDS-web,S,86M,17.33,DB
D26,Llama [38],H,48G,4.88,ML
"""
meta = pd.read_csv(StringIO(meta_csv))
suffix_map = {'S':'f32','D':'f64'}
meta['suffix'] = meta['Type'].map(suffix_map)
meta = meta.dropna(subset=['suffix'])
meta['DatasetName'] = meta['Application'].str.replace('-', '_') + '_' + meta['suffix']
df_entropy = meta[['DatasetName','Entropy']]

# ── 2) Load compression ratios ─────────────────────────────────────
df_fpzip = pd.read_csv("/home/jamalids/Documents/fpzip-zstd/fpzip_ratios.csv") \
    .rename(columns={'Dataset':'DatasetName','FPZIP_Ratio':'CompressionRatio'}) \
    .assign(tool='fpzip')

df_zstd = pd.read_csv("/home/jamalids/Documents/fpzip-zstd/zstd.csv") \
    .assign(RunType=lambda d: d['RunType'].replace({
        "Chunked_Decompose_Parallel":"TDT",
        "Chunk-decompose_Parallel":"TDT",
        "Decompose_Chunk_Parallel":"TDT",
        "Component":"TDT",
        "Whole":"standard","Full":"standard","Chunked_parallel":"standard"
    })) \
    .query("RunType in ['standard','TDT']") \
    .assign(tool=lambda d: d['RunType'].map({'standard':'zstd','TDT':'TDT+zstd'})) \
    [['DatasetName','CompressionRatio','tool']]

df_snappy = pd.read_csv("/home/jamalids/Documents/fpzip-zstd/snappy.csv") \
    .assign(RunType=lambda d: d['RunType'].replace({
        "Chunked_Decompose_Parallel":"TDT",
        "Chunk-decompose_Parallel":"TDT",
        "Decompose_Chunk_Parallel":"TDT",
        "Component":"TDT",
        "Whole":"standard","Full":"standard","Chunked_parallel":"standard"
    })) \
    .query("RunType in ['standard','TDT']") \
    .assign(tool=lambda d: d['RunType'].map({'standard':'snappy','TDT':'TDT+snappy'})) \
    [['DatasetName','CompressionRatio','tool']]

# ── 3) Merge with entropy ───────────────────────────────────────────
df = pd.concat([df_fpzip, df_zstd, df_snappy], ignore_index=True) \
       .merge(df_entropy, on='DatasetName')

# ── 4) Compute 3-point moving average per tool ─────────────────────
window = 3
ma_list = []
for tool, grp in df.groupby('tool'):
    grp = grp.sort_values('Entropy').reset_index(drop=True)
    grp['MA'] = grp['CompressionRatio'].rolling(window, min_periods=1).mean()
    grp['tool'] = tool
    ma_list.append(grp)
ma_df = pd.concat(ma_list, ignore_index=True)

# ── 5) Plot only the moving-average lines ─────────────────────────
sns.set_style("whitegrid")
plt.figure(figsize=(8,5))

palette = {
    'fpzip':'#e377c2',
    'zstd':sns.light_palette('#1f77b4', n_colors=6)[3],
    'TDT+zstd':'#1f77b4',
    'snappy':'#6a3d9a',
    'TDT+snappy':'#6a3d9a'
}

for tool, grp in ma_df.groupby('tool'):
    plt.plot(
        grp['Entropy'], grp['MA'],
        color=palette[tool], linewidth=2.5,
        label=f"{tool} {window}-pt MA"
    )

plt.xlabel("Dataset entropy (bits)")
plt.ylabel("Compression ratio")
plt.title("Compression ratio vs. entropy (Moving Averages)")
plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
plt.tight_layout()
plt.savefig("/home/jamalids/Documents/entropy_ma_float.png", dpi=300)

####################################
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

# ── 1) Inline entropy mapping ───────────────────────────────────────
meta_csv = """ID,Application,Type,Size,Entropy,Domain
D1,astro-mhd,D,548M,0.97,HPC
D2,rsim,S,94M,18.50,HPC
D3,turbulence,S,67M,23.73,HPC
D4,wave,S,537M,25.27,HPC
D5,num-brain,D,142M,23.97,HPC
D6,num-control,D,160M,24.14,HPC
D7,astro-pt,D,671M,26.32,HPC
D8,msg-bt,D,266M,23.67,HPC
D9,citytemp,S,12M,9.43,TS
D10,wesad-chest,D,272M,13.85,TS
D11,solar-wind,S,424M,14.06,TS
D12,hdr-night,S,537M,9.03,OBS
D13,hdr-palermo,S,843M,9.34,OBS
D14,hst-wfc3-ir,S,24M,15.04,OBS
D15,hst-wfc3-uvis,S,109M,15.61,OBS
D16,acs-wht,S,225M,20.13,OBS
D17,spitzer-irac,S,165M,20.54,OBS
D18,g24-78-usb,S,1.3G,26.02,OBS
D19,jws-mirimage,S,169M,23.16,OBS
D20,tpcxBB-store,D,790M,16.73,DB
D21,tpcxBB-web,D,987M,17.64,DB
D22,tpcH-order,D,120M,23.40,DB
D23,tpcDS-catalog,S,173M,17.34,DB
D24,tpcDS-store,S,277M,15.17,DB
D25,tpcDS-web,S,86M,17.33,DB
D26,Llama [38],H,48G,4.88,ML
"""
meta = pd.read_csv(StringIO(meta_csv))
suffix_map = {'S':'f32','D':'f64'}
meta['suffix'] = meta['Type'].map(suffix_map)
meta = meta.dropna(subset=['suffix'])
meta['DatasetName'] = meta['Application'].str.replace('-', '_') + '_' + meta['suffix']
df_entropy = meta[['DatasetName','Entropy']]

# ── 2) Load ratios for fpzip and zstd ──────────────────────────────
df_fpzip = (
    pd.read_csv("/home/jamalids/Documents/fpzip-zstd/fpzip_ratios.csv")
      .rename(columns={"Dataset":"DatasetName","FPZIP_Ratio":"CompressionRatio"})
      .assign(tool="fpzip")
)

df_zstd = (
    pd.read_csv("/home/jamalids/Documents/fpzip-zstd/zstd.csv")
      .assign(RunType=lambda d: d["RunType"].replace({
          "Chunked_Decompose_Parallel":"TDT",
          "Chunk-decompose_Parallel":"TDT",
          "Decompose_Chunk_Parallel":"TDT",
          "Component":"TDT",
          "Whole":"standard","Full":"standard","Chunked_parallel":"standard"
      }))
      .query("RunType in ['standard','TDT']")
      .assign(tool=lambda d: d["RunType"].map({"standard":"zstd","TDT":"TDT+zstd"}))
      [["DatasetName","CompressionRatio","tool"]]
)

# ── 3) Merge with entropy ───────────────────────────────────────────
df = pd.concat([df_fpzip, df_zstd], ignore_index=True) \
       .merge(df_entropy, on="DatasetName")

# ── 4) Compute 5-point moving average sorted by entropy ────────────
window =5
ma_frames = []
for tool, grp in df.groupby("tool"):
    grp_sorted = grp.sort_values("Entropy").reset_index(drop=True)
    grp_sorted["MA"] = grp_sorted["CompressionRatio"].rolling(window, min_periods=1).mean()
    ma_frames.append(grp_sorted)

ma_df = pd.concat(ma_frames, ignore_index=True)

# ── 5) Plot only moving-average lines ──────────────────────────────
sns.set_style("whitegrid")
plt.figure(figsize=(8,5))

palette = {
    "fpzip":    "#e377c2",
    "zstd":     sns.light_palette("#1f77b4", n_colors=6)[3],
    "TDT+zstd": "#1f77b4"
}

for tool, grp in ma_df.groupby("tool"):
    plt.plot(
        grp["Entropy"],
        grp["MA"],
        color=palette[tool],
        linewidth=2.5,
        label=f"{tool} {window}-pt MA"
    )

plt.xlabel("Dataset entropy (bits)")
plt.ylabel("Compression ratio")
plt.title("Compression ratio vs. entropy (5-pt Moving Average)")
plt.legend(bbox_to_anchor=(1.05,1), loc="upper left")
plt.tight_layout()
plt.savefig("/home/jamalids/Documents/entropy_5.png", dpi=300)
plt.show()
