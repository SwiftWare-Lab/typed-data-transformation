# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy.stats import gmean
# from io import StringIO
#
# # ── 1) Metadata table ──────────────────────────────────────────────
# meta_csv = """ID,Application,Type,Size,Entropy,Domain
# D1,astro-mhd,D,548M,0.97,HPC
# D2,rsim,S,94M,18.50,HPC
# D3,turbulence,S,67M,23.73,HPC
# D4,wave,S,537M,25.27,HPC
# D5,num-brain,D,142M,23.97,HPC
# D6,num-control,D,160M,24.14,HPC
# D7,astro-pt,D,671M,26.32,HPC
# D8,msg-bt,D,266M,23.67,HPC
# D9,citytemp,S,12M,9.43,TS
# D10,wesad-chest,D,272M,13.85,TS
# D11,solar-wind,S,424M,14.06,TS
# D12,hdr-night,S,537M,9.03,OBS
# D13,hdr-palermo,S,843M,9.34,OBS
# D14,hst-wfc3-ir,S,24M,15.04,OBS
# D15,hst-wfc3-uvis,S,109M,15.61,OBS
# D16,acs-wht,S,225M,20.13,OBS
# D17,spitzer-irac,S,165M,20.54,OBS
# D18,g24-78-usb,S,1.3G,26.02,OBS
# D19,jws-mirimage,S,169M,23.16,OBS
# D20,tpcxBB-store,D,790M,16.73,DB
# D21,tpcxBB-web,D,987M,17.64,DB
# D22,tpcH-order,D,120M,23.40,DB
# D23,tpcDS-catalog,S,173M,17.34,DB
# D24,tpcDS-store,S,277M,15.17,DB
# D25,tpcDS-web,S,86M,17.33,DB
# D26,Llama [38],H,48G,4.88,ML
# """
# meta = pd.read_csv(StringIO(meta_csv))
# suffix_map = {'S':'f32','D':'f64'}
# meta['suffix'] = meta['Type'].map(suffix_map)
# meta = meta.dropna(subset=['suffix'])
# meta['DatasetName'] = meta['Application'].str.replace('-', '_') + '_' + meta['suffix']
#
# # ── 2) Load compression ratios from uploaded files ──────────────────
# df_fpzip = pd.read_csv("/home/jamalids/Documents/fpzip-zstd/fpzip_ratios.csv") \
#     .rename(columns={'Dataset':'DatasetName','FPZIP_Ratio':'CompressionRatio'})
# df_fpzip['tool'] = 'fpzip'
#
# df_zstd = pd.read_csv("/home/jamalids/Documents/fpzip-zstd/zstd.csv")
# df_zstd['RunType'] = df_zstd['RunType'].replace({
#     "Chunked_Decompose_Parallel":"TDT",
#     "Chunk-decompose_Parallel":"TDT",
#     "Decompose_Chunk_Parallel":"TDT",
#     "Component":"TDT",
#     "Whole":"standard","Full":"standard","Chunked_parallel":"standard"
# })
# df_zstd = df_zstd[df_zstd['RunType'].isin(['standard','TDT'])].copy()
# df_zstd['tool'] = df_zstd['RunType'].map({'standard':'zstd','TDT':'TDT+zstd'})
# df_zstd = df_zstd[['DatasetName','CompressionRatio','tool']]
#
# df_snappy = pd.read_csv("/home/jamalids/Documents/fpzip-zstd/snappy.csv")
# df_snappy['RunType'] = df_snappy['RunType'].replace({
#     "Chunked_Decompose_Parallel":"TDT",
#     "Chunk-decompose_Parallel":"TDT",
#     "Decompose_Chunk_Parallel":"TDT",
#     "Component":"TDT",
#     "Whole":"standard","Full":"standard","Chunked_parallel":"standard"
# })
# df_snappy = df_snappy[df_snappy['RunType'].isin(['standard','TDT'])].copy()
# df_snappy['tool'] = df_snappy['RunType'].map({'standard':'snappy','TDT':'TDT+snappy'})
# df_snappy = df_snappy[['DatasetName','CompressionRatio','tool']]
#
# # ── 3) Merge everything with entropy ───────────────────────────────
# df = pd.concat([df_fpzip, df_zstd, df_snappy], ignore_index=True) \
#        .merge(meta[['DatasetName','Entropy','ID']], on='DatasetName')
#
# # ── 4) Compute geometric means for low vs. high entropy ───────────
# threshold = 16
# agg = []
# for tool, grp in df.groupby('tool'):
#     low = grp[grp['Entropy'] <= threshold]
#     high = grp[grp['Entropy'] > threshold]
#     if not low.empty:
#         agg.append({
#             'tool': tool,
#             'category': 'low',
#             'Entropy': low['Entropy'].mean(),
#             'gmean_ratio': gmean(low['CompressionRatio'])
#         })
#     if not high.empty:
#         agg.append({
#             'tool': tool,
#             'category': 'high',
#             'Entropy': high['Entropy'].mean(),
#             'gmean_ratio': gmean(high['CompressionRatio'])
#         })
# agg_df = pd.DataFrame(agg)
#
# # ── 5) Plot aggregated points ──────────────────────────────────────
# sns.set_style("whitegrid")
# plt.figure(figsize=(8,5))
#
# base_colors = {'zstd':'#1f77b4','snappy':'#6a3d9a','fpzip':'#e377c2'}
# palette = {
#     'fpzip':'#e377c2',
#     'zstd':sns.light_palette(base_colors['zstd'], n_colors=6)[3],
#     'TDT+zstd':base_colors['zstd'],
#     'snappy':sns.light_palette(base_colors['snappy'], n_colors=6)[3],
#     'TDT+snappy':base_colors['snappy']
# }
#
# markers = {'low':'o','high':'s'}
#
# for tool in agg_df['tool'].unique():
#     sub = agg_df[agg_df['tool']==tool]
#     for _, row in sub.iterrows():
#         plt.scatter(
#             row['Entropy'], row['gmean_ratio'],
#             label=f"{tool}-{row['category']}",
#             color=palette[tool],
#             marker=markers[row['category']],
#             edgecolor='k',
#             s=150,
#             alpha=0.9
#         )
#
# plt.xlabel("Mean dataset entropy (bits)")
# plt.ylabel("Geometric mean compression ratio")
# plt.title("Low vs. High Entropy Performance by Tool")
# plt.legend(title="Tool & Category", bbox_to_anchor=(1.02,1), loc='upper left')
# plt.tight_layout()
# plt.savefig("/home/jamalids/Documents/low-high.png", dpi=300)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gmean
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

# ── 2) Load compression ratios ─────────────────────────────────────
df_fpzip = pd.read_csv("/home/jamalids/Documents/fpzip-zstd/fpzip_ratios.csv") \
    .rename(columns={'Dataset':'DatasetName','FPZIP_Ratio':'CompressionRatio'})
df_fpzip['tool'] = 'fpzip'

df_zstd = pd.read_csv("/home/jamalids/Documents/fpzip-zstd/zstd.csv")
df_zstd['RunType'] = df_zstd['RunType'].replace({
    "Chunked_Decompose_Parallel":"TDT",
    "Chunk-decompose_Parallel":"TDT",
    "Decompose_Chunk_Parallel":"TDT",
    "Component":"TDT",
    "Whole":"standard","Full":"standard","Chunked_parallel":"standard"
})
df_zstd = df_zstd[df_zstd['RunType'].isin(['standard','TDT'])].copy()
df_zstd['tool'] = df_zstd['RunType'].map({'standard':'zstd','TDT':'TDT+zstd'})
df_zstd = df_zstd[['DatasetName','CompressionRatio','tool']]

df_snappy = pd.read_csv("/home/jamalids/Documents/fpzip-zstd/snappy.csv")
df_snappy['RunType'] = df_snappy['RunType'].replace({
    "Chunked_Decompose_Parallel":"TDT",
    "Chunk-decompose_Parallel":"TDT",
    "Decompose_Chunk_Parallel":"TDT",
    "Component":"TDT",
    "Whole":"standard","Full":"standard","Chunked_parallel":"standard"
})
df_snappy = df_snappy[df_snappy['RunType'].isin(['standard','TDT'])].copy()
df_snappy['tool'] = df_snappy['RunType'].map({'standard':'snappy','TDT':'TDT+snappy'})
df_snappy = df_snappy[['DatasetName','CompressionRatio','tool']]

# ── 3) Merge with entropy info ─────────────────────────────────────
df = pd.concat([df_fpzip, df_zstd, df_snappy], ignore_index=True) \
       .merge(meta[['DatasetName','Entropy']], on='DatasetName')

# ── 4) Compute gmean by tool & entropy category ───────────────────
threshold = 16
agg = []
for tool, grp in df.groupby('tool'):
    low = grp[grp['Entropy'] <= threshold]
    high = grp[grp['Entropy'] > threshold]
    agg.append({'tool': tool, 'category': 'low',  'gmean_ratio': gmean(low['CompressionRatio'])})
    agg.append({'tool': tool, 'category': 'high', 'gmean_ratio': gmean(high['CompressionRatio'])})
agg_df = pd.DataFrame(agg)

# ── 5) Plot grouped bar chart ─────────────────────────────────────
sns.set_style("whitegrid")
plt.figure(figsize=(8,5))

sns.barplot(
    data=agg_df,
    x='tool',
    y='gmean_ratio',
    hue='category',
    ci=None
)

plt.xlabel("Compression tool")
plt.ylabel("Geometric mean compression ratio")
plt.title("Low vs. High Entropy Performance by Tool")
plt.legend(title="Entropy category", loc='upper right')
plt.tight_layout()
plt.savefig("/home/jamalids/Documents/low-high.png", dpi=300)
