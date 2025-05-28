# import pandas as pd
# import matplotlib.pyplot as plt
#
# # ── 1) Load entropy data ─────────────────────────────────────────────
# df_entropy = pd.read_csv("/home/jamalids/Documents/fpzip-zstd/DatasetIdMapping.csv")[['DatasetName', 'Entropy']]
#
# # ── 2) Load fpzip ratios ────────────────────────────────────────────
# df_fpzip = pd.read_csv("/home/jamalids/Documents/fpzip-zstd/fpzip_ratios.csv")
# df_fpzip = df_fpzip.rename(columns={'Dataset': 'DatasetName', 'FPZIP_Ratio': 'FPZIP_Ratio'})
#
# # ── 3) Load zstd data and filter to standard runs ──────────────────
# df_zstd = pd.read_csv("/home/jamalids/Documents/fpzip-zstd/zstd.csv")
# # Map RunType to standard vs TDT
# df_zstd['RunType'] = df_zstd['RunType'].replace({
#     "Chunked_Decompose_Parallel": "TDT",
#     "Chunk-decompose_Parallel":  "TDT",
#     "Decompose_Chunk_Parallel":  "TDT",
#     "Component":                 "TDT",
#     "Whole":                     "standard",
#     "Full":                      "standard",
#     "Chunked_parallel":          "standard"
# })
# df_zstd = df_zstd[df_zstd['RunType']=='standard'][['DatasetName', 'CompressionRatio']]
# df_zstd = df_zstd.rename(columns={'CompressionRatio': 'ZSTD_Ratio'})
#
# # ── 4) Merge all on DatasetName ────────────────────────────────────
# df = df_entropy.merge(df_fpzip[['DatasetName','FPZIP_Ratio']], on='DatasetName')
# df = df.merge(df_zstd, on='DatasetName')
#
# # ── 5) Sort by entropy ─────────────────────────────────────────────
# df = df.sort_values('Entropy')
#
# # ── 6) Plot line trends ────────────────────────────────────────────
# plt.figure(figsize=(6, 4))
# plt.plot(df['Entropy'], df['FPZIP_Ratio'], marker='o', linestyle='-', label='fpzip')
# plt.plot(df['Entropy'], df['ZSTD_Ratio'],  marker='s', linestyle='--', label='zstd')
# plt.xlabel("Dataset entropy")
# plt.ylabel("Compression ratio")
# plt.title("Compression ratio vs. entropy\n(fpzip vs. zstd standard)")
# plt.legend()
# plt.tight_layout()
# plt.savefig("/home/jamalids/Documents/entropy-fpzip-zstd")
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

# ── 2) Load fpzip ratios ───────────────────────────────────────────
df_fpzip = pd.read_csv("/home/jamalids/Documents/fpzip-zstd/fpzip_ratios.csv") \
    .rename(columns={'Dataset':'DatasetName','FPZIP_Ratio':'CompressionRatio'})
df_fpzip['tool'] = 'fpzip'

# ── 3) Load zstd (standard & TDT) ─────────────────────────────────
df_zstd = pd.read_csv("/home/jamalids/Documents/fpzip-zstd/zstd.csv")
df_zstd['RunType'] = df_zstd['RunType'].replace({
    "Chunked_Decompose_Parallel":"TDT",
    "Chunk-decompose_Parallel":"TDT",
    "Decompose_Chunk_Parallel":"TDT",
    "Component":"TDT",
    "Whole":"standard","Full":"standard","Chunked_parallel":"standard"
})
df_zstd = df_zstd[df_zstd['RunType'].isin(['standard','TDT'])]
df_zstd['tool'] = df_zstd['RunType'].map({'standard':'zstd','TDT':'TDT+zstd'})
df_zstd = df_zstd[['DatasetName','CompressionRatio','tool']]

# ── 4) Load snappy (standard & TDT) ────────────────────────────────
df_snappy = pd.read_csv("/home/jamalids/Documents/fpzip-zstd/snappy.csv")
df_snappy['RunType'] = df_snappy['RunType'].replace({
    "Chunked_Decompose_Parallel":"TDT",
    "Chunk-decompose_Parallel":"TDT",
    "Decompose_Chunk_Parallel":"TDT",
    "Component":"TDT",
    "Whole":"standard","Full":"standard","Chunked_parallel":"standard"
})
df_snappy = df_snappy[df_snappy['RunType'].isin(['standard','TDT'])]
df_snappy['tool'] = df_snappy['RunType'].map({'standard':'snappy','TDT':'TDT+snappy'})
df_snappy = df_snappy[['DatasetName','CompressionRatio','tool']]

# ── 5) Merge everything with entropy ───────────────────────────────
df = pd.concat([df_fpzip, df_zstd, df_snappy], ignore_index=True) \
       .merge(meta[['DatasetName','Entropy','ID']], on='DatasetName')

# ── 6) Plot scatter ────────────────────────────────────────────────
sns.set_style("whitegrid")
plt.figure(figsize=(8,5))

# build your palette
base_colors = {
    'zstd':   '#1f77b4',
    'snappy': '#6a3d9a',
    'fpzip':  '#e377c2',
}
palette = {
    'fpzip':        base_colors['fpzip'],
    'zstd':         sns.light_palette(base_colors['zstd'], n_colors=6)[3],
    'TDT+zstd':     base_colors['zstd'],
    'snappy':       sns.light_palette(base_colors['snappy'], n_colors=6)[3],
    'TDT+snappy':   base_colors['snappy'],
}
markers = {
    'fpzip':        'X',
    'zstd':         'o',
    'TDT+zstd':     's',
    'snappy':       '^',
    'TDT+snappy':   'D'
}

for tool, grp in df.groupby('tool'):
    plt.scatter(
        grp['Entropy'], grp['CompressionRatio'],
        label=tool,
        c=palette[tool],
        marker=markers[tool],
        edgecolor='k',
        s=100,
        alpha=0.8
    )


plt.xlabel("Dataset entropy (bits)")
plt.ylabel("Compression ratio")
plt.title("Compression ratio vs. entropy\n(fpzip, zstd, snappy & TDT runs)")
plt.legend(title="Tool", bbox_to_anchor=(1.02,1), loc='upper left')
plt.tight_layout()
plt.savefig("/home/jamalids/Documents/entropy-fpzip-zstd-snappy.png", dpi=300)
