import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from io import StringIO

# 🌟 Set global font and plot style (VLDB style)
mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Latin Modern Roman", "Times", "DejaVu Serif"],
    "font.size": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

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
meta['suffix'] = meta['Type'].map({'S':'f32','D':'f64'})
meta = meta.dropna(subset=['suffix'])
meta['DatasetName'] = meta['Application'].str.replace('-', '_') + '_' + meta['suffix']
df_entropy = meta[['DatasetName','Entropy']]

# ── 2) Load fpzip & zstd ratios ────────────────────────────────────
df_fpzip = pd.read_csv("/home/jamalids/Documents/fpzip-zstd/fpzip_ratios.csv") \
    .rename(columns={'Dataset':'DatasetName','FPZIP_Ratio':'CompressionRatio'}) \
    .assign(tool='fpzip')

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

# ── 3) Merge with entropy ───────────────────────────────────────────
df = pd.concat([df_fpzip, df_zstd], ignore_index=True) \
       .merge(df_entropy, on='DatasetName')

# ── 4) Compute 5-point moving average sorted by entropy ─────────────
window = 5
ma_frames = []
for tool, grp in df.groupby('tool'):
    sorted_grp = grp.sort_values('Entropy').reset_index(drop=True)
    sorted_grp['MA'] = sorted_grp['CompressionRatio'].rolling(window, min_periods=1).mean()
    ma_frames.append(sorted_grp)
ma_df = pd.concat(ma_frames, ignore_index=True)

# ── 5) Plot only moving-average curves ──────────────────────────────
sns.set_style("white")
plt.figure(figsize=(6.8, 3))

palette = {
    'fpzip':    '#e377c2',
    'zstd':     sns.light_palette('#1f77b4', n_colors=6)[3],
    'TDT+zstd': '#1f77b4'
}

legend_labels = {
    'TDT+zstd': 'Typed Data Transformed (TDT) + Zstd',
    'zstd': 'Zstd',
    'fpzip': 'Fpzip'
}

for tool, grp in ma_df.groupby('tool'):
    plt.plot(
        grp['Entropy'],
        grp['MA'],
        color=palette[tool],
        linewidth=2.5,
        label=legend_labels[tool]
    )

plt.grid(False)
plt.xlabel("Entropy")
plt.ylabel("Compression ratio")
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig("/home/jamalids/Documents/entropy_5pt.pdf", format='pdf')

