# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from scipy.stats import gmean, sem
#
# # ── helper to load any CSV and tag it with tool+RunType ───────────────────────
# def load_tool(path, tool, dataset_col, ratio_col):
#     df = pd.read_csv(path)
#     df = df.rename(columns={ dataset_col: "DatasetName",
#                               ratio_col:    "CompressionRatio" })
#     df["compression_tool"] = tool
#     # normalize RunType exactly like zstd.csv
#     df["RunType"] = df["RunType"].replace({
#         "Chunked_Decompose_Parallel": "TDT",
#         "Chunk-decompose_Parallel":   "TDT",
#         "Decompose_Chunk_Parallel":   "TDT",
#         "Component":                  "TDT",
#         "Whole":                      "standard",
#         "Full":                       "standard",
#         "Chunked_parallel":           "standard"
#     })
#     df = df[df["RunType"].isin(["standard", "TDT"])]
#     return df[["DatasetName","compression_tool","RunType","CompressionRatio"]]
#
# # ── 1) Load each tool’s CSV ────────────────────────────────────────────────
# zstd    = load_tool("/home/jamalids/Documents/fpzip-zstd/zstd.csv",    "zstd",   "DatasetName",    "CompressionRatio")
# lz4     = load_tool("/home/jamalids/Documents/fpzip-zstd/lz4.csv",     "lz4",    "DatasetName",    "CompressionRatio")
# snappy  = load_tool("/home/jamalids/Documents/fpzip-zstd/snappy.csv",  "snappy", "DatasetName",    "CompressionRatio")
# bzip    = load_tool("/home/jamalids/Documents/fpzip-zstd/bzip.csv",    "bzip",   "DatasetName",    "CompressionRatio")
# nvcomp  = load_tool("/home/jamalids/Documents/fpzip-zstd/nvcomp.csv",  "nvCOMP","DatasetName",    "CompressionRatio")
# zlib  = load_tool("/home/jamalids/Documents/fpzip-zstd/zlib.csv",  "zlib","DatasetName",    "CompressionRatio")
# # add these lines to inspect what you actually loaded:
# print("== zlib RunType counts ==")
# print(zlib["RunType"].value_counts(), "\n")
#
# print("== example zlib rows ==")
# print(zlib.head(10), "\n")
#
# # fpzip only has "standard" so we force RunType
# fpzip = pd.read_csv("/home/jamalids/Documents/fpzip-zstd/fpzip_ratios.csv")
# fpzip = fpzip.rename(columns={"Dataset":"DatasetName", "FPZIP_Ratio":"CompressionRatio"})
# fpzip["compression_tool"] = "fpzip"
# fpzip["RunType"] = "standard"
# fpzip = fpzip[["DatasetName","compression_tool","RunType","CompressionRatio"]]
#
# # ── 2) Keep only datasets present in *all* tools ──────────────────────────
# all_dfs = [fpzip, zstd, lz4, snappy, bzip, nvcomp,zlib]
# common = set.intersection(*(set(df["DatasetName"]) for df in all_dfs))
# all_dfs = [df[df["DatasetName"].isin(common)] for df in all_dfs]
#
# # ── 3) Concatenate & compute g-means + SEMs ────────────────────────────────
# df_plot = pd.concat(all_dfs, ignore_index=True)
#
# groups = [
#     ("fpzip",  "standard"),
#     ("lz4",    "standard"), ("lz4",    "TDT"),
#     ("snappy", "standard"), ("snappy", "TDT"),
#     ("bzip",   "standard"), ("bzip",   "TDT"),
#     ("nvCOMP", "standard"), ("nvCOMP", "TDT"),
#     ("zstd",   "standard"), ("zstd",   "TDT"),
#     ("zlib",   "standard"), ("zlib",   "TDT"),
# ]
# labels = [
#     "fpzip",
#     "lz4",    "TDT+lz4",
#     "snappy", "TDT+snappy",
#     "bzip",   "TDT+bzip",
#     "nvCOMP", "TDT+nvCOMP",
#     "zstd",   "TDT+zstd",
#     "zlib", "TDT+zlib"
# ]
#
# means, errs = [], []
# for tool, run in groups:
#     vals = df_plot.query(
#         "compression_tool == @tool and RunType == @run"
#     )["CompressionRatio"].dropna()
#     means.append(gmean(vals))
#     errs.append(sem(vals))
#
# # ── 4) Build colors & plot ────────────────────────────────────────────────
# base_colors = {
#     'zstd':   '#1f77b4',
#     'snappy': '#6a3d9a',
#     'lz4':    '#d62728',
#     'bzip':   '#2ca02c',
#     'nvCOMP': '#bcbd22',
#     'zlib':   '#17becf',
#     'fpzip':  '#e377c2',
# }
#
# palette = {}
# for t, c in base_colors.items():
#     palette[t]        = sns.light_palette(c, n_colors=6, input="hex")[3]
#     palette[f"TDT+{t}"] = c
# # keep fpzip solid
# palette["fpzip"] = base_colors["fpzip"]
#
# colors = [palette[label] for label in labels]
#
# sns.set_style("ticks")
# fig, ax = plt.subplots(figsize=(8, 5))
# y_pos = range(len(labels))
# bars = ax.barh(y_pos, means, xerr=errs, height=0.6, color=colors, capsize=4)
#
# ax.set_yticks(y_pos)
# ax.set_yticklabels(labels, fontsize=11)
# ax.set_xlabel("Geometric mean compression ratio", fontsize=12)
# ax.set_xlim(0, max(means) * 1.2)
# sns.despine(left=True, bottom=False)
#
# for bar, m in zip(bars, means):
#     ax.text(
#         bar.get_width() + max(means)*0.015,
#         bar.get_y() + bar.get_height()/2,
#         f"{m:.2f}",
#         va="center", ha="left", fontsize=10,
#         bbox=dict(facecolor="white", alpha=0.8, pad=1, edgecolor="none")
#     )
#
# plt.tight_layout()
# plt.savefig('/home/jamalids/Documents/fpzip-compare-all-tools.png', dpi=300)
#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
#  Compare Typed-Data-Transformed (TDT) variants against fpzip
#  and plot CRI = (TDT+tool) / fpzip
# ─────────────────────────────────────────────────────────────────────────────

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gmean, sem

# ── helper to load any CSV and tag it with tool + RunType ────────────────────
def load_tool(path, tool, dataset_col, ratio_col):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df = df.rename(columns={dataset_col: "DatasetName",
                            ratio_col:    "CompressionRatio"})
    df["compression_tool"] = tool
    # normalise RunType names
    df["RunType"] = df["RunType"].replace({
        "Chunked_Decompose_Parallel": "TDT",
        "Chunk-decompose_Parallel":   "TDT",
        "Decompose_Chunk_Parallel":   "TDT",
        "Component":                  "TDT",
        "Whole":                      "standard",
        "Full":                       "standard",
        "Chunked_parallel":           "standard"
    })

    df = df[df["RunType"].isin(["standard", "TDT"])]
    return df[["DatasetName", "compression_tool", "RunType", "CompressionRatio"]]

# ── 1) Load each tool’s CSV ─────────────────────────────────────────────────
zstd   = load_tool("/home/jamalids/Documents/fpzip-zstd/zstd.csv",    "zstd",   "DatasetName", "CompressionRatio")
lz4    = load_tool("/home/jamalids/Documents/fpzip-zstd/lz4.csv",     "lz4",    "DatasetName", "CompressionRatio")
snappy = load_tool("/home/jamalids/Documents/fpzip-zstd/snappy.csv",  "snappy", "DatasetName", "CompressionRatio")
bzip   = load_tool("/home/jamalids/Documents/fpzip-zstd/bzip.csv",    "bzip",   "DatasetName", "CompressionRatio")
nvcomp = load_tool("/home/jamalids/Documents/fpzip-zstd/nvcomp.csv",  "nvCOMP", "DatasetName", "CompressionRatio")
zlib   = load_tool("/home/jamalids/Documents/fpzip-zstd/zlib.csv",    "zlib",   "DatasetName", "CompressionRatio")

# fpzip only has "standard"
fpzip = pd.read_csv("/home/jamalids/Documents/fpzip-zstd/fpzip_ratios.csv")
fpzip = fpzip.rename(columns={"Dataset": "DatasetName",
                              "FPZIP_Ratio": "CompressionRatio"})
fpzip["compression_tool"] = "fpzip"
fpzip["RunType"]          = "standard"
fpzip = fpzip[["DatasetName", "compression_tool", "RunType", "CompressionRatio"]]


# ── 2) Keep only datasets that exist in fpzip (baseline fairness) ───────────
baseline = set(fpzip["DatasetName"])          # fpzip has these datasets
all_dfs  = [               # filter every other tool to that same set
    fpzip,
    zstd   [zstd   ["DatasetName"].isin(baseline)],
    lz4    [lz4    ["DatasetName"].isin(baseline)],
    snappy [snappy ["DatasetName"].isin(baseline)],
    bzip   [bzip   ["DatasetName"].isin(baseline)],
    nvcomp [nvcomp ["DatasetName"].isin(baseline)],
    zlib   [zlib   ["DatasetName"].isin(baseline)],
]


# ── 3) Build wide table, compute CRI = (TDT+tool) / fpzip ───────────────────
df_long = pd.concat(all_dfs, ignore_index=True)
df_long["col_key"] = df_long.apply(
    lambda r: f"TDT+{r.compression_tool}" if r.RunType == "TDT"
              else r.compression_tool,
    axis=1
)

wide = (df_long
        .pivot_table(index="DatasetName",
                     columns="col_key",
                     values="CompressionRatio")
        .dropna(subset=["fpzip"])                 # need fpzip baseline
)

# one CRI column per TDT variant
cri_cols = {}
for col in wide.columns:
    if col.startswith("TDT+"):
        tool = col.replace("TDT+", "")
        cri_cols[f"TDT+{tool} / fpzip"] = wide[col] / wide["fpzip"]

cri = pd.DataFrame(cri_cols, index=wide.index)    # rows = datasets

# ── 3b) Aggregate: geometric mean (+SEM) across datasets ────────────────────
labels, means, errs = [], [], []
for col in cri.columns:
    vals = cri[col].dropna()
    labels.append(col)          # already “TDT+tool / fpzip”
    means.append(gmean(vals))
    errs.append(sem(vals))

# ── 4) Plot (VLDB style) ────────────────────────────────────────────────────
import matplotlib as mpl
sns.set_style("ticks")

# ── VLDB-style rcParams ─────────────────────────────────────────────────────
mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Latin Modern Roman", "Times",
                   "DejaVu Serif"],
    "font.size": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "pdf.fonttype": 42,      # embed TrueType, not Type 3
    "ps.fonttype": 42,
})

PINK = "#e78ac3"
colors = [PINK] * len(means)

fig, ax = plt.subplots(figsize=(6.8, 3.8))   # a bit smaller for 1-column width
y_pos = range(len(labels))

bars = ax.barh(
    y_pos, means, xerr=errs, height=0.48,
    color=colors, capsize=3.0, alpha=0.88
)

ax.set_yticks(y_pos)
ax.set_yticklabels(labels)
ax.set_xlabel(
    r"CRI $=\frac{\mathrm{TDT\,(geometric\ mean)}}{\mathrm{fpzip\,(geometric\ mean)}}$"
)



ax.axvline(1.0, ls="--", lw=0.8, color="black")
ax.set_xlim(0, max(means) * 1.15)
sns.despine(left=True, bottom=False)

# annotate each bar
for bar, m in zip(bars, means):
    ax.text(bar.get_width() + 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{m:.2f}",
            va="center", ha="left", fontsize=14)

fig.tight_layout(pad=0.6)

out_path_pdf = "/home/jamalids/Documents/CRI-TDT_vs_fpzip.pdf"
fig.savefig(out_path_pdf, dpi=300)
print(f"VLDB-formatted figure saved → {out_path_pdf}")
