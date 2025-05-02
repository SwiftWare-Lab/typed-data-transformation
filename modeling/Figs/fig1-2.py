import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# VLDB-style formatting
mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Latin Modern Roman", "Times", "DejaVu Serif"],
    "font.size": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def process_and_plot(folder):
    df_list = []

    for file in os.listdir(folder):
        if file.lower().endswith('.csv'):
            file_path = os.path.join(folder, file)
            try:
                df_temp = pd.read_csv(file_path)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue

            required_columns = [
                'dataset name',
                'standard zstd compression ratio',
                'decomposed zstd compression ratio',
                'decomposition'
            ]
            missing = [col for col in required_columns if col not in df_temp.columns]
            if missing:
                print(f"Skipping {file}: missing columns {missing}")
                continue

            df_list.append(df_temp)

    if not df_list:
        print("No valid CSV files found.")
        return

    df_combine = pd.concat(df_list, ignore_index=True)

    print("Number of possibilities per dataset:")
    print(df_combine.groupby('dataset name').size())

    # Compute improvement percentage
    df_combine['improvement'] = (
        (df_combine['decomposed zstd compression ratio'] - df_combine['standard zstd compression ratio']) /
        df_combine['standard zstd compression ratio']
    ) * 100

    # Assign index to each clustering possibility
    df_combine['possibility_index'] = df_combine.groupby('dataset name').cumcount() + 1

    # VLDB-sized plot
    plt.figure(figsize=(7, 4))

    datasets = df_combine['dataset name'].unique()
    cmap = plt.get_cmap('tab10', len(datasets))

    for i, ds in enumerate(datasets):
        df_ds = df_combine[df_combine['dataset name'] == ds].copy()
        df_ds.sort_values('possibility_index', inplace=True)

        x = df_ds['possibility_index']
        y = df_ds['improvement']
        color = cmap(i)

        plt.plot(x, y, marker='o', linestyle='-', color=color, linewidth=1, label=ds)
        plt.fill_between(x, 0, y, color=color, alpha=0.15)

    max_index = int(df_combine['possibility_index'].max())
    plt.xticks(np.arange(1, max_index + 1, step=5))

    plt.xlabel("Clustering Possibility Index")
    plt.ylabel("Compression Ratio Improvement (%)")
   # plt.legend(title='Dataset', loc='best')
   # plt.grid(True)
    plt.tight_layout()

    out_path = os.path.join(folder, "lineplot_improvement_by_index_combine.pdf")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {out_path}")


def main():
    folder = "/mnt/c/Users/jamalids/Downloads/fig1"
    process_and_plot(folder)


if __name__ == '__main__':
    main()
