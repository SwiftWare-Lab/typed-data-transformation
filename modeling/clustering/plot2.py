# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
#
#
# def main():
#     # File path for combine_file.
#     combine_file = '/home/jamalids/Documents/all-config-cluster/logs-zstd/combine'
#
#     # Load the combine CSV file.
#     df_combine = pd.read_csv(combine_file)
#
#     # Expected columns in combine_file:
#     # 'dataset name', 'standard zstd compression ratio', 'decomposed zstd compression ratio', 'decomposition'
#
#     # Debug: Check how many clustering possibilities each dataset has.
#     counts = df_combine.groupby('dataset name').size()
#     print("Number of possibilities per dataset:")
#     print(counts)
#
#     # Compute percentage improvement:
#     # Improvement (%) = ((decomposed zstd compression ratio - standard zstd compression ratio) / standard zstd compression ratio) * 100
#     df_combine['improvement'] = ((df_combine['decomposed zstd compression ratio'] - df_combine[
#         'standard zstd compression ratio']) /
#                                  df_combine['standard zstd compression ratio']) * 100
#
#     # Assign each clustering possibility an index (1..82) per dataset.
#     df_combine['possibility_index'] = df_combine.groupby('dataset name').cumcount() + 1
#
#     # Plotting: one line per dataset.
#     plt.figure(figsize=(14, 8))
#
#     # Get unique datasets and set up a colormap.
#     datasets = df_combine['dataset name'].unique()
#     cmap = plt.get_cmap('tab10', len(datasets))
#
#     for i, ds in enumerate(datasets):
#         df_ds = df_combine[df_combine['dataset name'] == ds].copy()
#         df_ds.sort_values('possibility_index', inplace=True)
#
#         x = df_ds['possibility_index']
#         y = df_ds['improvement']
#         color = cmap(i)
#
#         # Plot the line with markers.
#         plt.plot(x, y, marker='o', linestyle='-', color=color, linewidth=2, label=ds)
#
#         # Set the x-ticks using the 'decomposition' values for this dataset.
#         # To avoid overcrowding, we sample the tick labels (e.g. every 10th possibility).
#         step = max(1, len(x) // 10)
#         xticks = x.iloc[::step]
#         xticklabels = df_ds['decomposition'].iloc[::step]
#         plt.xticks(xticks, xticklabels, rotation=45, ha='right')
#
#     plt.xlabel("Clustering Possibility (Decomposition)")
#     plt.ylabel("Improvement (%)")
#     plt.title("Compression Ratio Improvement: Decomposed vs. Standard Zstd (Combine File)")
#     plt.legend(title='Dataset', loc='best')
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig("lineplot_improvement_by_decomposition_combine.png", dpi=300)
#     plt.show()
#
#
# if __name__ == '__main__':
#     main()
# # import os
# # import pandas as pd
# # import numpy as np
# # import matplotlib.pyplot as plt
# #
# #
# # def process_csv(file_path, out_path):
# #     # Load the CSV file.
# #     df = pd.read_csv(file_path)
# #
# #     # Define the required columns.
# #     required_columns = [
# #         'dataset name',
# #         'standard zstd compression ratio',
# #         'decomposed zstd compression ratio',
# #         'decomposition'
# #     ]
# #
# #     # Check if all required columns exist.
# #     missing = [col for col in required_columns if col not in df.columns]
# #     if missing:
# #         print(f"Skipping {os.path.basename(file_path)}: missing columns {missing}")
# #         return
# #
# #     # Debug: Print number of possibilities per dataset.
# #     counts = df.groupby('dataset name').size()
# #     print(f"Processing {os.path.basename(file_path)} - Possibilities per dataset:")
# #     print(counts)
# #
# #     # Compute percentage improvement:
# #     # Improvement (%) = ((decomposed zstd compression ratio - standard zstd compression ratio) / standard zstd compression ratio) * 100
# #     df['improvement'] = ((df['decomposed zstd compression ratio'] -
# #                           df['standard zstd compression ratio']) /
# #                          df['standard zstd compression ratio']) * 100
# #
# #     # Assign each clustering possibility an index (1..82) per dataset.
# #     df['possibility_index'] = df.groupby('dataset name').cumcount() + 1
# #
# #     # Create the plot.
# #     plt.figure(figsize=(14, 8))
# #
# #     # Get unique datasets and set up a colormap.
# #     datasets = df['dataset name'].unique()
# #     cmap = plt.get_cmap('tab10', len(datasets))
# #
# #     for i, ds in enumerate(datasets):
# #         df_ds = df[df['dataset name'] == ds].copy()
# #         df_ds.sort_values('possibility_index', inplace=True)
# #
# #         x = df_ds['possibility_index']  # Numeric index from 1 to 82.
# #         y = df_ds['improvement']
# #         color = cmap(i)
# #
# #         # Plot the line with markers.
# #         plt.plot(x, y, marker='o', linestyle='-', color=color, linewidth=2, label=ds)
# #         plt.fill_between(x, 0, y, color=color, alpha=0.15)
# #
# #     # Set x-axis ticks as numeric values.
# #     max_index = int(df['possibility_index'].max())
# #     plt.xticks(np.arange(1, max_index + 1, step=5))
# #
# #     plt.xlabel("Clustering Possibility Index (1..82)")
# #     plt.ylabel("Improvement (%)")
# #     plt.title("Compression Ratio Improvement: Clustered vs. Standard Zstd")
# #     plt.legend(title='Dataset', loc='best')
# #     plt.grid(True)
# #     plt.tight_layout()
# #     plt.savefig(out_path, dpi=300)
# #     plt.close()  # Close the figure to free memory.
# #
# #
# # def main():
# #     # Folder containing the CSV files.
# #     folder = '/home/jamalids/Documents/all-config-cluster/logs-zstd/'
# #
# #     # Process each CSV file in the folder.
# #     for file in os.listdir(folder):
# #         if file.lower().endswith('.csv'):
# #             file_path = os.path.join(folder, file)
# #             out_file = os.path.splitext(file)[0] + ".png"
# #             out_path = os.path.join(folder, out_file)
# #             print(f"Processing {file_path}...")
# #             process_csv(file_path, out_path)
# #             print(f"Saved plot to {out_path}")
# #
# #
# # if __name__ == '__main__':
# #     main()
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def process_and_plot(folder):
    # List to hold DataFrames from each CSV file.
    df_list = []

    # Loop over all files in the folder.
    for file in os.listdir(folder):
        if file.lower().endswith('.csv'):
            file_path = os.path.join(folder, file)
            try:
                df_temp = pd.read_csv(file_path)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue
            # Check for required columns.
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

    # Combine all DataFrames.
    df_combine = pd.concat(df_list, ignore_index=True)

    # Debug: Print number of possibilities per dataset.
    counts = df_combine.groupby('dataset name').size()
    print("Number of possibilities per dataset:")
    print(counts)

    # Compute percentage improvement:
    # Improvement (%) = ((decomposed zstd compression ratio - standard zstd compression ratio) /
    #                      standard zstd compression ratio) * 100
    df_combine['improvement'] = (
                                        (df_combine['decomposed zstd compression ratio'] - df_combine[
                                            'standard zstd compression ratio']) /
                                        df_combine['standard zstd compression ratio']
                                ) * 100

    # Assign each clustering possibility an index (1..82) per dataset.
    df_combine['possibility_index'] = df_combine.groupby('dataset name').cumcount() + 1

    # Plotting: one line per dataset.
    plt.figure(figsize=(14, 8))

    # Get unique datasets and set up a colormap.
    datasets = df_combine['dataset name'].unique()
    cmap = plt.get_cmap('tab10', len(datasets))

    for i, ds in enumerate(datasets):
        df_ds = df_combine[df_combine['dataset name'] == ds].copy()
        df_ds.sort_values('possibility_index', inplace=True)

        x = df_ds['possibility_index']  # Numeric index (1..82).
        y = df_ds['improvement']
        color = cmap(i)

        # Plot the line with markers.
        plt.plot(x, y, marker='o', linestyle='-', color=color, linewidth=2, label=ds)
        plt.fill_between(x, 0, y, color=color, alpha=0.15)

    # Set x-axis ticks as numeric values.
    max_index = int(df_combine['possibility_index'].max())
    plt.xticks(np.arange(1, max_index + 1, step=5))

    plt.xlabel("Clustering Possibility Index (1..82)")
    plt.ylabel("Improvement (%)")
    plt.title("Compression Ratio Improvement: Clustered vs. Standard Zstd")
    plt.legend(title='Dataset', loc='best')
    plt.grid(True)
    plt.tight_layout()

    # Save the plot in the same folder.
    out_path = os.path.join(folder, "lineplot_improvement_by_index_combine.png")
    plt.savefig(out_path, dpi=300)
    plt.show()
    print(f"Plot saved to {out_path}")


def main():
    # Folder that contains the CSV files to combine.
    folder = '/home/jamalids/Documents/all-config-cluster/logs-zstd/combine'
    process_and_plot(folder)


if __name__ == '__main__':
    main()
