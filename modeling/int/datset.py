# import kagglehub
# import os
#
# # Desired save location
# save_folder = "/home/jamalids/Documents"
#
#
#
# # Create the directory if it doesn't exist
# os.makedirs(save_folder, exist_ok=True)
#
# # Change the working directory
# os.chdir(save_folder)
#
# # Download the dataset (it will be saved in the current directory)
# path = kagglehub.dataset_download("raddar/amex-data-integer-dtypes-parquet-format")
#
# print("Dataset saved in:", path)
##########################################################################33
# import shutil
# import os
#
# # Your target path
# target_path = "/home/jamalids/Documents/dataset"
# path="/home/jamalids/.cache/kagglehub/datasets/raddar/amex-data-integer-dtypes-parquet-format/versions/2"
# # Make sure target directory exists
# os.makedirs(target_path, exist_ok=True)
#
# # Move all files from the kagglehub cache to /m/doc/dataset
# shutil.copytree(path, target_path, dirs_exist_ok=True)
#
# print("Dataset copied to:", target_path)

####################################
# import pandas as pd
# import pyarrow.parquet as pq
#
# input_path = "/home/jamalids/Documents/dataset/train.parquet"
# output_path = "/home/jamalids/Documents/dataset/train.tsv"
#
# # Open Parquet file
# parquet_file = pq.ParquetFile(input_path)
#
# with open(output_path, 'w') as f_out:
#     for i, batch in enumerate(parquet_file.iter_batches(batch_size=50000)):
#         df_chunk = batch.to_pandas()
#         df_chunk.to_csv(f_out, sep='\t', index=False, header=(i == 0))
#         print(f"Written chunk {i + 1}")
#
# print("✅ Successfully converted to test.tsv")
###################################################
#
# import kagglehub
#
# # Download latest version
# path = kagglehub.dataset_download("tanhim/agricultural-dataset-bangladesh-44-parameters")
#
# print("Path to dataset files:", path)
##################################################33
# import shutil
# import os
# import kagglehub
#
# # Download latest version
# path = kagglehub.dataset_download("raddar/icr-integer-data")
#
# print("Path to dataset files:", path)
#
# print("Path to dataset files:", path)
# # Your target path
# target_path = "/home/jamalids/Documents/dataset"
#
# # Make sure target directory exists
# os.makedirs(target_path, exist_ok=True)
#
# # Move all files from the kagglehub cache to /m/doc/dataset
# shutil.copytree(path, target_path, dirs_exist_ok=True)
#
# print("Dataset copied to:", target_path)
#######################################################
import pandas as pd

df = pd.read_csv("/home/jamalids/Documents/dataset/poker-hand-testing.data",  sep=',')
df.to_csv("/home/jamalids/Documents/dataset/poker-hand.csv")
print(df.dtypes)
