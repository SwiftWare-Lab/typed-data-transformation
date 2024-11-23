import os
import argparse


def generate_cpp_command(datasets, executable_name, output_csv_path, num_threads=1):
    script_content = ""
    for dataset in datasets:
        script_content += f"echo 'Processing {dataset}'\n"
        dataset_name = dataset.split("/")[-1].split(".")[0]
        output_csv_name = f"output_{dataset_name}.csv"  # Name for the output CSV file
        # Adjust the command to your executable's expected parameters
        script_content += f"./{executable_name} --dataset \"{dataset}\" --outcsv \"{os.path.join(output_csv_path, output_csv_name)}\" --threads {num_threads}\n\n"
    return script_content


def generate_sbatch_header(time, num_tasks, email, memory="64000M", name="Compression"):
    return f"""#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --output={name}.%j.%N.out
#SBATCH --error={name}.%j.%N.err
#SBATCH --time={time}
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user={email}
#SBATCH --export=ALL
#SBATCH --cpus-per-task={num_tasks}
#SBATCH --nodes=1
#SBATCH --mem={memory}
#SBATCH --cpus-per-task=64
#SBATCH --constraint=rome
"""


def generate_module_loads(modules):
    return "\n".join([f"module load {module}" for module in modules]) + "\n"


def arg_parser():
    parser = argparse.ArgumentParser(description='Generate a shell script to process datasets using a C++ executable.')
    parser.add_argument('--dataset', dest='dataset_path', required=True, help='Path to the dataset directory.')
    parser.add_argument('--pscript', dest='executable_name', required=True, help='Path to the C++ executable.')
    parser.add_argument('--outcsv', dest='output_csv_dir', required=True,
                        help='Directory where output CSV files are stored.')
    parser.add_argument('--nsbatch', dest='num_scripts', type=int, default=5,
                        help='Number of scripts to run in parallel.')
    parser.add_argument('--output', dest='output_dir', default="./", help='Output directory for the sbatch scripts.')
    parser.add_argument('--testmode', dest='test_mode', action='store_true',
                        help='Test mode to print the script instead of writing it.')
    parser.add_argument('--nthreads', dest='num_threads', type=int, default=10, help='Number of threads to use.')
    parser.add_argument('--email', dest='email', default="jamalids@mcmaster.ca",
                        help='Email to send the job status notifications.')
    return parser


def partition_datasets_by_size(file_list, num_partition):
    partitioned_files = []
    file_sizes = [os.path.getsize(file) for file in file_list]
    total_size = sum(file_sizes)
    partition_size = total_size // num_partition
    current_partition_size = 0
    current_partition = []

    for file, size in zip(file_list, file_sizes):
        if current_partition_size + size <= partition_size:
            current_partition.append(file)
            current_partition_size += size
        else:
            if current_partition:
                partitioned_files.append(current_partition)
            current_partition = [file]
            current_partition_size = size
    if current_partition:
        partitioned_files.append(current_partition)
    return partitioned_files


if __name__ == "__main__":
    args = arg_parser().parse_args()

    datasets = [os.path.join(dp, f) for dp, dn, filenames in os.walk(args.dataset_path) for f in filenames if
                f.endswith('.tsv')]
    dataset_groups = partition_datasets_by_size(datasets, args.num_scripts)

    for i, dataset_group in enumerate(dataset_groups):
        sbatch_script_content = generate_sbatch_header("05:00:00", str(args.num_threads), args.email)
        sbatch_script_content += generate_module_loads(["StdEnv", "gcc/10.2.0"])  # Example modules
        sbatch_script_content += generate_cpp_command(dataset_group, args.executable_name, args.output_csv_dir,
                                                      args.num_threads)

        sbatch_script_name = f"run_datasets_{i}.sh"
        if args.test_mode:
            print(sbatch_script_content)
        else:
            with open(os.path.join(args.output_dir, sbatch_script_name), 'w') as file:
                file.write(sbatch_script_content)
