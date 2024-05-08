import os
import sys


def generate_python_command(datasets, script_name, output_csv_path, num_threads=1):
    script_content = ""
    for dataset in datasets:
        script_content += f"echo 'Processing {dataset}'\n"
        script_content += f"python3 {script_name} --dataset={dataset} --outcsv={output_csv_path} --nthreads={num_threads}\n\n"
    return script_content


# generate the sbatch header for the given number of tasks, name, and time, all inputs are strings
def generate_sbatch_header(time, num_tasks, email, memory="64000M", name="Compression"):
    return f"""#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --output={name}.%j.%N.out
#SBATCH --error={name}.%j.%N.err
#SBATCH --time={time}
#SBATCH --mail-type=begin  # email me when the job starts
#SBATCH --mail-type=end    # email me when the job finishes
#SBATCH --mail-user={email}
#SBATCH --export=ALL
#SBATCH --cpus-per-task={num_tasks}
#SBATCH --nodes=1
#SBATCH --mem={memory}

                """


# generate module loads for the given modules
def generate_module_loads(modules):
    return "\n".join([f"module load {module}\n" for module in modules])


# generate a command line parser
def arg_parser():
    import argparse
    parser = argparse.ArgumentParser(description='Generate a shell script to process datasets.')
    parser.add_argument('--dataset', dest='dataset_path', default="./data/UCRArchive_2018/", help='Path to the UCR dataset directory.')
    parser.add_argument('--pscript', dest='script_name', default="compress_one_dataset.py", help='Name of the script to run on each dataset.')
    parser.add_argument('--outcsv', dest='out_csv_dir', default="./logs/",  help='where csv log files are stored.')
    parser.add_argument('--nsbatch', dest='num_scripts', default=5, type=int, help='Number of scripts to run in parallel.')
    parser.add_argument('--output', dest='output_dir', default="./", help='Output directory for the sbatch scripts.')
    parser.add_argument('--testmode', dest='test_mode', default=False, type=bool, help='Test mode to print the script instead of writing it.')
    parser.add_argument('--nthreads', dest='num_threads', default=1, type=int, help='Number of threads to use.')
    parser.add_argument('--email', dest='email', default="jamalids@mcmaster.ca", help='Email to send the job status.')
    return parser


def partition_by_size_files(file_list, num_partition):
    partitioned_files = []
    # get the size of each file
    file_sizes = [os.path.getsize(file) for file in file_list]
    # sum of all file sizes
    total_size = sum(file_sizes)
    # average size of each partition, 0.1 to account for the last partition
    partition_size = (total_size + 0.00001*total_size) // num_partition
    # current partition size
    current_partition_size = 0
    # current partition
    current_partition = []
    for file, size in zip(file_list, file_sizes):
        if current_partition_size + size <= partition_size:
            current_partition.append(file)
            current_partition_size += size
        else:
            partitioned_files.append(current_partition)
            current_partition = [file]
            current_partition_size = size
    return partitioned_files


if __name__ == "__main__":
    parser = arg_parser()
    args = parser.parse_args()
    ucr_path = args.dataset_path
    script_name = args.script_name
    num_scripts = args.num_scripts
    output_dir = args.output_dir
    test_mode = args.test_mode
    csv_output_dir = args.out_csv_dir
    num_threads = args.num_threads
    email = args.email



    # get the list of all tsv file in ucr_path recursively
    datasets = [os.path.join(dp, f) for dp, dn, filenames in os.walk(ucr_path) for f in filenames if f.endswith('.tsv')]
    dataset_groups = partition_by_size_files(datasets, num_scripts)
    for i, dataset_group in enumerate(dataset_groups):
        sbatch_script_ucr_content = ""
        sbatch_script_ucr_content += generate_sbatch_header("05:00:00", num_threads, "samira@mcmaster.ca")
        sbatch_script_ucr_content += generate_module_loads(["StdEnv", "python/3.10"])

        output_csv_path = f"output_{i}.csv"
        sbatch_script_ucr_content += generate_python_command(dataset_group, script_name, os.path.join(csv_output_dir,output_csv_path))

        # Write the shell script
        sbatch_script_name = f"run_ucr_datasets_{i}.sh"
        if not test_mode:
            with open(os.path.join(output_dir, sbatch_script_name), "w") as shell_script:
                shell_script.write(sbatch_script_ucr_content)
        else:
            print(sbatch_script_ucr_content)

