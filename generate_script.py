import os
import sys

def generate_shell_script(ucr_path, script_name):
    datasets = [f for f in os.listdir(ucr_path) if f.endswith('.csv')]
    script_content = ""

    for dataset in datasets:
        dataset_path = os.path.join(ucr_path, dataset)
        script_content += f"python {script_name} {dataset_path}\n"

    # Write the shell script
    with open("run_datasets.sh", "w") as shell_script:
        shell_script.write(script_content)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python generate_script.py <UCR_path> <script_name>")
        sys.exit(1)

    ucr_path = sys.argv[1]
    script_name = sys.argv[2]
    generate_shell_script(ucr_path, script_name)

