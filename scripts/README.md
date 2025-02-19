# Scripts

## Installation
To install the required packages, run the following command:
```bash
venv -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The above setup should be done before running on the server.

## generate_script.py
This script generates sbatch files for running the simulation on the cluster. See the args for more information.

## compress_one_dataset.py
This script compresses the data for one dataset and generates a CSV log. See the args for more information.