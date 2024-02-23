# Big data for Health

## Ovarian Cancer Data Analysis

### Requirements
- numpy>=1.19.5
- scipy>=1.5.4
### Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```





## Running OC Dataset-350
Assuming dataset is located in `./data` folder, it generates accuracy, time, and memory plots and 
saves them in `./plots` folder:
```bash
python3 analyze_dataset.py ./data/OC_Blood_Routine.csv
```

TODO: add more details like seed number, etc.

## Running on Narval
You will need to modify `run_narval.sh` and put the script you want to execute and then run:
```bash
sbatch run_narval.sh
```

