# nvCOMP Examples & Batch-Run Guide
*(CUDA-accelerated compression demos with a ready-to-launch SLURM script)*


---

## 1  Overview
This repository now contains **only the official nvCOMP ** supplied by NVIDIA **plus a single SLURM batch script** (`bash1.slurm`) that automates running the **`nvcomp_gds`** sample across a directory of `.tsv` datasets .

> **Removed content:** All custom benchmarks and extra tools have been stripped. What remains is the stock nvCOMP sample tree plus the helper batch script.

---

## 2  Directory Structure
```
.
├── examples/                    # nvCOMP stock samples

│   ├── nvcomp_gds.cu            ← GPUDirect-Storage demo
│  │
├── bash1.slurm                 ← ★ Batch script (see § 6)
├── CMakeLists.txt
└── README.md                   ← this file
```

---

## 3  Prerequisites
| Requirement                | Notes                                                    |
|----------------------------|----------------------------------------------------------|
| **CUDA ≥ 12.3**            | `nvcc --version` must report 12.3 or newer               |
| **nvCOMP ≥ 4.1.1.1**       | Download from NVIDIA or install via pkg-manager          |
| **CPU helper libs**        | `zlib`, `lz4`, `libdeflate` headers & libs at build time |

### 3.1  Quick install on Ubuntu
```bash
sudo apt-get update
sudo apt-get install liblz4-dev liblz4-1 \
                     zlib1g-dev zlib1g \
                     libdeflate-dev libdeflate0
```

---

## 4  Building the Examples (Linux)
```bash
git clone <repo-url>
cd nvcomp-examples

mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=<path-to-nvcomp> \
         -DCMAKE_BUILD_TYPE=Release \
         -DBUILD_GDS_EXAMPLE=ON        # builds nvcomp_gds sample
cmake --build . --parallel $(nproc)
```
The GPUDirect sample ends up at:
```
build/nvcomp_gds
```

---

## 5  Quick Manual Test
```bash
# decompress a .gz file via GPUDirect Storage
./build/nvcomp_gds compressed_sample.gz
```
Add `-h` to view all flags.

---

## 6  Batch Processing with **bash1.slurm**
`bash1.slurm` loops over every `.tsv` file in `$DATASET_DIR`, calls **`nvcomp_gds`** with precision 32, and records per-file runtime into `$RESULTS_DIR`.

```
#!/bin/bash
#SBATCH --job-name="gdeflate_cpu_compression"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --constraint=rome
#SBATCH --mem=254000M
#SBATCH --time=47:59:00
#SBATCH --output="gdeflate_cpu_compression.%j.%N.out"
#SBATCH --mail-user=jamalids@mcmaster.ca
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --export=ALL

module load StdEnv/2023
module load gcc/13.3
module load cmake/3.27.7
export LD_LIBRARY_PATH=/usr/local/cuda-12.3/targets/x86_64-linux/lib:$LD_LIBRARY_PATH

# ---------- user paths ----------
DATASET_DIR="/home/jamalids/Documents/2D/data1/Fcbench/Fcbench-dataset/32"
EXECUTABLE="/home/jamalids/nvcomp-examples/build/nvcomp_gds"
RESULTS_DIR="/home/jamalids/Documents/results1"
mkdir -p "$RESULTS_DIR"

echo "=====> Starting nvcomp_gds batch <====="

for dataset in "$DATASET_DIR"/*.tsv; do
  if [ -f "$dataset" ]; then
    base=$(basename "$dataset" .tsv)
    log="$RESULTS_DIR/${base}_run.log"

    echo "Processing $dataset" | tee "$log"
    start=$(date +%s.%N)

    "$EXECUTABLE" "$dataset" 32            # run nvcomp_gds

    end=$(date +%s.%N)
    echo "Execution Time: $(echo "$end - $start" | bc) s" >> "$log"
  fi
done

echo "=====> All datasets processed successfully <====="
```

### 6.1  Submit the job
```bash
sbatch bash1.slurm
```

### 6.2  Things you should edit
| Setting                | Variable / directive inside script |
|------------------------|-------------------------------------|
| Dataset folder         | `DATASET_DIR`                      |
| Output logs folder     | `RESULTS_DIR`                      |
| nvCOMP binary path     | `EXECUTABLE`                       |
| CUDA library path      | `LD_LIBRARY_PATH` export           |
| Wall-time, RAM, CPUs   | `#SBATCH --time`, `--mem`, `--cpus-per-task` |

---

## 7  Outputs
| File                                     | Produced by           | Content                               |
|------------------------------------------|-----------------------|---------------------------------------|
| `results1/<dataset>_run.log`             | `bash1.slurm` loop    | start line + total execution time     |
| `gdeflate_cpu_compression.<job>.<node>.out` | Slurm stdout/stderr   | whole-job console output              |

---

## 9  License
nvCOMP and its examples are © NVIDIA Corporation, distributed under the license in the original nvCOMP package.  
The batch script (`bash1.slurm`) and this README are released under the MIT License.