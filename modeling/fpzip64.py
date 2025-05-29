import os
import tempfile
import pandas as pd
import numpy as np
import fpzip  # pip install fpzip

# -----------------------------------------------------------------------------
# FPZIP Compression Helper (in-memory via Python binding)
# -----------------------------------------------------------------------------
def fpzip_comp(raw_path: str, bits: int = 32, precision: int = 0, order: str = 'C') -> float:
    """
    Compress `raw_path` (a binary float32 or float64 file) losslessly with
    fpzip Python binding and return original_size / compressed_size.

    Args:
        raw_path:   Path to a .f32 or .f64 file containing binary floats.
        bits:       32 or 64 for single- or double-precision.
        precision:  Number of least-significant mantissa bits to discard (0 = lossless).
        order:      Memory order for compression ('C' or 'F').
    """
    if bits not in (32, 64):
        raise ValueError("`bits` must be 32 or 64")
    dtype = np.float32 if bits == 32 else np.float64

    arr = np.fromfile(raw_path, dtype=dtype)
    orig_size = arr.nbytes
    if orig_size == 0:
        raise ValueError(f"{raw_path!r} is empty")

    compressed_bytes = fpzip.compress(arr, precision=precision, order=order)
    comp_size = len(compressed_bytes)
    if comp_size == 0:
        raise RuntimeError("fpzip.compress returned empty output")

    return orig_size / comp_size


# -----------------------------------------------------------------------------
# Analysis: measure FPZIP ratios for all .tsv files in a folder
# -----------------------------------------------------------------------------
def run_analysis_fpzip(folder_path: str,
                       output_csv: str,
                       bits: int = 32,
                       precision: int = 0,
                       order: str = 'C'):
    """
    For each .tsv in `folder_path`:
      1. Load column 1 as float32 or float64
      2. Dump to a temp .f32/.f64 file
      3. Compress via fpzip.compress()
      4. Record ratio and TSV path
      5. Save all results to `output_csv`

    Args:
        folder_path:  Directory containing .tsv files.
        output_csv:   Path to write results CSV.
        bits:         32 or 64 for dtype.
        precision:    Mantissa bits to discard (0 = lossless).
        order:        Memory order for compression ('C' or 'F').
    """
    if bits not in (32, 64):
        raise ValueError("`bits` must be 32 or 64")
    dtype = np.float32 if bits == 32 else np.float64
    suffix = '.f32' if bits == 32 else '.f64'

    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Invalid folder: {folder_path}")

    records = []
    for fname in sorted(os.listdir(folder_path)):
        if not fname.lower().endswith('.tsv'):
            continue
        dataset = os.path.splitext(fname)[0]
        tsv_path = os.path.join(folder_path, fname)
        print("Processing", dataset)

        # Load TSV (retry Python engine on failure)
        try:
            df = pd.read_csv(tsv_path, sep='\t', header=None)
        except Exception:
            df = pd.read_csv(tsv_path, sep='\t', header=None, engine='python')

        # Extract column 1 and cast to the right dtype
        arr = df.iloc[:, 1].astype(dtype).values

        # Write raw bytes to temp file with correct suffix
        tmp_path = os.path.join(tempfile.gettempdir(), f"{dataset}{suffix}")
        arr.tofile(tmp_path)

        try:
            ratio = fpzip_comp(tmp_path, bits=bits, precision=precision, order=order)
        except Exception as e:
            print(f"  ↳ fpzip failed on {dataset}: {e}")
            ratio = None
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

        records.append({
            "Dataset":     dataset,
            "TSV_Path":    tsv_path,
            "FPZIP_Ratio": ratio
        })

    pd.DataFrame(records).to_csv(output_csv, index=False)
    print("Done. Results in", output_csv)


if __name__ == "__main__":
    # Example: run for 64-bit floats
    folder = '/home/jamalids/Documents/2D/data1/Fcbench/dataset/Low-Entropy'
    output = "/home/jamalids/Documents/fpzip_astro.csv"
    run_analysis_fpzip(folder, output, bits=64)
