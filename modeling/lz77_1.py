import numpy as np
import pandas as pd
import os
from numpy.lib.stride_tricks import as_strided


class LZ77Compressor:
    def __init__(self, window_size=20, lookahead_buffer_size=15):
        self.window_size = window_size
        self.lookahead_buffer_size = lookahead_buffer_size

    def compress(self, text):
        i = 0
        output = []
        while i < len(text):
            window_start = max(0, i - self.window_size)
            window = text[window_start:i]
            match_len = 0
            match_pos = 0
            buffer = text[i:i + self.lookahead_buffer_size]

            # Find the longest match in the window
            for j in range(len(window)):
                k = 0
                while k < len(buffer) and j + k < len(window) and buffer[k] == window[j + k]:
                    k += 1
                if k > match_len:
                    match_len = k
                    match_pos = i - (window_start + j)

            # Ensuring the next character after match is within bounds
            next_char = buffer[match_len] if i + match_len < len(text) and match_len < len(buffer) else ''

            if match_len > 0:
                output.append((match_pos, match_len, next_char))
                i += match_len + 1
            else:
                output.append((0, 0, text[i]))
                i += 1

        return output



def split_bytes_into_components(byte_array, component_sizes):
    byte_array = np.frombuffer(byte_array, dtype=np.uint8)
    total_bytes = sum(component_sizes)
    num_elements = len(byte_array) // total_bytes
    components = []
    offset = 0
    for size in component_sizes:
        component_view = as_strided(byte_array[offset:], shape=(num_elements, size), strides=(total_bytes, 1))
        components.append(component_view.flatten())
        offset += size
    return components

def calculate_compression_ratio(original, compressed):
    original_size = len(original)
    compressed_size = sum(len(c) for c in compressed)  # Assuming compressed is a list of compressed data pieces
    return original_size, compressed_size  # Return both sizes for further calculations

def run_and_collect_data():
    import numpy as np
    import pandas as pd
    import os

    dataset_path = "/home/jamalids/Documents/2D/data1/Fcbench/Low-Entropy/32/citytemp_f32.tsv"
    ts_data1 = pd.read_csv(dataset_path, delimiter='\t', header=None)
    dataset_name = os.path.basename(dataset_path).replace('.tsv', '')
    print(f"Processing dataset: {dataset_name}")

    group = ts_data1.drop(ts_data1.columns[0], axis=1).astype(float).T
    byte_array = group.to_numpy().astype(np.float32).tobytes()
    component_sizes = [2, 1, 1]  # Example sizes in bytes, adjust based on your actual component structure

    components = split_bytes_into_components(byte_array, component_sizes)

    compressor = LZ77Compressor(window_size=100, lookahead_buffer_size=50)
    total_compressed_size = 0
    total_original_size = 0

    # Compress each component
    for i, component in enumerate(components):
        if isinstance(component, np.ndarray):
            component = component.tobytes()
        decoded_component = component.decode('latin-1')
        compressed_component = compressor.compress(decoded_component)
        original_size, compressed_size = calculate_compression_ratio(decoded_component, [compressed_component])
        total_compressed_size += compressed_size
        total_original_size += original_size
        compression_ratio = original_size / compressed_size if compressed_size else 1
        print(f"Component {i + 1} LZ77 compression size: {compressed_size} bytes, Ratio: {compression_ratio:.2f}")

    # Overall compression ratio for decomposed data
    decomposed_ratio = total_original_size / total_compressed_size if total_compressed_size else 1
    print(f"Decomposed data overall compression ratio: {decomposed_ratio:.2f}")

    # Compress the whole array
    decoded_full_array = byte_array.decode('latin-1')  # Ensure byte_array is correctly decoded
    compressed_full = compressor.compress(decoded_full_array)
    full_original_size, full_compressed_size = calculate_compression_ratio(decoded_full_array, [compressed_full])
    full_ratio = full_original_size / full_compressed_size if full_compressed_size else 1
    print(f"Overall LZ77 compression for full data: {full_compressed_size} bytes, Ratio: {full_ratio:.2f}")

if __name__ == "__main__":
    run_and_collect_data()
