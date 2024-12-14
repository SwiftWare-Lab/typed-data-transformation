import numpy as np
import pandas as pd
import os
import zstandard as zstd
from numpy.lib.stride_tricks import as_strided
from dataclasses import dataclass

@dataclass
class Token:
    offset: int
    length: int
    indicator: str

    def __repr__(self) -> str:
        return f"({self.offset}, {self.length}, {self.indicator})"

from dataclasses import dataclass

@dataclass
class Token:
    offset: int
    length: int
    indicator: str

    def __repr__(self) -> str:
        return f"({self.offset}, {self.length}, {self.indicator})"

class LZ77Compressor:
    def __init__(self, window_size: int = 20, lookahead_buffer_size: int =15):
        self.window_size = window_size
        self.lookahead_buffer_size = lookahead_buffer_size
        self.search_buffer_size = self.window_size - self.lookahead_buffer_size

    def compress(self, text: str) -> list[Token]:
        output = []
        search_buffer = ""
        while text:
            token = self._find_encoding_token(text, search_buffer)
            search_buffer += text[: token.length + 1]
            if len(search_buffer) > self.search_buffer_size:
                search_buffer = search_buffer[-self.search_buffer_size:]
            text = text[token.length + 1:]
            output.append(token)
        return output

    def decompress(self, tokens: list[Token]) -> str:
        output = ""
        for token in tokens:
            if token.length > 0 and token.offset > 0:
                start = len(output) - token.offset
                for i in range(token.length):
                    if start + i < len(output):
                        output += output[start + i]
            output += token.indicator
        return output

    def _find_encoding_token(self, text: str, search_buffer: str) -> Token:
        if not text:
            raise ValueError("We need some text to work with.")

        length, offset = 0, 0
        indicator = text[0] if text else ''  # Use the first character as the indicator or empty if text is empty

        if not search_buffer:
            return Token(offset, length, indicator)

        for i in range(len(search_buffer)):
            if search_buffer[i] == indicator:
                found_length = self._match_length_from_index(text, search_buffer, 0, i)
                found_offset = len(search_buffer) - i
                if found_length > length:
                    offset, length = found_offset, found_length

        if length < len(text):  # Ensure the indicator index does not go out of bounds
            indicator = text[length]
        else:  # Prevent out of bounds by setting indicator to an empty string if at the end
            indicator = ''

        return Token(offset, length, indicator)

    def _match_length_from_index(self, text: str, window: str, text_index: int, window_index: int) -> int:
        max_length = 0
        while text_index < len(text) and window_index < len(window) and text[text_index] == window[window_index]:
            max_length += 1
            text_index += 1
            window_index += 1
        return max_length


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

def lz77_compress_component_data(data):
    lz77 = LZ77Compressor()
    text = data.tobytes().decode('latin1')  # Ensure safe encoding conversion
    compressed_tokens = lz77.compress(text)
    compressed_text = lz77.decompress(compressed_tokens)
    return compressed_tokens, len(text) / len(compressed_text)
def lz77_compress_component_data_full(data):
    lz77 = LZ77Compressor()
    # Directly decode the data from bytes to string
    text = data.decode('latin1')  # Ensure safe encoding conversion
    compressed_tokens = lz77.compress(text)
    compressed_text = lz77.decompress(compressed_tokens)
    return compressed_tokens, len(text) / len(compressed_text)

def decomposition_based_compression(byte_array, component_sizes):
    components = split_bytes_into_components(byte_array, component_sizes)
    for i, comp in enumerate(components):
        lz77_tokens, lz77_ratio = lz77_compress_component_data(comp)
        print(f"Component {i + 1} LZ77 compression ratio: {lz77_ratio}")
    # full
    full_text = byte_array.decode('latin1')  # This line should directly decode byte_array
    full_tokens, full_ratio = lz77_compress_component_data_full(byte_array)
    print(f"Overall LZ77 compression ratio for full data: {full_ratio}")


def run_and_collect_data():
    dataset_path = "/home/jamalids/Documents/2D/data1/Fcbench/Low-Entropy/32/citytemp_f32.tsv"
    datasets = [dataset_path]
    for dataset_path in datasets:
        ts_data1 = pd.read_csv(dataset_path, delimiter='\t', header=None)
        dataset_name = os.path.basename(dataset_path).replace('.tsv', '')
        print(f"Processing dataset: {dataset_name}")
        group = ts_data1.drop(ts_data1.columns[0], axis=1).astype(float).T
        group = group.iloc[:, 0:1000000]
        byte_array = group.to_numpy().astype(np.float32).tobytes()
        component_sizes = [2, 1, 1]  # Example sizes
        decomposition_based_compression(byte_array, component_sizes)

if __name__ == "__main__":
    run_and_collect_data()
