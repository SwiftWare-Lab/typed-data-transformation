import os  # Ensure this import is present
import pandas as pd
import matplotlib

# Choose a backend: 'Agg' for non-interactive, 'TkAgg' or 'Qt5Agg' for interactive
# Uncomment one of the following lines based on your preference

# Option A: Non-interactive backend (suitable for saving plots to files)
matplotlib.use('Agg')

# Option B: Interactive backend (ensure dependencies are installed)
# matplotlib.use('TkAgg')  # or 'Qt5Agg'

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import re

# Define a mapping from verbose method names to concise labels
label_mapping = {
    'decomposed huffman_compress compressed size (B)': 'Decomposed Huffman',
    'reordered huffman_compress compressed size (B)': 'Reordered Huffman',
    'standard huffman_compress compressed size (B)': 'Standard Huffman',

    'decomposed fastlz compressed size (B)': 'Decomposed FastLZ',
    'reordered fastlz compressed size (B)': 'Reordered FastLZ',
    'standard fastlz compressed size (B)': 'Standard FastLZ',

    'decomposed rle compressed size (B)': 'Decomposed RLE',
    'reordered rle compressed size (B)': 'Reordered RLE',
    'standard rle compressed size (B)': 'Standard RLE',

    # Added mappings for additional compression methods
    'decomposed zstd compressed size (B)': 'Decomposed Zstd',
    'reordered zstd compressed size (B)': 'Reordered Zstd',
    'standard zstd compressed size (B)': 'Standard Zstd',

    'decomposed zlib compressed size (B)': 'Decomposed Zlib',
    'reordered zlib compressed size (B)': 'Reordered Zlib',
    'standard zlib compressed size (B)': 'Standard Zlib',



    'decomposed snappy compressed size (B)': 'Decomposed Snappy',
    'reordered snappy compressed size (B)': 'Reordered Snappy',
    'standard snappy compressed size (B)': 'Standard Snappy',
}


def shorten_label(method_name):
    """Remove ' compressed size (B)' suffix from method names."""
    suffix = ' compressed size (B)'
    if method_name.endswith(suffix):
        return method_name[:-len(suffix)]
    return method_name


def load_data(file_path):
    """Load and return the data from the CSV file."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist. Please check the path.")
    data = pd.read_csv(file_path)
    return data


def calculate_compression_ratios(data, compression_methods, original_size_column):
    """Calculate and add compression ratios to the DataFrame."""
    original_size = data[original_size_column].iloc[0]
    for category, methods in compression_methods.items():
        for method in methods:
            if method not in data.columns:
                raise KeyError(f"Column '{method}' not found in the CSV file.")
            # Prevent division by zero
            if data[method].iloc[0] == 0:
                raise ValueError(f"Compression size for '{method}' is zero, cannot compute ratio.")
            data[method + '_ratio'] = original_size / data[method]
    return data


def plot_compression_ratios(data, compression_methods, label_mapping, colors, plot_filename):
    """Generate and save the compression ratios plot."""
    num_methods = len(compression_methods)
    fig, axes = plt.subplots(nrows=len(compression_methods),
                             figsize=(15, 8 * len(compression_methods)))  # Adjust height as needed
    axes = axes.flatten()

    for ax, (category, methods) in zip(axes, compression_methods.items()):
        # Extracting ratio data for each method
        ratios = [data[method + '_ratio'].iloc[0] for method in methods]  # Assumes single data point for simplicity

        # Generate shorter labels using the mapping dictionary
        short_labels = []
        for method in methods:
            if method in label_mapping:
                short_labels.append(label_mapping[method])
            else:
                # If the method is not in the mapping, use a default shortened label
                # For example, remove ' compressed size (B)' suffix
                suffix = ' compressed size (B)'
                if method.endswith(suffix):
                    short_label = method[:-len(suffix)]
                else:
                    short_label = method  # Use the original name if no suffix
                short_labels.append(short_label)
                print(f"Warning: Method '{method}' not found in label_mapping. Using default label '{short_label}'.")

        # Create a list of colors: red for 'Decomposed' methods, others from the palette
        colors_for_bars = []
        for label in short_labels:
            if label.startswith('Decomposed'):
                colors_for_bars.append('red')
            else:
                # Assign colors cyclically from the palette if there are more methods than colors
                color_index = short_labels.index(label) % len(colors)
                colors_for_bars.append(colors[color_index])

        # Create bar plot with consistent colors and thinner bars
        ax.bar(range(len(methods)), ratios, color=colors_for_bars, width=0.6)  # Reduced bar width for thinner bars

        # Set title and labels with increased font sizes
        ax.set_title(f'{category} Compression Ratios', fontsize=16)
        ax.set_xlabel('Methods', fontsize=14)
        ax.set_ylabel('Compression Ratio', fontsize=14)

        # Set tick positions and labels
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(labels=short_labels, rotation=45, ha='right', fontsize=12)  # Increased font size

        # Add gridlines for better readability
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)

        # Annotate bars with ratio values
        for i, ratio in enumerate(ratios):
            ax.text(
                i, ratio + max(ratios) * 0.01, f"{ratio:.2f}",
                ha='center', va='bottom', fontsize=10  # Increased font size for annotations
            )

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)  # Adjust vertical spacing if necessary

    # Save the plot
    plt.savefig(plot_filename)  # e.g., 'compression_ratios_existing.png'
    plt.close()


def plot_entropy(byte_entropies, plot_filename):
    """Generate and save the entropy plot."""
    fig, ax = plt.subplots(figsize=(10, 6))  # Adjust size as needed

    # Byte labels
    byte_labels = ['Byte 1', 'Byte 2', 'Byte 3', 'Byte 4']

    # Entropy values
    entropy_values = byte_entropies.values

    # Create bar plot
    ax.bar(range(len(byte_labels)), entropy_values, color='blue', width=0.6)

    # Set title and labels with increased font sizes
    ax.set_title('Entropy of Each Byte', fontsize=16)
    ax.set_xlabel('Bytes', fontsize=14)
    ax.set_ylabel('Entropy', fontsize=14)

    # Set tick positions and labels
    ax.set_xticks(range(len(byte_labels)))
    ax.set_xticklabels(labels=byte_labels, fontsize=12)

    # Add gridlines for better readability
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Annotate bars with entropy values
    for i, entropy in enumerate(entropy_values):
        ax.text(
            i, entropy + max(entropy_values) * 0.01, f"{entropy:.2f}",
            ha='center', va='bottom', fontsize=10  # Increased font size for annotations
        )

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the plot
    plt.savefig(plot_filename)  # e.g., 'byte_entropies.png'
    plt.close()


def main():
    # Load the data from the CSV file
    file_path = "/home/jamalids/Documents/2D/CR-Ct-DT/python-results/logs/turbulence_f32_decomposition_stats.csv"  # Update this path
    data = load_data(file_path)

    # Debugging: Print column names and first few rows
    print("Columns in the CSV file:")
    print(data.columns.tolist())
    print("\nFirst few rows of the CSV file:")
    print(data.head())

    # Define the actual column name for original size
    original_size_column = 'original size'  # Update this based on your CSV

    if original_size_column not in data.columns:
        raise KeyError(f"Column '{original_size_column}' not found in the CSV file. Please check the column names.")

    original_size = data[original_size_column].iloc[0]
    print(f"Original Size: {original_size}")

    # Define method categories based on your compression methods and their configurations
    compression_methods_existing = {
        'Huffman': [
            'decomposed huffman_compress compressed size (B)',
            'reordered huffman_compress compressed size (B)',
            'standard huffman_compress compressed size (B)'
        ],

        'FastLZ': [
            'decomposed fastlz compressed size (B)',
            'reordered fastlz compressed size (B)',
            'standard fastlz compressed size (B)'
        ],

        'RLE': [
            'decomposed rle compressed size (B)',
            'reordered rle compressed size (B)',
            'standard rle compressed size (B)'
        ]
    }

    # New compression methods
    compression_methods_new = {
        'Zstd': [
            'decomposed zstd compressed size (B)',
            'reordered zstd compressed size (B)',
            'standard zstd compressed size (B)'
        ],
        'Zlib': [
            'decomposed zlib compressed size (B)',
            'reordered zlib compressed size (B)',
            'standard zlib compressed size (B)'
        ],


        'Snappy': [
            'decomposed snappy compressed size (B)',
            'reordered snappy compressed size (B)',
            'standard snappy compressed size (B)'
        ]
    }

    # Calculate compression ratios for existing methods
    data = calculate_compression_ratios(data, compression_methods_existing, original_size_column)

    # Calculate compression ratios for new methods
    data = calculate_compression_ratios(data, compression_methods_new, original_size_column)

    # Define a color palette (e.g., Tableau Colors)
    colors = list(mcolors.TABLEAU_COLORS.values())

    # Plot 1: Compression Ratios for Existing Methods
    plot_compression_ratios(
        data=data,
        compression_methods=compression_methods_existing,
        label_mapping=label_mapping,
        colors=colors,
        plot_filename='compression_ratios_existing.png'  # Name your plot accordingly
    )

    # Plot 2: Compression Ratios for New Methods
    plot_compression_ratios(
        data=data,
        compression_methods=compression_methods_new,
        label_mapping=label_mapping,
        colors=colors,
        plot_filename='compression_ratios_new.png'  # Name your plot accordingly
    )

    print("Both plots have been generated and saved successfully.")


if __name__ == "__main__":
    main()
