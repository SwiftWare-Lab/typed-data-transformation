import math
import os
import sys
import zstandard as zstd
import pandas as pd
import numpy as np
#from matplotlib import pyplot as plt
import gzip

from matplotlib import pyplot as plt

from utils import binary_to_int
import argparse
from huffman_code import create_huffman_tree, create_huffman_codes,decode,calculate_size_of_huffman_tree,create_huffman_tree_from_dict,encode_data,decode_decompose,concat_decompose
def float_to_ieee754(f):
    """Convert a float or a numpy array of floats to their IEEE 754 binary representation and return as an integer array."""
    def float_to_binary_array(single_f):
        """Convert a single float to an integer array representing its IEEE 754 binary form."""
        binary_str = format(np.float32(single_f).view(np.uint32), '032b')
        return np.array([int(bit) for bit in binary_str], dtype=np.uint8)

    if isinstance(f, np.ndarray):
        # Apply the conversion to each element in the numpy array
        return np.array([float_to_binary_array(single_f) for single_f in f.ravel()]).reshape(f.shape + (32,))
    else:
        # Apply the conversion to a single float
        return float_to_binary_array(f)
def plot_bitmap_standalone(bool_array,name):
    """
    Plots a standalone bitmap visualization of the boolean array representing IEEE 754 binary format.

    Args:
        bool_array: Numpy array of shape (n, 32), where n is the number of floats and 32 is the bit length.
        filename: The name of the file to save the bitmap plot.
    """
    plt.figure(figsize=(10, 10))  # Create a new figure with a specific size

    # Create the bitmap plot
    plt.imshow(bool_array, cmap='gray_r', aspect='auto')

    # Add labels and title
    #plt.title('IEEE 754 Bit Representation Bitmap')
    plt.xlabel('Bit Position')
    plt.ylabel('Float Index')

    # Add a color bar to indicate 0 and 1 mapping
    plt.colorbar(label='Bit Value')

    # Save the figure
    plt.savefig(name)
    plt.show()  # Display the plot
##############################################################
import plotly.graph_objects as go
def plot_table_float1(float_values, name):
    """
    Plots a simple table with float values, removing extra white space and color.

    Args:
        float_values: List of float values.
        name: The filename to save the image.
    """
    # Column headers for the float values
    columns = ["Float Value"]

    # Prepare the data for the table
    table_data = [float_values]

    # Create the table without any color formatting
    fig = go.Figure(data=[go.Table(
        columnwidth=[400],  # Adjust the column width for float values
        header=dict(values=columns,  # Only "Float Value" header
                    fill_color='white',  # No color
                    align='center'),
        cells=dict(values=table_data,  # Only float values
                   fill_color='white',  # No color
                   align='center')
    )])

    # Update layout to remove white space and margins
    fig.update_layout(
        autosize=False,
        width=400,  # Adjust width based on table size
        height=900,  # Adjust height based on table size
        margin=dict(l=0, r=0, t=0, b=0),  # Remove extra margins
        paper_bgcolor='rgba(0,0,0,0)',  # Make the background transparent (no white space)
    )

    # Save the figure as a PNG with proper scaling
    fig.write_image(name, scale=3)  # Increase scale for higher resolution
import plotly.graph_objects as go

import plotly.graph_objects as go

import plotly.graph_objects as go

import plotly.graph_objects as go
import plotly.graph_objects as go

import plotly.graph_objects as go


def plot_table_float(float_values, name):
    """
    Plots a simple table with float values, removing extra white space and color.

    Args:
        float_values: List of float values or values convertible to float.
        name: The filename to save the image.
    """
    # Round the float values to two decimal places and handle non-float cases
    formatted_values = []
    for value in float_values:
        if isinstance(value, (int, float)):  # Check if the value is a number
            formatted_values.append(f"{value:.2f}")
        else:
            formatted_values.append(str(value))  # Convert non-float values to strings

    # Column headers for the float values
    columns = ["Float Value"]

    # Prepare the data for the table
    table_data = [formatted_values]

    # Create the table without any color formatting
    fig = go.Figure(data=[go.Table(
        columnwidth=[400],  # Adjust the column width for float values
        header=dict(values=columns,  # Only "Float Value" header
                    fill_color='white',  # No color
                    align='center',
                    font=dict(size=14, color='black', family="Arial", weight="bold")),  # Header bold
        cells=dict(values=table_data,  # Only float values
                   fill_color='white',  # No color
                   align='center',
                   font=dict(size=12, color='black', family="Arial", weight="bold"))  # Make values bold
    )])

    # Update layout to remove white space and margins
    fig.update_layout(
        autosize=False,
        width=400,  # Adjust width based on table size
        height=1200,  # Adjust height to fit 50 rows or more
        margin=dict(l=0, r=0, t=0, b=0),  # Remove extra margins
        paper_bgcolor='rgba(0,0,0,0)',  # Make the background transparent (no white space)
    )

    # Save the figure as a PNG with proper scaling
    fig.write_image(name, scale=3)  # Increase scale for higher resolution
def plot_table(binary_data, float_values, name):
    """
    Plots a table with 32-bit binary data and corresponding float values, expanding it to cover the full page and removing extra white space.

    Args:
        binary_data: List of lists containing binary data (e.g., 0s and 1s).
        float_values: List of float values corresponding to the binary data rows.
        name: The filename to save the image.
    """
    # Column headers for the 32 bits and the float value
    columns = [f"Bit {i + 1}" for i in range(32)] + ["Float Value"]

    # Prepare data for the table: Transpose binary data and then append float values
    binary_transposed = list(zip(*binary_data))  # Transpose binary data
    binary_transposed.append(float_values)  # Add the float values as the last column

    # Create the table with no colors
    fig = go.Figure(data=[go.Table(
        columnwidth=[40] * 32 + [400],  # Adjust the column widths, increasing float column width
        header=dict(values=columns,  # Header for 32 bits + float value
                    fill_color='white',  # No color
                    align='center'),
        cells=dict(values=binary_transposed,  # Binary columns + float value column
                   fill_color='white',  # No color
                   align='center')
    )])

    # Update layout to remove white space and margins
    fig.update_layout(
        autosize=False,
        width=800,  # Adjust width based on table size
        height=250,  # Adjust height based on table size
        margin=dict(l=0, r=0, t=0, b=0),  # Remove extra margins
        paper_bgcolor='rgba(0,0,0,0)',  # Make the background transparent (no white space)
    )

    # Save the figure as a PNG with proper scaling
    fig.write_image(name, scale=3)  # Increase scale for higher resolution
import plotly.graph_objects as go

####################################################################
dataset_path ="/home/jamalids/Documents/2D/data1/Fcbench/Low-Entropy/32/citytemp_f32.tsv"
ts_data1 = pd.read_csv(dataset_path, delimiter='\t', header=None)
dataset_name = os.path.basename(dataset_path).replace('.tsv', '')

group = ts_data1.drop(ts_data1.columns[0], axis=1)
group = group.T
group = group.iloc[0:1,1:10]
group = group.astype(np.float32).to_numpy().reshape(-1)

        ##############################
bool_array = float_to_ieee754(group)

plot_table(bool_array, group, "table_with_binary_and_float22.png")


