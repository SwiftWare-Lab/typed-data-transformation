import plotly.graph_objects as go

def plot_table(float_values, name):
    """
    Plots a table with each tuple (float value, second value) displayed in a single row
    with multiple columns, without a header and with borders around each cell.

    Args:
        float_values: List of tuples with float and second values.
        name: The filename to save the image.
    """
    # Prepare the data to be displayed in one row: Each tuple in a separate column
    data = [f"({val[0]:.6f}, {val[1]})" for val in float_values]  # Format each tuple as a string

    # Transpose the data so that each tuple is in its own column
    transposed_data = [[val] for val in data]  # Each value should be in its own column, so it goes into its own list

    # Create the table with one row and multiple columns, no header, with cell borders
    fig = go.Figure(data=[go.Table(
        cells=dict(
            values=transposed_data,  # Transposed data for each column
            fill_color='lightgrey',  # Cell background color
            align='center',  # Center-align the text in each cell
            line_color='black'  # Add borders to each cell
        )
    )])

    # Update layout to remove white space and margins
    fig.update_layout(
        autosize=False,
        width=150 * len(float_values),  # Adjust width based on the number of columns
        height=100,  # Adjust height for a single row
        margin=dict(l=0, r=0, t=0, b=0),  # Remove extra margins
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
    )

    # Save the figure as a PNG with proper scaling
    fig.write_image(name, scale=3)  # Save the figure

# Example usage
float_values = [
    (0.791931, 1), (0.799409, 1), (0.800794, 1), (0.803958, 1),
    (0.795394, 1), (0.780945, 1), (0.759362, 1), (0.733009, 1)
]

plot_table(float_values, "tuple_table_single_row_with_borders.png")
