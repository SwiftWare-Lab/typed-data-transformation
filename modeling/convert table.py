import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = '/home/jamalids/content_array_orig.csv'
df = pd.read_csv(file_path, header=None)


# Create an image of the DataFrame
def dataframe_to_image(df, output_path='dataframe_image.png'):
    fig, ax = plt.subplots(figsize=(10, len(df) * 0.5))  # Adjust figsize for content
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)

    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return output_path


# Convert DataFrame to image
image_path = dataframe_to_image(df, '/home/jamalids/content_array_orig.png')
image_path
