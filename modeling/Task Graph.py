import networkx as nx
import matplotlib.pyplot as plt

def draw_ordered_graph(graph, title, pos):
    plt.figure(figsize=(16, 10))
    nx.draw(graph, pos, with_labels=True, node_size=3000, node_color='lightblue',
            font_size=10, font_weight='bold', edge_color='gray', arrows=True)
    plt.title(title)
    plt.show()

def create_compression_task_graph():
    # Create a directed graph for compression
    G = nx.DiGraph()

    # Add nodes representing each step
    nodes = [
        "Input Data",
        "Separate Consecutive Values",
        "Apply RLE to Consecutive Values",
        "Decompose Leading Part", "Decompose Content Part", "Decompose Trailing Part",
        "Compress Leading", "Compress Content", "Compress Trailing"
    ]
    G.add_nodes_from(nodes)

    # Add edges in the order of the steps
    edges = [
        ("Input Data", "Separate Consecutive Values"),
        ("Separate Consecutive Values", "Apply RLE to Consecutive Values"),
        ("Separate Consecutive Values", "Decompose Leading Part"),
        ("Separate Consecutive Values", "Decompose Content Part"),
        ("Separate Consecutive Values", "Decompose Trailing Part"),
        ("Decompose Leading Part", "Compress Leading"),
        ("Decompose Content Part", "Compress Content"),
        ("Decompose Trailing Part", "Compress Trailing")
    ]
    G.add_edges_from(edges)

    # Define positions for the nodes for a tidy and ordered visualization
    pos = {
        "Input Data": (0, 3),
        "Separate Consecutive Values": (2, 3),
        "Apply RLE to Consecutive Values": (4, 5),
        "Decompose Leading Part": (4, 4), "Decompose Content Part": (4, 3), "Decompose Trailing Part": (4, 2),
        "Compress Leading": (6, 4), "Compress Content": (6, 3), "Compress Trailing": (6, 2)
    }

    # Draw the graph
    draw_ordered_graph(G, "Ordered Compression Task Graph (with RLE)", pos)

# Create and display the compression task graph
create_compression_task_graph()
###############################################################
import networkx as nx
import matplotlib.pyplot as plt

def draw_ordered_graph(graph, title, pos):
    plt.figure(figsize=(16, 10))
    nx.draw(graph, pos, with_labels=True, node_size=3000, node_color='lightgreen',
            font_size=10, font_weight='bold', edge_color='gray', arrows=True)
    plt.title(title)
    plt.show()

def create_decompression_task_graph():
    # Create a directed graph for decompression
    G = nx.DiGraph()

    # Add nodes representing each step
    nodes = [
        "Compressed Input",
        "Decompress Leading", "Decompress Content", "Decompress Trailing",
        "Decompress RLE",
        "Merge Components",
       # "Reconstruct Original Data"
    ]
    G.add_nodes_from(nodes)

    # Add edges in the order of the steps
    edges = [
        ("Compressed Input", "Decompress Leading"),
        ("Compressed Input", "Decompress Content"),
        ("Compressed Input", "Decompress Trailing"),
        ("Compressed Input", "Decompress RLE"),
        ("Decompress Leading", "Merge Components"),
        ("Decompress Content", "Merge Components"),
        ("Decompress Trailing", "Merge Components"),
        ("Decompress RLE", "Merge Components"),
       # ("Merge Components", "Reconstruct Original Data")
    ]
    G.add_edges_from(edges)

    # Define positions for the nodes for a tidy and ordered visualization
    pos = {
        "Compressed Input": (0, 4),
        "Decompress Leading": (2, 5), "Decompress Content": (2, 4), "Decompress Trailing": (2, 3),
        "Decompress RLE": (2, 2),
        "Merge Components": (4, 4),
        #"Reconstruct Original Data": (6, 4)
    }

    # Draw the graph
    draw_ordered_graph(G, "Ordered Decompression Task Graph", pos)

# Create and display the decompression task graph
create_decompression_task_graph()
