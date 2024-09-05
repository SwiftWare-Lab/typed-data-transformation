import networkx as nx
import matplotlib.pyplot as plt


class Node:
    def __init__(self, symbol, freq):
        self.symbol = symbol
        self.freq = freq
        self.left = None
        self.right = None


# Function to add edges in the Huffman tree
def add_edges(tree, graph, label='', pos=None, x=0, y=0, layer=1):
    if tree is not None:
        pos[tree] = (x, y)
        if tree.left:
            graph.add_edge(tree, tree.left)
            l = label + '0'  # Append 0 for left child
            add_edges(tree.left, graph, l, pos=pos, x=x - 1 / layer, y=y - 1, layer=layer + 1)
        if tree.right:
            graph.add_edge(tree, tree.right)
            l = label + '1'  # Append 1 for right child
            add_edges(tree.right, graph, l, pos=pos, x=x + 1 / layer, y=y - 1, layer=layer + 1)


# Plot the Huffman Tree
def plot_huffman_tree(tree):
    graph = nx.DiGraph()
    pos = {}
    add_edges(tree, graph, pos=pos)

    # Drawing the tree
    plt.figure(figsize=(10, 10))
    nx.draw(graph, pos, with_labels=False, node_size=1000, node_color='skyblue', font_size=10)

    # Draw labels for nodes
    node_labels = {node: f"{node.symbol}\n{node.freq}" for node in graph.nodes()}
    nx.draw_networkx_labels(graph, pos, labels=node_labels)

    plt.show()


# Example Huffman tree (manually built for demonstration)
root = Node('Root', 0.5)
root.left = Node('A', 0.2)
root.right = Node('B', 0.3)
root.left.left = Node('C', 0.1)
root.left.right = Node('D', 0.1)
root.right.left = Node('E', 0.15)
root.right.right = Node('F', 0.15)

# Plot the Huffman tree
plot_huffman_tree(root)
