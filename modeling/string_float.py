from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import fetch_california_housing
from sklearn.datasets import load_diabetes

# Load the 20 Newsgroups dataset and print its length
def load_20newsgroups_dataset():
    dataset = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    data = dataset.data  # List of strings (documents)
    return data



# Load the California Housing dataset and print its length
def load_california_housing_dataset():
    dataset = fetch_california_housing()
    data = dataset.data  # 2D array of floating-point numbers
    return data



# Load the Diabetes dataset and select one variable
def load_single_variable_dataset():
    dataset = load_diabetes()
    data = dataset.data[:, 0]  # Select the first variable (column)
    return data



def decompose_strings(ds):
    b0, b1, b2, b3 = [], [], [], []
    len_m4 = (len(ds) - len(ds) % 4)
    for i in range(0, len_m4, 4):
        b0.append(ds[i+0])
        b1.append(ds[i+1])
        b2.append(ds[i+2])
        b3.append(ds[i+3])
    return b0, b1, b2, b3


import math
from collections import Counter


def compute_entropy(stream):
    # Count occurrences of each element
    counts = Counter(stream)
    total = len(stream)

    # Calculate probabilities and entropy
    entropy = 0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)

    return entropy


# # Example usage
# dataset = load_single_variable_dataset()

# Example usage
# dataset = load_california_housing_dataset()
# print(f"Number of rows in the dataset: {len(dataset)}")


if __name__ == "__main__":
    # Example usage
    dataset = load_20newsgroups_dataset()
    merged_dataset = ""
    for item in dataset:
        merged_dataset = merged_dataset + item
    # convert list to array pf characters
    #dataset = [list(item) for item in dataset]

    print(f"Number of strings in the dataset: {len(merged_dataset)}")

    b0, b1, b2, b3 = decompose_strings(merged_dataset)

    # compute entropy of each string
    ent_b0 = compute_entropy(b0)
    ent_b1 = compute_entropy(b1)
    ent_b2 = compute_entropy(b2)
    ent_b3 = compute_entropy(b3)
    ent_all = compute_entropy(merged_dataset)
    #
    print(f"Entropy of b0: {ent_b0}, b1: {ent_b1}, b2: {ent_b2}, b3: {ent_b3}, all: {ent_all}")


