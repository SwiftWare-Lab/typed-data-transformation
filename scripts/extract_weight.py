
import torch
import zstd
import io
import numpy as np

def convert_int16array_toboolean(int16array):
    return torch.tensor(int16array, dtype=torch.bool)

def float16_tensor_to_int16(tensor):
    # view as int16
    int16_tensor = tensor.view(dtype=torch.int16)
    return int16_tensor


def calculate_entropy(tensor):
    """Calculates the entropy of an int16 tensor.

    Args:
        tensor: The input tensor.

    Returns:
        The entropy of the tensor.
    """

    # Flatten the tensor into a 1D array
    flattened_tensor = tensor.flatten().numpy()
    #flattened_tensor = tensor.numpy()
    # Count the frequency of each unique value
    unique, counts = np.unique(flattened_tensor, return_counts=True)

    # Calculate probabilities
    probabilities = counts / len(flattened_tensor)

    # Calculate entropy using base-2 logarithm
    entropy = -np.sum(probabilities * np.log2(probabilities))

    return entropy


def export_torch_tensor_to_tsv_file(tensor, out_path):
    # convert tensor to numpy array
    tensor_np = tensor.numpy()
    # save numpy array to a tsv file
    np.savetxt(out_path, tensor_np, delimiter="\t")

def int16_to_int8_pair(int16_tensor):
  """Converts an int16 tensor into two int8 tensors.

  Args:
    int16_tensor: The input int16 tensor.

  Returns:
    A tuple of two int8 tensors, representing the higher and lower 8 bits.
  """

  # Extract the higher 8 bits
  higher_bits = (int16_tensor >> 8).to(torch.int8)

  # Extract the lower 8 bits
  lower_bits = (int16_tensor & 0xFF).to(torch.int8)

  return higher_bits, lower_bits


def load_tsv_as_tensor_int16(file_path):
    # load the tsv file as numpy array
    tensor_np = np.loadtxt(file_path, delimiter="\t")
    # convert numpy array to tensor
    tensor = torch.tensor(tensor_np, dtype=torch.int16)
    return tensor


def compress_torch_tensor(tensor):
    # convert tensor to int16
    tensor_int16 = tensor.view(dtype=torch.int16)
    # convert weight_int to bytes
    buffer = io.BytesIO()
    torch.save(tensor_int16.flatten(), buffer)
    bytes_data = buffer.getvalue()
    # compress the weight
    compressed_weight = zstd.compress(bytes_data)
    return compressed_weight

export_enable = True

# Load the model
model = torch.load('/home/kazem/Downloads/llama-2-7b/consolidated.00.pth')
out_path = ""

total_size, total_size_manual = 0, 0
model_int16, model_bool = {}, {}
for key in model.keys():
    # get the tensor weight
    weight = model[key]
    # print the shape of the tensor
    print("====================================")
    print(key, weight.shape, "size:", weight.numel(), "element size:", weight.element_size())
    # if one dimensional tensor, continue
    if len(weight.shape) == 1:
        continue
    # calculate the size of the tensor
    total_size += weight.numel()
    # calculate the size of the tensor manually
    tensor_elements = weight.shape[0] * weight.shape[1] if len(weight.shape) == 2 else weight.shape[0]
    total_size_manual += (tensor_elements * weight.element_size())
    # convert tensors to int16
    model_int16[key] = weight.view(dtype=torch.int16)
    weight_int = weight.view(dtype=torch.int16)
    # compress the weight
    compressed_weight = compress_torch_tensor(weight_int)
    # comp ratio
    print(f"Compression ratio: {(tensor_elements * weight.element_size()) / len(compressed_weight) }")
    # compute entropy
    print(f"Entropy: {calculate_entropy(weight_int)}")
    # compute entropy of each byte
    even_bytes, odd_bytes = int16_to_int8_pair(weight_int)
    print(f"Entropy of odd bytes: {calculate_entropy(odd_bytes)}")
    print(f"Entropy of even bytes: {calculate_entropy(even_bytes)}")
    # compute entropy of each byte
    compressed_even = compress_torch_tensor(even_bytes)
    compressed_odd = compress_torch_tensor(odd_bytes)
    print(f"Compression ratio of odd bytes: {(tensor_elements * 1) / len(compressed_odd)}")
    print(f"Compression ratio of even bytes: {(tensor_elements * 1) / len(compressed_even)}")
    print(f"decomposed compression ratio: {(tensor_elements * 2) / (len(compressed_odd) + len(compressed_even))}")

    if export_enable:
        # export the tensor to a tsv file
        export_torch_tensor_to_tsv_file(weight_int, f"./weight/{key}.tsv")
        # load the tensor from the tsv file
        tensor_loaded = load_tsv_as_tensor_int16(f"./weight/{key}.tsv")
        # test if the tensor is loaded correctly
        assert torch.allclose(tensor_loaded, weight_int)



print(f"Total size of all tensors: {total_size / 1024:.2f} KB")
print(f"Total size of all tensors manually: {total_size_manual / 1024:.2f} KB")
