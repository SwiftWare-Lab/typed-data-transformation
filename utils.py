import numpy as np
import matplotlib.pyplot as plt
import struct
def generate_smooth_array(n, type=np.float32):
    import numpy as np
    a = np.linspace(0, 1, n).astype(type)
    return a


# generate a 2D array with a given shape of smooth values
def generate_smooth_2d_array(shape, lower_bound=0, upper_bound=1, type=np.float32):
    import numpy as np
    a = np.linspace(lower_bound, upper_bound, shape[0] * shape[1]).astype(type)
    return a.reshape(shape)


# generate 2D array of smooth oscillating values
def generate_oscillating_2d_array(shape, lower_bound=0, upper_bound=1, type=np.float32):
    a = np.linspace(lower_bound, upper_bound, shape[0] * shape[1]).astype(type)
    a = np.sin(a)
    return a.reshape(shape)


# line plot each row of a 2D array
def line_plot_2d_array(array):
    for row in array:
        plt.plot(row)
    plt.show()
    plt.close()


# store a 2D array in a tsv file
def store_tsv_file(file_name, array):
    #header = "signal id\t"
    #header += "\t".join([str(i) for i in range(array.shape[1])])
    # create an array of signal ids
    signal_ids = np.arange(array.shape[0]).reshape(-1, 1)
    # concatenate signal ids and the array
    array = np.concatenate((signal_ids, array), axis=1)
    np.savetxt(file_name, array, delimiter="\t")



# generate a 2D numpy array of boolean values and its equivalent float32 array
def generate_boolean_array(n_rows, n_cols):
    bool_array = np.random.choice([True, False], size=(n_rows, n_cols))
    float_array = bool_array_to_float32(bool_array)
    return bool_array, float_array


# a function to take a 2D bool array and retuns its decimal representation
def bool_to_int(bool_array):
    """
    packbit pads the last byte with zeros if the number of bits is not a multiple of 8
    :param bool_array:
    :return:
    """
    byte_array = np.packbits(bool_array.flatten())
    return int.from_bytes(byte_array, byteorder='big')

def char_to_bool(char_array):
    bool_array = np.array([1 if x == b'1' else 0 for x in char_array.flatten()], dtype='bool')
    return bool_array.reshape(char_array.shape)


def bool_array_to_float32(bool_array):
    ba_flatten = bool_array.flatten()
    # a float array with size of len(ba_flatten)//32
    float_array = np.zeros(len(ba_flatten)//32, dtype='float32')
    for i in range(0, len(ba_flatten), 32):
        float_array[i//32] = bool_to_int(ba_flatten[i:i+32])
    return float_array

# a function to take a decimal number and return its 2D bool array representation
def int_to_bool(int_val, m, n):
    byte_array = int_val.to_bytes((m*n)//8, byteorder='big')
    bool_array = np.unpackbits(np.frombuffer(byte_array, dtype='uint8'))
    return bool_array.reshape(m, n)


def bool_to_binary_string(bool_array):
    return ''.join(['1' if x else '0' for x in bool_array.flatten()])
def float32_to_bool_array1(float_array):
    # Convert float32 array to byte array
    byte_array = float_array.view(np.uint8)
    # Convert byte array to boolean array
    bit_array = np.unpackbits(byte_array)
    # Return only the original length of the bit array
    return bit_array

def bool_to_int1(bit_array):
    # Convert boolean array to binary string
    bit_string = ''.join(str(int(bit)) for bit in bit_array)
    # Convert binary string to integer
    value = int(bit_string, 2)
    return value
def int_to_bool1(value, m, n):
    # Convert the integer to a binary string, zero-padded to the required length
    bit_string = bin(value)[2:].zfill(m * n)
    # Convert the binary string to a boolean array
    bit_array = np.array([int(bit) for bit in bit_string], dtype=np.bool_)
    return bit_array

def bool_array_to_float321(bit_array):
    # Ensure bit_array is a boolean array
    bit_array = np.asarray(bit_array, dtype=np.bool_)
    # Pack bits into bytes
    byte_array = np.packbits(bit_array)
    # Convert byte array to float32
    # Make sure the length is divisible by 4 to match float32 size
    if len(byte_array) % 4 != 0:
        byte_array = np.pad(byte_array, (0, 4 - len(byte_array) % 4), 'constant')
    float_array = np.frombuffer(byte_array, dtype=np.float32)
    return float_array