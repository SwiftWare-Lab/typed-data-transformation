
import struct

# convert a floating point number to a binary string
def float_to_bin(f):
    return format(struct.unpack('!I', struct.pack('!f', f))[0], '032b')

# convert a binary string to a floating point number
def bin_to_float(b):
    return struct.unpack('!f', struct.pack('!I', int(b, 2)))[0]


# convert an array of floating point numbers to an array of binary strings
def float_to_bin_array(a):
    array = []
    for f in a:
        array.append(float_to_bin(f))
    return array


# convert an array binary strings to a bit map image
def bin_to_image(b):
    img = []
    for i in range(len(b)):
        row = []
        for j in range(len(b[0])):
            row.append(int(b[i][j]))
        img.append(row)
    return img

# plot a bit map image
def plot_image(img):
    import matplotlib.pyplot as plt
    plt.imshow(img, cmap='gray')
    plt.show()


# def generate random array of float32 numbers smoothly increasing from 0 to 1
def generate_smooth_array(n):
    import numpy as np
    a = np.linspace(0, 1, n).astype(np.float32)
    return a

# define main entry point
if __name__ == '__main__':

    # convert a floating point number to a binary string
    f = 3.14159
    b = float_to_bin(f)
    print(f, b)

    # convert a binary string to a floating point number
    f = bin_to_float(b)
    print(b, f)

    # convert an array binary strings to a bit map image
    a_len = 16
    # generate a random array of float32 numbers
    import numpy as np
    a = np.random.rand(a_len).astype(np.float32)
    str_array = float_to_bin_array(a)
    img = bin_to_image(str_array)
    plot_image(img)

    # scale a random array of float32 numbers by 10
    a = a * 10
    str_array = float_to_bin_array(a)
    img = bin_to_image(str_array)
    plot_image(img)

    # generate a random array of float32 of less than 1
    a = generate_smooth_array(a_len)
    str_array = float_to_bin_array(a)
    img = bin_to_image(str_array)
    plot_image(img)

    # scale smooth array by 10
    a = a * 10
    str_array = float_to_bin_array(a)
    img = bin_to_image(str_array)
    plot_image(img)

    exit(0)
