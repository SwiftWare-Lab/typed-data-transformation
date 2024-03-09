import os.path

# import the necessary packages
import numpy as np
import pandas as pd
import fpzip
import zlib
import pickle


precision_cut = 4
def xor_float(x, y):
    return (x.view("i")^y.view("i")).view("f")


# generate vectors a and b with a given cofficient of correlation rho
def generate_vectors(rho, n):
    # generate a random vector
    a = np.random.randn(n).astype(np.float32)
    # generate a random vector
    b = np.random.randn(n).astype(np.float32)
    # make the vector b dependent on the vector a
    b = rho * a + np.sqrt(1 - rho ** 2) * b
    return a, b


# calculate the correlation coefficient between two vectors
def correlation_coefficient(a, b):
    # calculate the mean of the vector a
    mean_a = np.mean(a)
    # calculate the mean of the vector b
    mean_b = np.mean(b)
    # calculate the correlation coefficient
    numerator = np.sum((a - mean_a) * (b - mean_b))
    denominator = np.sqrt(np.sum((a - mean_a) ** 2) * np.sum((b - mean_b) ** 2))
    return numerator / denominator


# compress two floating point vectors a and b and return the compressed vectors and the length of the compressed vectors
def compress_vectors(a, b, precision_cut):
    compressed_a = fpzip.compress(a, precision=precision_cut, order='F')  # returns byte string
    compressed_b = fpzip.compress(b, precision=precision_cut, order='F')  # returns byte string
    return compressed_a, compressed_b, len(compressed_a), len(compressed_b)

# rewrite compress vector to use zlib
def compress_vectors_zlib(a, b):
    compressed_a = zlib.compress(pickle.dumps(a))
    compressed_b = zlib.compress(pickle.dumps(b))
    return compressed_a, compressed_b, len(compressed_a), len(compressed_b)


# compress correlated vectors
def compress_correlated(a, b, precision_cut):
    compressed_a = fpzip.compress(a, precision=precision_cut, order='F')  # returns byte string
    diff_b = xor_float(b, a)
    compressed_diff_b = fpzip.compress(diff_b, precision=precision_cut, order='F')  # returns byte string
    return compressed_a, compressed_diff_b, len(compressed_a), len(compressed_diff_b)


def compress_shifted_diff(a, b, precision_cut):
    shifted_a = a[1:]
    # concatenate shifted_a with one zero
    shifted_a = np.concatenate((shifted_a, np.zeros(1, dtype=np.float32)))
    diff_a = xor_float(a, shifted_a)
    compressed_diff_a = fpzip.compress(diff_a, precision=precision_cut, order='F')  # returns byte string
    shifted_b = b[1:]
    # concatenate shifted_b with one zero
    shifted_b = np.concatenate((shifted_b, np.zeros(1, dtype=np.float32)))
    diff_b = xor_float(b, shifted_b)
    compressed_diff_b = fpzip.compress(diff_b, precision=precision_cut, order='F')  # returns byte string
    return compressed_diff_a, compressed_diff_b, len(compressed_diff_a), len(compressed_diff_b)


def compress_shifted_diff_zlib(a, b, rho):
    shifted_a = a[1:]
    # concatenate shifted_a with one zero
    shifted_a = np.concatenate((shifted_a, np.zeros(1, dtype=np.float32)))
    diff_a = xor_float(a, shifted_a)
    compressed_diff_a = zlib.compress(pickle.dumps(diff_a))
    shifted_b = b[1:]
    # concatenate shifted_b with one zero
    shifted_b = np.concatenate((shifted_b, np.zeros(1, dtype=np.float32)))
    diff_b = xor_float(b, shifted_b)
    compressed_diff_b = zlib.compress(pickle.dumps(diff_b))
    return compressed_diff_a, compressed_diff_b, len(compressed_diff_a), len(compressed_diff_b)


# rewrite compress correlated to use zlib
def compress_correlated_zlib(a, b, rho):
    compressed_a = zlib.compress(pickle.dumps(a))
    diff_b = xor_float(b, a)
    compressed_diff_b = zlib.compress(pickle.dumps(diff_b))
    return compressed_a, compressed_diff_b, len(compressed_a), len(compressed_diff_b)


def evaluate_compression(a, b, rho, precision_range):
    log_dic_array = []
    for p in precision_range:
        log_dic = {}
        log_dic['vector length'] = len(a)
        log_dic['coef'] = rho
        log_dic['precision'] = p
        uc_len = len(a) * a.itemsize + len(b) * b.itemsize
        log_dic['uncompressed len'] = uc_len
        compressed_a, compressed_b, len_a, len_b = compress_vectors(a, b, p)
        sep_comp = len_a + len_b
        log_dic['fpzip sep comp'] = sep_comp
        compressed_a, compressed_diff_b, len_a, len_diff_b = compress_correlated(a, b, p)
        log_dic['fpzip corr comp'] = len_a + len_diff_b

        compressed_diff_a, compressed_diff_b, len_diff_a, len_diff_b = compress_shifted_diff(a, b, p)
        log_dic['fpzip shifted comp'] = len_diff_a + len_diff_b

        compressed_a, compressed_b, len_a, len_b = compress_vectors_zlib(a, b)
        sep_comp = len_a + len_b
        log_dic['zlib sep comp'] = sep_comp
        compressed_a, compressed_diff_b, len_a, len_diff_b = compress_correlated_zlib(a, b, rho)
        log_dic['zlib corr comp'] = len_a + len_diff_b
        compressed_diff_a, compressed_diff_b, len_diff_a, len_diff_b = compress_shifted_diff_zlib(a, b, rho)
        log_dic['zlib shifted comp'] = len_diff_a + len_diff_b
        log_dic_array.append(log_dic)
    # convert the dictionary to a pandas dataframe
    df = pd.DataFrame(log_dic_array)
    return df


def plot_one_pair(df, plot_name, output_dir=None):
    import matplotlib.pyplot as plt
    baseline = df['uncompressed len'].values
    plt.plot(df['precision'], baseline/df['fpzip sep comp'], label='fpzip sep comp')
    plt.plot(df['precision'], baseline/df['fpzip corr comp'], label='fpzip corr comp')
    plt.plot(df['precision'], baseline/df['fpzip shifted comp'], label='fpzip shifted comp')
    plt.plot(df['precision'], baseline/df['zlib sep comp'], label='zlib sep comp')
    plt.plot(df['precision'], baseline/df['zlib corr comp'], label='zlib corr comp')
    plt.plot(df['precision'], baseline/df['zlib shifted comp'], label='zlib shifted comp')
    plt.xlabel('precision')
    plt.ylabel('uncompressed size / compressed size')
    plt.yscale('log')
    plt.title(plot_name)
    plt.legend()
    if output_dir is not None:
        # make output file if not exists
        if not os.path.exists(os.path.join(output_dir,'plot')):
            os.makedirs(os.path.join(output_dir,'plot'))
        plt.savefig(os.path.join(output_dir, 'plot', plot_name + '.png'))
        plt.close()
        return
    plt.show()
    plt.close()

def synthetic_bench(r_range, precision_range):
    # empty dataframe
    df_all = pd.DataFrame()
    for rho in r_range:
        a, b = generate_vectors(rho, 100000)
        df = evaluate_compression(a, b, r_range, precision_range)
        plot_one_pair(df, 'rand1_rand_2_rho_{}'.format(rho), './correlation/')
        # add the dataframe to the empty dataframe
        df_all = pd.concat([df_all, df])

    return df_all


def electricity_bench(in_file, precision_range):
    import pandas as pd
    df = pd.read_csv(in_file, sep=';')
    # go over all strings, if string, in the dataframe and replace ',' with nothing
    df = df.applymap(lambda x: x.replace(',', '.') if type(x) == str else x)
    # convert 2d array of string ts_list to float
    df = df.apply(pd.to_numeric, errors='coerce', downcast='float')
    # normalize the data
    #df = (df - df.mean()) / df.std()

    corr_matrix = df.corr(numeric_only=True)
    #print(corr_matrix)
    # find unique location of values more than .60 in the correlation matrix
    locs = np.where(corr_matrix > 0.95)
    # for every location, extract the row and column and calculate the correlation coefficient
    for i in range(len(locs[0])):
        row = locs[0][i]
        col = locs[1][i]
        a = df[df.columns[row]].values
        b = df[df.columns[col]].values
        rho = corr_matrix.iloc[row, col]
        df_log = evaluate_compression(a, b, rho, precision_range)
        plot_one_pair(df_log, 'col1_{}_col2_{}_rho_{}'.format(df.columns[row], df.columns[col], rho), './correlation/')


# main entry point
if __name__ == '__main__':
    synth_df = synthetic_bench([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1],
                               [0, 2, 3, 4, 5, 6, 7, 8])
    # plot synthetic data
    print(synth_df)

    electricity_bench('data/LD2011_2014.txt', [0, 2, 4, 8])

