
import sys

import numba
import numpy as np
from scipy.io import mmread
import math
import os
import datetime
#from numba import njit, prange, jit
from scipy.sparse import csr_matrix
from multiprocessing import Pool


def verify_mining(mat, pat_list, occurence):
    total_nnz, mined_nnz = np.count_nonzero(mat), 0
    for i in range(len(pat_list)):
        mined_nnz += np.count_nonzero(pat_list[i]) * occurence[i]
    #print("total nnz: ", total_nnz, "mined nnz: ", mined_nnz, "error: ", total_nnz - mined_nnz)


# a function to get a matrix and a list of 2D patters and returns the occurance of each pattern in the matrix in a non-overlapping manner
def get_pattern_occurance_non_overlapping(mat, pattern_list):
    # a list to store the occurance of each pattern
    pattern_occurance = [0] * len(pattern_list)
    pm, pn = pattern_list[0].shape[0], pattern_list[0].shape[1]

    # go over each row in the matrix
    i, j = 0, 0
    # while i < mat.shape[0]:  #
    for i in range(0, mat.shape[0], pm):
        # go over each column in the matrix
        j = 0
        while j < mat.shape[1]:  # for j in range(0, mat.shape[1], pn):
            # check if mat slice is all zero
            #if np.count_nonzero(mat[i:i + pm, j:j + pn]) == 0:
            #    continue
                # go over each pattern in the pattern_list
            for k in range(len(pattern_list)):
                pattern = pattern_list[k]
                pm, pn = pattern.shape[0], pattern.shape[1]
                # check if the pattern is in the matrix
                if (i + pm <= mat.shape[0]) and (j + pn <= mat.shape[1]):
                    if np.array_equal(mat[i:i + pm, j:j + pn], pattern):
                        pattern_occurance[k] += 1
                        j += pn
                        #i += pm
                        break
                else:
                    # pad mat with zeros
                    slice = mat[i:i + pm, j:j + pn]
                    # pad slice with zeros to be a multiple of pattern size
                    slice_mat = np.pad(slice, ((0, pm - slice.shape[0] ), (0, pn - slice.shape[1] )), 'constant')
                    if np.array_equal(slice_mat, pattern):
                        pattern_occurance[k] += 1
                        j += pn
                        #i += pm
                        break

    verify_mining(mat, pattern_list, pattern_occurance)
    return pattern_occurance


def generate_patterns(m, n):
    # a list to store all patterns
    pattern_list = []
    # go over all possible patterns
    for i in range(2 ** (m * n)):
        # convert i to binary
        bin_i = bin(i)[2:].zfill(m * n)
        # skip if all zeros in bin_i
        #if bin_i == '0' * (m * n):
        #    continue
        # convert bin_i to a matrix
        pattern = np.array([int(i) for i in bin_i]).reshape((m, n))
        # add pattern to pattern_list
        pattern_list.append(pattern)
    return pattern_list

