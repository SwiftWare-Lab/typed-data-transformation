
import zstandard as zstd


def zstd_comp(data_set):
    return zstd.compress(data_set, 3)