//
// Created by jamalids on 13/11/24.
//

#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <zlib.h>
#include <chrono>
#include <cstdint>
#include <omp.h>
#include <cstring>
#include <cxxopts.hpp>
#include "profiling_info.h"

// Declare globalByteArray as an external variable
extern std::vector<uint8_t> globalByteArray;

// Verify if original and reconstructed data match
bool verifyDataMatch(const std::vector<uint8_t>& original, const std::vector<uint8_t>& reconstructed) {
    if (original.size() != reconstructed.size()) {
        std::cerr << "Size mismatch: Original size = " << original.size() << ", Reconstructed size = " << reconstructed.size() << std::endl;
        return false;
    }

    for (size_t i = 0; i < original.size(); i++) {
        if (original[i] != reconstructed[i]) {
            std::cerr << "Data mismatch at index " << i << ": Original = " << static_cast<int>(original[i]) << ", Reconstructed = " << static_cast<int>(reconstructed[i]) << std::endl;
            return false;
        }
    }
    return true;
}

void splitBytesIntoComponents(const std::vector<uint8_t>& byteArray,
                              std::vector<uint8_t>& leading,
                              std::vector<uint8_t>& content,
                              std::vector<uint8_t>& trailing,
                              size_t leadingBytes,
                              size_t contentBytes,
                              size_t trailingBytes) {
    size_t numElements = byteArray.size() / (leadingBytes + contentBytes + trailingBytes);

    leading.resize(numElements * leadingBytes);
    content.resize(numElements * contentBytes);
    trailing.resize(numElements * trailingBytes);

    #pragma omp simd
    for (size_t i = 0; i < numElements; ++i) {
        size_t base = i * (leadingBytes + contentBytes + trailingBytes);
        for (size_t j = 0; j < leadingBytes; ++j) {
            leading[i * leadingBytes + j] = byteArray[base + j];
        }
        for (size_t k = 0; k < contentBytes; ++k) {
            content[i * contentBytes + k] = byteArray[base + leadingBytes + k];
        }
        for (size_t l = 0; l < trailingBytes; ++l) {
            trailing[i * trailingBytes + l] = byteArray[base + leadingBytes + contentBytes + l];
        }
    }
}

// Compress with gzip (zlib)
size_t compressWithGzip(const std::vector<uint8_t>& data, std::vector<uint8_t>& compressedData, int compressionLevel) {
    uLongf compressedSize = compressBound(data.size());
    compressedData.resize(compressedSize);

    int result = compress2(compressedData.data(), &compressedSize, data.data(), data.size(), compressionLevel);
    if (result != Z_OK) {
        std::cerr << "Gzip compression error: " << result << std::endl;
        return 0;
    }
    compressedData.resize(compressedSize);
    return compressedSize;
}

// Decompress with gzip (zlib)
size_t decompressWithGzip(const std::vector<uint8_t>& compressedData, std::vector<uint8_t>& decompressedData, size_t originalSize) {
    decompressedData.resize(originalSize);

    uLongf decompressedSize = originalSize;
    int result = uncompress(decompressedData.data(), &decompressedSize, compressedData.data(), compressedData.size());
    if (result != Z_OK) {
        std::cerr << "Gzip decompression error: " << result << std::endl;
        return 0;
    }
    return decompressedSize;
}

// Full compression without decomposition
size_t gzipCompression(const std::vector<uint8_t>& data, ProfilingInfo &pi, std::vector<uint8_t>& compressedData) {
    size_t compressedSize = compressWithGzip(data, compressedData, Z_BEST_COMPRESSION);
    pi.type = "Full Compression (Gzip)";
    return compressedSize;
}

// Full decompression without decomposition
void gzipDecompression(const std::vector<uint8_t>& compressedData, std::vector<uint8_t>& decompressedData, ProfilingInfo &pi) {
    decompressWithGzip(compressedData, decompressedData, globalByteArray.size());

    // Verify decompressed data
    if (!verifyDataMatch(globalByteArray, decompressedData)) {
        std::cerr << "Error: Decompressed data doesn't match the original." << std::endl;
    }
}

// Sequential compression with decomposition that takes dynamic byte sizes as parameters
size_t gzipDecomposedSequential(const std::vector<uint8_t>& data, ProfilingInfo &pi,
                                std::vector<uint8_t>& compressedLeading,
                                std::vector<uint8_t>& compressedContent,
                                std::vector<uint8_t>& compressedTrailing,
                                size_t leadingBytes, size_t contentBytes, size_t trailingBytes) {
    std::vector<uint8_t> leading, content, trailing;
    splitBytesIntoComponents(data, leading, content, trailing, leadingBytes, contentBytes, trailingBytes);

    auto start = std::chrono::high_resolution_clock::now();
    size_t compressedSize_l = compressWithGzip(leading, compressedLeading, Z_BEST_COMPRESSION);
    auto end = std::chrono::high_resolution_clock::now();
    pi.leading_time = std::chrono::duration<double>(end - start).count();

    start = std::chrono::high_resolution_clock::now();
    size_t compressedSize_c = compressWithGzip(content, compressedContent, Z_BEST_COMPRESSION);
    end = std::chrono::high_resolution_clock::now();
    pi.content_time = std::chrono::duration<double>(end - start).count();

    start = std::chrono::high_resolution_clock::now();
    size_t compressedSize_t = compressWithGzip(trailing, compressedTrailing, Z_BEST_COMPRESSION);
    end = std::chrono::high_resolution_clock::now();
    pi.trailing_time = std::chrono::duration<double>(end - start).count();

    pi.type = "Sequential Decomposition (Gzip)";
    return compressedSize_l + compressedSize_c + compressedSize_t;
}

// Sequential decompression with decomposition
void gzipDecomposedSequentialDecompression(const std::vector<uint8_t>& compressedLeading,
                                           const std::vector<uint8_t>& compressedContent,
                                           const std::vector<uint8_t>& compressedTrailing,
                                           ProfilingInfo &pi,
                                           size_t leadingBytes, size_t contentBytes, size_t trailingBytes) {
    size_t totalSize = globalByteArray.size();
    size_t floatCount = totalSize / (leadingBytes + contentBytes + trailingBytes);
    std::vector<uint8_t> reconstructedData(totalSize);

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<uint8_t> tempLeading(floatCount * leadingBytes);
    decompressWithGzip(compressedLeading, tempLeading, floatCount * leadingBytes);
    for (size_t i = 0; i < floatCount; ++i) {
        for (size_t j = 0; j < leadingBytes; ++j) {
            reconstructedData[i * (leadingBytes + contentBytes + trailingBytes) + j] = tempLeading[i * leadingBytes + j];
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    pi.leading_time = std::chrono::duration<double>(end - start).count();

    start = std::chrono::high_resolution_clock::now();
    std::vector<uint8_t> tempContent(floatCount * contentBytes);
    decompressWithGzip(compressedContent, tempContent, floatCount * contentBytes);
    for (size_t i = 0; i < floatCount; ++i) {
        for (size_t j = 0; j < contentBytes; ++j) {
            reconstructedData[i * (leadingBytes + contentBytes + trailingBytes) + leadingBytes + j] = tempContent[i * contentBytes + j];
        }
    }
    end = std::chrono::high_resolution_clock::now();
    pi.content_time = std::chrono::duration<double>(end - start).count();

    start = std::chrono::high_resolution_clock::now();
    std::vector<uint8_t> tempTrailing(floatCount * trailingBytes);
    decompressWithGzip(compressedTrailing, tempTrailing, floatCount * trailingBytes);
    for (size_t i = 0; i < floatCount; ++i) {
        for (size_t j = 0; j < trailingBytes; ++j) {
            reconstructedData[i * (leadingBytes + contentBytes + trailingBytes) + leadingBytes + contentBytes + j] = tempTrailing[i * trailingBytes + j];
        }
    }
    end = std::chrono::high_resolution_clock::now();
    pi.trailing_time = std::chrono::duration<double>(end - start).count();

    if (!verifyDataMatch(globalByteArray, reconstructedData)) {
        std::cerr << "Error: Decompressed data doesn't match the original." << std::endl;
    }
}
// Parallel compression with decomposition using gzip
size_t gzipDecomposedParallel(const std::vector<uint8_t>& data, ProfilingInfo &pi,
                              std::vector<uint8_t>& compressedLeading,
                              std::vector<uint8_t>& compressedContent,
                              std::vector<uint8_t>& compressedTrailing,
                              size_t leadingBytes, size_t contentBytes, size_t trailingBytes, int numThreads) {
    std::vector<uint8_t> leading, content, trailing;
    splitBytesIntoComponents(data, leading, content, trailing, leadingBytes, contentBytes, trailingBytes);

    double compressedSize_l = 0, compressedSize_c = 0, compressedSize_t = 0;
    omp_set_num_threads(numThreads);

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            auto start = std::chrono::high_resolution_clock::now();
            compressedSize_l = compressWithGzip(leading, compressedLeading, Z_BEST_COMPRESSION);
            auto end = std::chrono::high_resolution_clock::now();
            pi.leading_time = std::chrono::duration<double>(end - start).count();
        }

        #pragma omp section
        {
            auto start = std::chrono::high_resolution_clock::now();
            compressedSize_c = compressWithGzip(content, compressedContent, Z_BEST_COMPRESSION);
            auto end = std::chrono::high_resolution_clock::now();
            pi.content_time = std::chrono::duration<double>(end - start).count();
        }

        #pragma omp section
        {
            auto start = std::chrono::high_resolution_clock::now();
            compressedSize_t = compressWithGzip(trailing, compressedTrailing, Z_BEST_COMPRESSION);
            auto end = std::chrono::high_resolution_clock::now();
            pi.trailing_time = std::chrono::duration<double>(end - start).count();
        }
    }

    pi.type = "Parallel Decomposition (Gzip)";
    return (compressedLeading.size() + compressedContent.size() + compressedTrailing.size());
}

// Parallel decompression function with dynamic byte segmentation using gzip
void gzipDecomposedParallelDecompression(const std::vector<uint8_t>& compressedLeading,
                                         const std::vector<uint8_t>& compressedContent,
                                         const std::vector<uint8_t>& compressedTrailing,
                                         ProfilingInfo &pi,
                                         size_t leadingBytes, size_t contentBytes, size_t trailingBytes, int numThreads) {
    size_t totalSize = globalByteArray.size();
    size_t floatCount = totalSize / (leadingBytes + contentBytes + trailingBytes);
    std::vector<uint8_t> reconstructedData(totalSize);
    omp_set_num_threads(numThreads);

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            auto start = std::chrono::high_resolution_clock::now();
            std::vector<uint8_t> tempLeading(floatCount * leadingBytes);
            decompressWithGzip(compressedLeading, tempLeading, floatCount * leadingBytes);
            for (size_t i = 0; i < floatCount; ++i) {
                for (size_t j = 0; j < leadingBytes; ++j) {
                    reconstructedData[i * (leadingBytes + contentBytes + trailingBytes) + j] = tempLeading[i * leadingBytes + j];
                }
            }
            auto end = std::chrono::high_resolution_clock::now();
            pi.leading_time = std::chrono::duration<double>(end - start).count();
        }

        #pragma omp section
        {
            auto start = std::chrono::high_resolution_clock::now();
            std::vector<uint8_t> tempContent(floatCount * contentBytes);
            decompressWithGzip(compressedContent, tempContent, floatCount * contentBytes);
            for (size_t i = 0; i < floatCount; ++i) {
                for (size_t j = 0; j < contentBytes; ++j) {
                    reconstructedData[i * (leadingBytes + contentBytes + trailingBytes) + leadingBytes + j] = tempContent[i * contentBytes + j];
                }
            }
            auto end = std::chrono::high_resolution_clock::now();
            pi.content_time = std::chrono::duration<double>(end - start).count();
        }

        #pragma omp section
        {
            auto start = std::chrono::high_resolution_clock::now();
            std::vector<uint8_t> tempTrailing(floatCount * trailingBytes);
            decompressWithGzip(compressedTrailing, tempTrailing, floatCount * trailingBytes);
            for (size_t i = 0; i < floatCount; ++i) {
                for (size_t j = 0; j < trailingBytes; ++j) {
                    reconstructedData[i * (leadingBytes + contentBytes + trailingBytes) + leadingBytes + contentBytes + j] = tempTrailing[i * trailingBytes + j];
                }
            }
            auto end = std::chrono::high_resolution_clock::now();
            pi.trailing_time = std::chrono::duration<double>(end - start).count();
        }
    }

    // Verify decompressed data
    if (!verifyDataMatch(globalByteArray, reconstructedData)) {
        std::cerr << "Error: Decompressed data doesn't match the original." << std::endl;
    }
}
double calculateCompressionRatio(size_t originalSize, size_t compressedSize) {
  return static_cast<double>(originalSize) / static_cast<double>(compressedSize);
}

