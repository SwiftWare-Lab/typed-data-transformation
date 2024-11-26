//
// Created by jamalids on 25/11/24.
//

#ifndef SNAPPY_PARALLEL_H
#define SNAPPY_PARALLEL_H
#include <vector>
#include <iostream>
#include <cstring>
#include <snappy.h>
#include "profiling_info.h"
#include <omp.h>

// Declare globalByteArray as an external variable
extern std::vector<uint8_t> globalByteArray;
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

  // Resize the vectors to accommodate the exact number of bytes
  leading.resize(numElements * leadingBytes);
  content.resize(numElements * contentBytes);
  trailing.resize(numElements * trailingBytes);

  // Using OpenMP SIMD to optimize vector operations
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

// Compress with Snappy
size_t compressWithSnappy(const std::vector<uint8_t>& data, std::vector<uint8_t>& compressedData) {
  size_t compressedSize = snappy::MaxCompressedLength(data.size());
  compressedData.resize(compressedSize);

  snappy::RawCompress(reinterpret_cast<const char*>(data.data()), data.size(),
                      reinterpret_cast<char*>(compressedData.data()), &compressedSize);

  compressedData.resize(compressedSize); // Resize to actual compressed size
  return compressedSize;
}

// Decompress with Snappy
size_t decompressWithSnappy(const std::vector<uint8_t>& compressedData, std::vector<uint8_t>& decompressedData, size_t originalSize) {
  decompressedData.resize(originalSize);

  bool success = snappy::RawUncompress(reinterpret_cast<const char*>(compressedData.data()),
                                       compressedData.size(),
                                       reinterpret_cast<char*>(decompressedData.data()));
  if (!success) {
    std::cerr << "Snappy decompression error." << std::endl;
    return 0;
  }

  return originalSize;
}



// Full compression without decomposition
size_t snappyCompression(const std::vector<uint8_t>& data, ProfilingInfo &pi, std::vector<uint8_t>& compressedData) {
  auto start = std::chrono::high_resolution_clock::now();
  size_t compressedSize = compressWithSnappy(data, compressedData); // Fast compression
  auto end = std::chrono::high_resolution_clock::now();

  pi.type = "LZ4 Full Compression";

  return compressedSize;
}
// Full decompression without decomposition
void snappyDecompression(const std::vector<uint8_t>& compressedData, std::vector<uint8_t>& decompressedData, ProfilingInfo &pi) {

  decompressWithSnappy(compressedData, decompressedData, globalByteArray.size());

  // Verify decompressed data
  if (!verifyDataMatch(globalByteArray, decompressedData)) {
    std::cerr << "Error: Decompressed data doesn't match the original." << std::endl;
  }
}
// Sequential compression with decomposition that takes dynamic byte sizes as parameters
size_t snappyDecomposedSequential(const std::vector<uint8_t>& data, ProfilingInfo &pi,
                                std::vector<uint8_t>& compressedLeading,
                                std::vector<uint8_t>& compressedContent,
                                std::vector<uint8_t>& compressedTrailing,
                                size_t leadingBytes, size_t contentBytes, size_t trailingBytes) {
    std::vector<uint8_t> leading, content, trailing;
    splitBytesIntoComponents(data, leading, content, trailing, leadingBytes, contentBytes, trailingBytes);

    auto start = std::chrono::high_resolution_clock::now();
    size_t compressedSize_l = compressWithSnappy(leading, compressedLeading);
    auto end = std::chrono::high_resolution_clock::now();
    pi.leading_time = std::chrono::duration<double>(end - start).count();

    start = std::chrono::high_resolution_clock::now();
    size_t compressedSize_c = compressWithSnappy(content, compressedContent);
    end = std::chrono::high_resolution_clock::now();
    pi.content_time = std::chrono::duration<double>(end - start).count();

    start = std::chrono::high_resolution_clock::now();
    size_t compressedSize_t = compressWithSnappy(trailing, compressedTrailing);
    end = std::chrono::high_resolution_clock::now();
    pi.trailing_time = std::chrono::duration<double>(end - start).count();

    pi.type = "Sequential Decomposition";
    return compressedSize_l + compressedSize_c + compressedSize_t;
}

// Sequential decompression with decomposition that  takes dynamic byte sizes as parameters
void snappyDecomposedSequentialDecompression(const std::vector<uint8_t>& compressedLeading,
                                           const std::vector<uint8_t>& compressedContent,
                                           const std::vector<uint8_t>& compressedTrailing,
                                           ProfilingInfo &pi,
                                           size_t leadingBytes, size_t contentBytes, size_t trailingBytes) {
    size_t totalSize = globalByteArray.size();  // Ensure globalByteArray is appropriately defined and accessible
    size_t floatCount = totalSize / (leadingBytes + contentBytes + trailingBytes);

    std::vector<uint8_t> reconstructedData(totalSize);

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<uint8_t> tempLeading(floatCount * leadingBytes);
    decompressWithSnappy(compressedLeading, tempLeading, floatCount * leadingBytes);
    for (size_t i = 0; i < floatCount; ++i) {
        for (size_t j = 0; j < leadingBytes; ++j) {
            reconstructedData[i * (leadingBytes + contentBytes + trailingBytes) + j] = tempLeading[i * leadingBytes + j];
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    pi.leading_time = std::chrono::duration<double>(end - start).count();

    start = std::chrono::high_resolution_clock::now();
    std::vector<uint8_t> tempContent(floatCount * contentBytes);
    decompressWithSnappy(compressedContent, tempContent, floatCount * contentBytes);
    for (size_t i = 0; i < floatCount; ++i) {
        for (size_t j = 0; j < contentBytes; ++j) {
            reconstructedData[i * (leadingBytes + contentBytes + trailingBytes) + leadingBytes + j] = tempContent[i * contentBytes + j];
        }
    }
    end = std::chrono::high_resolution_clock::now();
    pi.content_time = std::chrono::duration<double>(end - start).count();

    start = std::chrono::high_resolution_clock::now();
    std::vector<uint8_t> tempTrailing(floatCount * trailingBytes);
    decompressWithSnappy(compressedTrailing, tempTrailing, floatCount * trailingBytes);
    for (size_t i = 0; i < floatCount; ++i) {
        for (size_t j = 0; j < trailingBytes; ++j) {
            reconstructedData[i * (leadingBytes + contentBytes + trailingBytes) + leadingBytes + contentBytes + j] = tempTrailing[i * trailingBytes + j];
        }
    }
    end = std::chrono::high_resolution_clock::now();
    pi.trailing_time = std::chrono::duration<double>(end - start).count();

    // Verify decompressed data
    if (!verifyDataMatch(globalByteArray, reconstructedData)) {
        std::cerr << "Error: Decompressed data doesn't match the original." << std::endl;
    }
}
// Parallel compression with decomposition that takes dynamic byte sizes as parameters
size_t snappyDecomposedParallel(const std::vector<uint8_t>& data, ProfilingInfo &pi,
                              std::vector<uint8_t>& compressedLeading,
                              std::vector<uint8_t>& compressedContent,
                              std::vector<uint8_t>& compressedTrailing,
                              size_t leadingBytes, size_t contentBytes, size_t trailingBytes,int numThreads) {
    std::vector<uint8_t> leading, content, trailing;
    splitBytesIntoComponents(data, leading, content, trailing, leadingBytes, contentBytes, trailingBytes);

    double compressedSize_l = 0, compressedSize_c = 0, compressedSize_t = 0;
    omp_set_num_threads(numThreads);

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            auto start = std::chrono::high_resolution_clock::now();
            compressedSize_l = compressWithSnappy(leading, compressedLeading);
            auto end = std::chrono::high_resolution_clock::now();
            pi.leading_time = std::chrono::duration<double>(end - start).count();
        }

        #pragma omp section
        {
            auto start = std::chrono::high_resolution_clock::now();
            compressedSize_c = compressWithSnappy(content, compressedContent);
            auto end = std::chrono::high_resolution_clock::now();
            pi.content_time = std::chrono::duration<double>(end - start).count();
        }

        #pragma omp section
        {
            auto start = std::chrono::high_resolution_clock::now();
            compressedSize_t = compressWithSnappy(trailing, compressedTrailing);
            auto end = std::chrono::high_resolution_clock::now();
            pi.trailing_time = std::chrono::duration<double>(end - start).count();
        }
    }

    pi.type = "Parallel Decomposition";
    return (compressedLeading.size() + compressedContent.size() + compressedTrailing.size());
}



// Decompression function with dynamic byte segmentation and OpenMP parallelization
void sanppyDecomposedParallelDecompression(const std::vector<uint8_t>& compressedLeading,
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
            decompressWithSnappy(compressedLeading, tempLeading, floatCount * leadingBytes);
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
            decompressWithSnappy(compressedContent, tempContent, floatCount * contentBytes);
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
            decompressWithSnappy(compressedTrailing, tempTrailing, floatCount * trailingBytes);
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

#endif //SNAPPY_PARALLEL_H
