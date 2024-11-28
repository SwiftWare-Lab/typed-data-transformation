//
// Created by samira on 11/4/24.
//

#ifndef ZSTD_PARALLEL_H
#define ZSTD_PARALLEL_H

#endif //ZSTD_PARALLEL_H
// Created by jamalids on 31/10/24.
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <zstd.h>
#include <chrono>
#include <cstdint>
#include <omp.h>
#include <cstring>
#include <vector>
#include <cxxopts.hpp>
#include "profiling_info.h"
#include <numeric>




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
                              std::vector<std::vector<uint8_t>>& components,
                              const std::vector<size_t>& componentSizes) {
  size_t numComponents = componentSizes.size();
  size_t totalBytes = std::accumulate(componentSizes.begin(), componentSizes.end(), 0);
  size_t numElements = byteArray.size() / totalBytes;

  // Resize components to hold the split data
  components.resize(numComponents);
  for (size_t i = 0; i < numComponents; ++i) {
    components[i].resize(numElements * componentSizes[i]);
  }

  // Use OpenMP to parallelize the component processing
#pragma omp parallel for
  for (size_t i = 0; i < numComponents; ++i) {
    size_t offset = std::accumulate(componentSizes.begin(), componentSizes.begin() + i, 0);
    for (size_t j = 0; j < numElements; ++j) {
      std::copy(byteArray.begin() + j * totalBytes + offset,
                byteArray.begin() + j * totalBytes + offset + componentSizes[i],
                components[i].begin() + j * componentSizes[i]);
    }
  }
}

void splitBytesIntoComponents1(const std::vector<uint8_t>& byteArray,
                              std::vector<std::vector<uint8_t>>& components,
                              const std::vector<size_t>& componentSizes) {
  size_t numComponents = componentSizes.size();
  size_t totalBytes = std::accumulate(componentSizes.begin(), componentSizes.end(), 0);
  size_t numElements = byteArray.size() / totalBytes;

  // Resize components and split the data
  components.resize(numComponents);
  size_t offset = 0;

  for (size_t i = 0; i < numComponents; ++i) {
    components[i].resize(numElements * componentSizes[i]);
    for (size_t j = 0; j < numElements; ++j) {
      std::copy(byteArray.begin() + j * totalBytes + offset,
                byteArray.begin() + j * totalBytes + offset + componentSizes[i],
                components[i].begin() + j * componentSizes[i]);
    }
    offset += componentSizes[i];
  }
}

// Compress with Zstd
size_t compressWithZstd(const std::vector<uint8_t>& data, std::vector<uint8_t>& compressedData, int compressionLevel) {
    size_t const cBuffSize = ZSTD_compressBound(data.size());
    compressedData.resize(cBuffSize);
    size_t const cSize = ZSTD_compress(compressedData.data(), cBuffSize, data.data(), data.size(), compressionLevel);
    if (ZSTD_isError(cSize)) {
        std::cerr << "Zstd compression error: " << ZSTD_getErrorName(cSize) << std::endl;
        return 0;
    }
    compressedData.resize(cSize);
    return cSize;
}

// Decompress with Zstd
size_t decompressWithZstd(const std::vector<uint8_t>& compressedData, std::vector<uint8_t>& decompressedData, size_t originalSize) {
    decompressedData.resize(originalSize);
    size_t const dSize = ZSTD_decompress(decompressedData.data(), originalSize, compressedData.data(), compressedData.size());
    if (ZSTD_isError(dSize)) {
        std::cerr << "Zstd decompression error: " << ZSTD_getErrorName(dSize) << std::endl;
        return 0;
    }
    return dSize;
}
// Full compression without decomposition
size_t zstdCompression(const std::vector<uint8_t>& data, ProfilingInfo &pi, std::vector<uint8_t>& compressedData) {

    size_t compressedSize = compressWithZstd(data, compressedData, 3);

    pi.type = "Full Compression";
    return compressedSize;
}

// Full decompression without decomposition
void zstdDecompression(const std::vector<uint8_t>& compressedData, std::vector<uint8_t>& decompressedData, ProfilingInfo &pi) {

    decompressWithZstd(compressedData, decompressedData, globalByteArray.size());

    // Verify decompressed data
    if (!verifyDataMatch(globalByteArray, decompressedData)) {
        std::cerr << "Error: Decompressed data doesn't match the original." << std::endl;
    }
}

void zstdDecomposedSequentialDecompression(const std::vector<std::vector<uint8_t>>& compressedComponents,
                                           ProfilingInfo &pi,
                                           const std::vector<size_t>& componentBytes) {
  size_t totalSize = globalByteArray.size();
  size_t numComponents = componentBytes.size();
  size_t totalBytesPerElement = std::accumulate(componentBytes.begin(), componentBytes.end(), 0);
  size_t floatCount = totalSize / totalBytesPerElement;
  std::vector<uint8_t> reconstructedData(totalSize);

  size_t baseOffset = 0;

  for (size_t compIdx = 0; compIdx < numComponents; ++compIdx) {
    auto start = std::chrono::high_resolution_clock::now();

    // Temporary buffer for the decompressed component
    std::vector<uint8_t> tempComponent(floatCount * componentBytes[compIdx]);

    // Decompress the current component
    decompressWithZstd(compressedComponents[compIdx], tempComponent, floatCount * componentBytes[compIdx]);

    // Reassemble the decompressed data
    for (size_t i = 0; i < floatCount; ++i) {
      std::copy(tempComponent.begin() + i * componentBytes[compIdx],
                tempComponent.begin() + (i + 1) * componentBytes[compIdx],
                reconstructedData.begin() + i * totalBytesPerElement + baseOffset);
    }

    auto end = std::chrono::high_resolution_clock::now();

    // Record the decompression time for the current component
    pi.component_times[compIdx] = std::chrono::duration<double>(end - start).count();

    // Update the base offset for the next component
    baseOffset += componentBytes[compIdx];
  }

  // Verify decompressed data
  if (!verifyDataMatch(globalByteArray, reconstructedData)) {
    std::cerr << "Error: Decompressed data doesn't match the original." << std::endl;
  }
}

size_t zstdDecomposedSequential(const std::vector<uint8_t>& data, ProfilingInfo &pi,
                                std::vector<std::vector<uint8_t>>& compressedComponents,
                                const std::vector<size_t>& componentSizes) {
  // Split data into components
  std::vector<std::vector<uint8_t>> components(8);
  splitBytesIntoComponents(data, components, componentSizes);

  size_t compressedSizeTotal = 0;
  pi.component_times.resize(8); // Ensure the vector can hold 8 component times

  // Compress each component sequentially and record the compression time
  for (size_t i = 0; i < 8; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    compressedSizeTotal += compressWithZstd(components[i], compressedComponents[i], 3);
    auto end = std::chrono::high_resolution_clock::now();
    pi.component_times[i] = std::chrono::duration<double>(end - start).count();
  }

  pi.type = "Sequential Decomposition with 8 Components";
  return compressedSizeTotal;
}

size_t zstdDecomposedParallel(const std::vector<uint8_t>& data, ProfilingInfo &pi,
                              std::vector<std::vector<uint8_t>>& compressedComponents,
                              const std::vector<size_t>& componentSizes, int numThreads) {
  // Split data into components
  std::vector<std::vector<uint8_t>> components(8);
  splitBytesIntoComponents(data, components, componentSizes);

  omp_set_num_threads(numThreads);

  size_t compressedSizeTotal = 0;
  pi.component_times.resize(8); // Ensure the vector can hold 8 component times

#pragma omp parallel for
  for (size_t i = 0; i < 8; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    compressedSizeTotal += compressWithZstd(components[i], compressedComponents[i], 3);
    auto end = std::chrono::high_resolution_clock::now();
    pi.component_times[i] = std::chrono::duration<double>(end - start).count();
  }

  pi.type = "Parallel Decomposition with 8 Components";
  return compressedSizeTotal;
}

void zstdDecomposedParallelDecompression(const std::vector<std::vector<uint8_t>>& compressedComponents,
                                         ProfilingInfo &pi,
                                         const std::vector<size_t>& componentBytes,
                                         int numThreads) {
  size_t totalSize = globalByteArray.size();
  size_t numComponents = componentBytes.size();
  size_t totalBytesPerElement = std::accumulate(componentBytes.begin(), componentBytes.end(), 0);
  size_t floatCount = totalSize / totalBytesPerElement;
  std::vector<uint8_t> reconstructedData(totalSize);

  pi.component_times.resize(numComponents); // Ensure space for component times
  omp_set_num_threads(numThreads);

#pragma omp parallel for
  for (size_t compIdx = 0; compIdx < numComponents; ++compIdx) {
    auto start = std::chrono::high_resolution_clock::now();

    // Temporary buffer for the decompressed component
    std::vector<uint8_t> tempComponent(floatCount * componentBytes[compIdx]);

    // Decompress the current component
    decompressWithZstd(compressedComponents[compIdx], tempComponent, floatCount * componentBytes[compIdx]);

    // Calculate the base offset for this component
    size_t baseOffset = std::accumulate(componentBytes.begin(), componentBytes.begin() + compIdx, 0);

    // Reassemble the decompressed data
    for (size_t i = 0; i < floatCount; ++i) {
      std::copy(tempComponent.begin() + i * componentBytes[compIdx],
                tempComponent.begin() + (i + 1) * componentBytes[compIdx],
                reconstructedData.begin() + i * totalBytesPerElement + baseOffset);
    }

    auto end = std::chrono::high_resolution_clock::now();
    pi.component_times[compIdx] = std::chrono::duration<double>(end - start).count();
  }

  // Verify decompressed data
  if (!verifyDataMatch(globalByteArray, reconstructedData)) {
    std::cerr << "Error: Decompressed data doesn't match the original." << std::endl;
  }
}



double calculateCompressionRatio(size_t originalSize, size_t compressedSize) {
  return static_cast<double>(originalSize) / static_cast<double>(compressedSize);
}


