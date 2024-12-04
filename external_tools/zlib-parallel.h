//
// Created by jamalids on 04/12/24.
//

#ifndef ZLIB_PARALLEL_H
#define ZLIB_PARALLEL_H
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <zlib.h>
#include <cstdint>
#include <omp.h>
#include "profiling_info.h"
#include <numeric>

// Declare  as an external variable
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
                              const std::vector<size_t>& componentSizes,int numThreads) {
  size_t numComponents = componentSizes.size();
  size_t totalBytes = std::accumulate(componentSizes.begin(), componentSizes.end(), 0);
  size_t numElements = byteArray.size() / totalBytes;

  // Resize components to hold the split data
  components.resize(numComponents);
  for (size_t i = 0; i < numComponents; ++i) {
    components[i].resize(numElements * componentSizes[i]);
  }

  // Use OpenMP to parallelize the component processing
#pragma omp parallel for num_threads(numThreads)
  for (size_t i = 0; i < numComponents; ++i) {
    size_t offset = std::accumulate(componentSizes.begin(), componentSizes.begin() + i, 0);
    for (size_t j = 0; j < numElements; ++j) {
      std::copy(byteArray.begin() + j * totalBytes + offset,
                byteArray.begin() + j * totalBytes + offset + componentSizes[i],
                components[i].begin() + j * componentSizes[i]);
    }
  }
}


// Compress with zlib

size_t compressWithzlib(const std::vector<uint8_t>& data, std::vector<uint8_t>& compressedData, int compressionLevel) {
  if (compressionLevel < Z_NO_COMPRESSION || compressionLevel > Z_BEST_COMPRESSION) {
    throw std::invalid_argument("Invalid compression level. Must be between Z_NO_COMPRESSION (0) and Z_BEST_COMPRESSION (9).");
  }

  // Calculate the upper bound for the compressed data size
  uLong bound = compressBound(data.size());

  // Resize the compressedData vector to hold the compressed data
  compressedData.resize(bound);

  // Perform compression using compress2
  int ret = compress2(
      compressedData.data(),       // Destination buffer
      &bound,                      // Pointer to the size of the destination buffer
      data.data(),                 // Source data
      data.size(),                 // Size of the source data
      compressionLevel             // Compression level
  );

  if (ret != Z_OK) {
    // Handle different error codes
    switch (ret) {
    case Z_MEM_ERROR:
      throw std::runtime_error("Compression failed: Not enough memory.");
    case Z_BUF_ERROR:
      throw std::runtime_error("Compression failed: Output buffer was not large enough.");
    case Z_STREAM_ERROR:
      throw std::runtime_error("Compression failed: Invalid compression level.");
    default:
      throw std::runtime_error("Compression failed: Unknown error.");
    }
  }

  // Resize the compressedData vector to the actual compressed size
  compressedData.resize(bound);

  return bound; // Return the size of the compressed data
}

// Decompress with gzip (zlib)
size_t decompressWithzlib(const std::vector<uint8_t>& compressedData, std::vector<uint8_t>& decompressedData, size_t originalSize) {
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
size_t zlibCompression(const std::vector<uint8_t>& data, ProfilingInfo &pi, std::vector<uint8_t>& compressedData) {
    size_t compressedSize = compressWithzlib(data, compressedData, 6);
    pi.type = "Full Compression (Gzip)";
    return compressedSize;
}

// Full decompression without decomposition
void zlibDecompression(const std::vector<uint8_t>& compressedData, std::vector<uint8_t>& decompressedData, ProfilingInfo &pi) {
    decompressWithzlib(compressedData, decompressedData, globalByteArray.size());

    // Verify decompressed data
    if (!verifyDataMatch(globalByteArray, decompressedData)) {
        std::cerr << "Error: Decompressed data doesn't match the original." << std::endl;
    }
}

// Sequential compression with decomposition
size_t zlibDecomposedSequential(const std::vector<uint8_t>& data, ProfilingInfo &pi,
                                std::vector<std::vector<uint8_t>>& compressedComponents,
                                const std::vector<size_t>& componentSizes) {
  // Split data into components
  std::vector<std::vector<uint8_t>> components(componentSizes.size());
  splitBytesIntoComponents(data, components, componentSizes,1);

  size_t compressedSizeTotal = 0;

  pi.component_times.assign(componentSizes.size(), 0.0);

  // Compress each component sequentially and record the compression time
  for (size_t i = 0; i < componentSizes.size(); ++i) {
   // compressedSizeTotal += compressWithzlib(components[i], compressedComponents[i],Z_BEST_COMPRESSION);
    compressedSizeTotal += compressWithzlib(components[i], compressedComponents[i],6);

  }
  return compressedSizeTotal;

}

// Sequential decompression with decomposition
void zlibDecomposedSequentialDecompression(const std::vector<std::vector<uint8_t>>& compressedComponents,
                                           ProfilingInfo &pi,
                                           const std::vector<size_t>& componentBytes) {
  size_t totalSize = globalByteArray.size();
  size_t numComponents = componentBytes.size();
  size_t totalBytesPerElement = std::accumulate(componentBytes.begin(), componentBytes.end(), 0);
  size_t floatCount = totalSize / totalBytesPerElement;
  std::vector<uint8_t> reconstructedData(totalSize);

  size_t baseOffset = 0;

  for (size_t compIdx = 0; compIdx < numComponents; ++compIdx) {

    std::vector<uint8_t> tempComponent(floatCount * componentBytes[compIdx]);

    // Decompress the current component
    decompressWithzlib(compressedComponents[compIdx], tempComponent, floatCount * componentBytes[compIdx]);

    // Reassemble the decompressed data
    for (size_t i = 0; i < floatCount; ++i) {
      std::copy(tempComponent.begin() + i * componentBytes[compIdx],
                tempComponent.begin() + (i + 1) * componentBytes[compIdx],
                reconstructedData.begin() + i * totalBytesPerElement + baseOffset);
    }

    baseOffset += componentBytes[compIdx];
  }

  // Verify decompressed data
  if (!verifyDataMatch(globalByteArray, reconstructedData)) {
    std::cerr << "Error: Decompressed data doesn't match the original." << std::endl;
  }
}
// Parallel compression with decomposition

size_t zlibDecomposedParallel(const std::vector<uint8_t>& data, ProfilingInfo &pi,
                              std::vector<std::vector<uint8_t>>& compressedComponents,
                              const std::vector<size_t>& componentSizes, int numThreads) {
  // Split data into components
  std::vector<std::vector<uint8_t>> components(componentSizes.size());
  splitBytesIntoComponents(data, components, componentSizes, numThreads);

  // omp_set_num_threads(numThreads);

  size_t compressedSizeTotal = 0;

  pi.component_times.assign(componentSizes.size(), 0.0);

  // #pragma omp parallel  for num_threads(numThreads)
#pragma omp parallel for schedule(dynamic) num_threads(numThreads)
  for (size_t i = 0; i < componentSizes.size(); ++i) {

    compressedSizeTotal += compressWithzlib(components[i], compressedComponents[i],6);


  }

  return compressedSizeTotal;
}

std::vector<uint8_t> zlibDecomposedParallelDecompression(
    const std::vector<std::vector<uint8_t>>& compressedComponents,
    ProfilingInfo& pi,
    const std::vector<size_t>& componentBytes,
    int numThreads) {
  size_t totalSize = globalByteArray.size(); // Size of the original data
  size_t numComponents = componentBytes.size();
  size_t totalBytesPerElement = std::accumulate(componentBytes.begin(), componentBytes.end(), 0);
  size_t floatCount = totalSize / totalBytesPerElement;
  std::vector<uint8_t> reconstructedData(totalSize);

  pi.component_times.resize(numComponents);
  omp_set_num_threads(numThreads);

#pragma omp parallel for
  for (size_t compIdx = 0; compIdx < numComponents; ++compIdx) {


    // Temporary buffer for the decompressed component
    std::vector<uint8_t> tempComponent(floatCount * componentBytes[compIdx]);

    // Decompress the current component
    decompressWithzlib(compressedComponents[compIdx], tempComponent, floatCount * componentBytes[compIdx]);

    // Calculate the base offset for this component
    size_t baseOffset = std::accumulate(componentBytes.begin(), componentBytes.begin() + compIdx, 0);

    // Reassemble the decompressed data with unrolling
    size_t comByteComp = componentBytes[compIdx];

#pragma omp parallel for schedule(static, 100000)
    for (size_t i = 0; i < floatCount; ++i) {
      size_t baseIndex = i * totalBytesPerElement + baseOffset;
      size_t tempIndex = i * comByteComp;

#pragma omp simd
      for (size_t j = 0; j < comByteComp; ++j) {
        reconstructedData[baseIndex + j] = tempComponent[tempIndex + j];
      }
    }


  }

  // Verify decompressed data
  if (!verifyDataMatch(globalByteArray, reconstructedData)) {
    std::cerr << "Error: Decompressed data doesn't match the original." << std::endl;
  }

  return reconstructedData;
}


double calculateCompressionRatio(size_t originalSize, size_t compressedSize) {
  return static_cast<double>(originalSize) / static_cast<double>(compressedSize);
}

#endif //ZLIB_PARALLEL_H
