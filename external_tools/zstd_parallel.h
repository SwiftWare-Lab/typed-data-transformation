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
#include <zstd.h>
#include <chrono>
#include <cstdint>
#include <omp.h>

#include "profiling_info.h"
#include <numeric>
#include <filesystem>




// Declare  as an external variable
extern std::vector<uint8_t> globalByteArray;


// Save compressed data to a file
void saveCompressedData(const std::string& filename, const std::vector<uint8_t>& compressedData) {
  // Specify the directory path
  std::string directory = "/home/samira/Documents/file";

  // Ensure the directory exists
  std::filesystem::create_directories(directory);

  // Prepend the directory path to the filename
  std::string fullPath = directory + "/" + filename;

  // Open the file in binary mode
  std::ofstream outFile(fullPath, std::ios::binary);
  if (!outFile) {
    std::cerr << "Error: Could not open file for writing: " << fullPath << std::endl;
    return;
  }

  // Write the compressed data to the file
  outFile.write(reinterpret_cast<const char*>(compressedData.data()), compressedData.size());
  outFile.close();
 // std::cout << "Compressed data saved to " << fullPath << std::endl;
}
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


// Compress with Zstd and enable debug logging
size_t compressWithZstd(const std::vector<uint8_t>& data, std::vector<uint8_t>& compressedData, int compressionLevel) {
    // Create a Zstd compression context
    ZSTD_CCtx* cctx = ZSTD_createCCtx();
    if (!cctx) {
        std::cerr << "Zstd compression error: Failed to create compression context" << std::endl;
        return 0;
    }

    // Set advanced compression parameters
    ZSTD_CCtx_setParameter(cctx, ZSTD_c_nbWorkers, 10); // Single-threaded for simplicity
    ZSTD_CCtx_setParameter(cctx, ZSTD_c_compressionLevel, compressionLevel);
    ZSTD_CCtx_setParameter(cctx, ZSTD_c_enableLongDistanceMatching, 1); // Enable long-distance matching
  //  ZSTD_CCtx_setParameter(cctx, ZSTD_c_format, ZSTD_f_zstd1);          // Zstd format

    // Enable verbose logging (debug level)
    //ZSTD_CCtx_setParameter(cctx, ZSTD_c_verbosity, 2); // 0: Silent, 1: Minimal, 2: Debug

    // Allocate compression buffer
    size_t const cBuffSize = ZSTD_compressBound(data.size());
    compressedData.resize(cBuffSize);

    // Perform compression with debug logging
    std::cout << "Starting Zstd compression with debug logging...\n";
    size_t const cSize = ZSTD_compress2(
        cctx,                            // Compression context
        compressedData.data(),           // Destination buffer
        cBuffSize,                       // Destination buffer size
        data.data(),                     // Source data
        data.size()                      // Source data size
    );

    // Check for errors
    if (ZSTD_isError(cSize)) {
        std::cerr << "Zstd compression error: " << ZSTD_getErrorName(cSize) << std::endl;
        ZSTD_freeCCtx(cctx); // Free the compression context
        return 0;
    }

    // Resize compressedData to the actual compressed size
    compressedData.resize(cSize);

    // Free the compression context
    ZSTD_freeCCtx(cctx);

    std::cout << "Compression complete. Compressed size: " << cSize << " bytes\n";

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
  std::cout << "File compressed  full: " <<compressedSize << "\nCompressed size: " << compressedSize << " bytes\n";
  if (compressedSize > 0) {

    saveCompressedData(std::to_string(33) + ".zst", compressedData);
  }

    pi.type = "Full Compression";
    return compressedSize;
}

// Full decompression without decomposition
void zstdDecompression(const std::vector<uint8_t>& compressedData, std::vector<uint8_t>& decompressedData, ProfilingInfo &pi) {

    decompressWithZstd(compressedData, decompressedData, globalByteArray.size());

    // Verify decompressed data
   //  if (!verifyDataMatch(globalByteArray, decompressedData)) {
   //    std::cerr << "Error: Decompressed data doesn't match the original." << std::endl;
   // }
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

    // Temporary buffer for the decompressed component(????????????????????)
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

  // // Verify decompressed data
   // if (!verifyDataMatch(globalByteArray, reconstructedData)) {
   //   std::cerr << "Error: Decompressed data doesn't match the original." << std::endl;
   // }
}

size_t zstdDecomposedSequential(const std::vector<uint8_t>& data, ProfilingInfo &pi,
                                std::vector<std::vector<uint8_t>>& compressedComponents,
                                const std::vector<size_t>& componentSizes) {
  // Split data into components
  std::vector<std::vector<uint8_t>> components(componentSizes.size());
  splitBytesIntoComponents(data, components, componentSizes,1);


  size_t compressedSizeTotal = 0;

  pi.component_times.assign(componentSizes.size(), 0.0);

  // Compress each component sequentially and record the compression time
  for (size_t i = 0; i < componentSizes.size(); ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    compressedSizeTotal += compressWithZstd(components[i], compressedComponents[i], 3);
    auto end = std::chrono::high_resolution_clock::now();
    pi.component_times[i] = std::chrono::duration<double>(end - start).count();
    // std::cout << "Component " << i << ": Original size = " << components[i].size()
    //       << ", Compressed size = " << compressedComponents[i].size() << std::endl;

  }

  pi.type = "Sequential Decomposition with 8 Components";
  std::cout << "compressedSizeTotal "<<compressedSizeTotal;
  return compressedSizeTotal;
}

size_t zstdDecomposedParallel(const std::vector<uint8_t>& data, ProfilingInfo &pi,
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
    size_t compressedSize = compressWithZstd(components[i], compressedComponents[i], 3);

    // Accumulate the total compressed size
    compressedSizeTotal += compressedSize;
    std::cout << "\nFile compressed : " <<compressedSize <<"com"<<i  << " bytes\n";
    std::cout << "File uncompressed : " <<components[i].size() <<"com"<<i <<" bytes\n";

    // Save the compressed component (thread-safe if `saveCompressedData` is safe)
    if (compressedSize > 0) {
      saveCompressedData(std::to_string(i) + ".zst", compressedComponents[i]);
    }


  }
  pi.type = "Parallel Decomposition with 8 Components";
  return compressedSizeTotal;
}



std::vector<uint8_t> zstdDecomposedParallelDecompression(
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
    auto start = std::chrono::high_resolution_clock::now();

    // Temporary buffer for the decompressed component
    std::vector<uint8_t> tempComponent(floatCount * componentBytes[compIdx]);

    // Decompress the current component
    decompressWithZstd(compressedComponents[compIdx], tempComponent, floatCount * componentBytes[compIdx]);

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

    auto end = std::chrono::high_resolution_clock::now();
    pi.component_times[compIdx] = std::chrono::duration<double>(end - start).count();
  }

  // // Verify decompressed data
  // if (!verifyDataMatch(globalByteArray, reconstructedData)) {
  //   std::cerr << "Error: Decompressed data doesn't match the original." << std::endl;
  // }

  return reconstructedData;
}


double calculateCompressionRatio(size_t originalSize, size_t compressedSize) {
  return static_cast<double>(originalSize) / static_cast<double>(compressedSize);
}


