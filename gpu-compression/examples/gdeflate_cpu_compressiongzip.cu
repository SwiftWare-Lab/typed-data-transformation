/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or its affiliates
 * is strictly prohibited.
 */

// Include BatchData header (make sure this is in your include path)
#include "BatchData.h"

// nvCOMP headers for gdeflate (existing)
#include <nvcomp/native/gdeflate_cpu.h>
#include <nvcomp/gdeflate.h>

// NEW: Additional headers for gzip
#include "zlib.h"
#include "nvcomp/gzip.h"

// STL and other headers
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <iomanip>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <chrono>
#include <cstring>
#ifdef _OPENMP
#include <omp.h>
#endif

// (Assume CUDA_CHECK is defined elsewhere to check CUDA API return codes)

// -----------------------------------------------------------------------------
// Dataset Loading and Conversion Functions (for TSV files)
// -----------------------------------------------------------------------------
std::pair<std::vector<float>, size_t> loadTSVDataset(const std::string& filePath) {
  std::vector<float> floatArray;
  std::ifstream file(filePath);
  std::string line;
  size_t rowCount = 0;
  if (file.is_open()) {
    while (std::getline(file, line)) {
      std::stringstream ss(line);
      std::string value;
      std::getline(ss, value, '\t'); // skip first column
      while (std::getline(ss, value, '\t')) {
        floatArray.push_back(std::stof(value));
      }
      ++rowCount;
    }
    file.close();
  } else {
    std::cerr << "Unable to open file: " << filePath << std::endl;
  }
  return std::make_pair(floatArray, rowCount);
}

std::pair<std::vector<double>, size_t> loadTSVDatasetdouble(const std::string& filePath) {
  std::vector<double> doubleArray;
  std::ifstream file(filePath);
  std::string line;
  size_t rowCount = 0;
  if (file.is_open()) {
    while (std::getline(file, line)) {
      std::stringstream ss(line);
      std::string value;
      std::getline(ss, value, '\t'); // skip first column
      while (std::getline(ss, value, '\t')) {
        doubleArray.push_back(std::stod(value));
      }
      ++rowCount;
    }
    file.close();
  } else {
    std::cerr << "Unable to open file: " << filePath << std::endl;
  }
  return std::make_pair(doubleArray, rowCount);
}

std::vector<uint8_t> convertFloatToBytes(const std::vector<float>& floatArray) {
  std::vector<uint8_t> byteArray(floatArray.size() * sizeof(float));
  for (size_t i = 0; i < floatArray.size(); i++) {
    const uint8_t* floatBytes = reinterpret_cast<const uint8_t*>(&floatArray[i]);
    for (size_t j = 0; j < sizeof(float); j++) {
      byteArray[i * sizeof(float) + j] = floatBytes[j];
    }
  }
  return byteArray;
}

std::vector<uint8_t> convertDoubleToBytes(const std::vector<double>& doubleArray) {
  std::vector<uint8_t> byteArray(doubleArray.size() * sizeof(double));
  for (size_t i = 0; i < doubleArray.size(); i++) {
    const uint8_t* doubleBytes = reinterpret_cast<const uint8_t*>(&doubleArray[i]);
    for (size_t j = 0; j < sizeof(double); j++) {
      byteArray[i * sizeof(double) + j] = doubleBytes[j];
    }
  }
  return byteArray;
}

// -----------------------------------------------------------------------------
// Helper: Convert Component Configuration to String
// -----------------------------------------------------------------------------
std::string configToString(const std::vector<std::vector<size_t>>& config) {
  std::stringstream ss;
  ss << "\"";
  for (size_t i = 0; i < config.size(); i++) {
    ss << "[";
    for (size_t j = 0; j < config[i].size(); j++) {
      ss << config[i][j];
      if (j < config[i].size() - 1)
        ss << ",";
    }
    ss << "]";
    if (i < config.size() - 1)
      ss << ",";
  }
  ss << "\"";
  return ss.str();
}

// -----------------------------------------------------------------------------
// Component-Based Decomposition Functions
// -----------------------------------------------------------------------------
inline void splitBytesIntoComponentsNested(
    const std::vector<uint8_t>& byteArray,
    std::vector<std::vector<uint8_t>>& outputComponents,
    const std::vector<std::vector<size_t>>& allComponentSizes,
    int numThreads)
{
  size_t totalBytesPerElement = 0;
  for (const auto &group : allComponentSizes)
    totalBytesPerElement += group.size();
  size_t numElements = byteArray.size() / totalBytesPerElement;
  outputComponents.resize(allComponentSizes.size());
  for (size_t i = 0; i < allComponentSizes.size(); i++) {
    outputComponents[i].resize(numElements * allComponentSizes[i].size());
  }
  #pragma omp parallel for num_threads(numThreads)
  for (size_t elem = 0; elem < numElements; elem++) {
    for (size_t compIdx = 0; compIdx < allComponentSizes.size(); compIdx++) {
      const auto &groupIndices = allComponentSizes[compIdx];
      size_t groupSize = groupIndices.size();
      size_t writePos = elem * groupSize;
      for (size_t sub = 0; sub < groupSize; sub++) {
        size_t idxInElem = groupIndices[sub] - 1; // 1-based to 0-based
        size_t globalSrcIdx = elem * totalBytesPerElement + idxInElem;
        outputComponents[compIdx][writePos + sub] = byteArray[globalSrcIdx];
      }
    }
  }
}

inline void reassembleBytesFromComponentsNestedlz4(
    const std::vector<std::vector<uint8_t>>& inputComponents,
    uint8_t* byteArray,
    size_t byteArraySize,
    const std::vector<std::vector<size_t>>& allComponentSizes,
    int numThreads)
{
  size_t totalBytesPerElement = 0;
  for (const auto &group : allComponentSizes)
    totalBytesPerElement += group.size();
  size_t numElements = byteArraySize / totalBytesPerElement;
  #pragma omp parallel for num_threads(numThreads)
  for (size_t compIdx = 0; compIdx < inputComponents.size(); compIdx++) {
    const auto &groupIndices = allComponentSizes[compIdx];
    const auto &componentData = inputComponents[compIdx];
    size_t groupSize = groupIndices.size();
    for (size_t elem = 0; elem < numElements; elem++) {
      size_t readPos = elem * groupSize;
      for (size_t sub = 0; sub < groupSize; sub++) {
        size_t idxInElem = groupIndices[sub] - 1;
        size_t globalIndex = elem * totalBytesPerElement + idxInElem;
        byteArray[globalIndex] = componentData[readPos + sub];
      }
    }
  }
}

// -----------------------------------------------------------------------------
// (A) Whole-Dataset Compression/Decompression (gdeflate mode)
// -----------------------------------------------------------------------------
struct WholeDatasetMetrics {
  size_t totalBytes;
  size_t compBytes;
  double compressionRatio;
  float compTimeMs;
  float decompTimeMs;
  double compThroughputGBs;
  double decompThroughputGBs;
};

WholeDatasetMetrics run_whole_dataset(const std::vector<std::vector<char>>& files) {
  WholeDatasetMetrics metrics;
  size_t total_bytes = 0;
  for (const auto &part : files)
    total_bytes += part.size();
  metrics.totalBytes = total_bytes;
  std::cout << "----------" << std::endl;
  std::cout << "Whole-dataset mode (gdeflate):" << std::endl;
  std::cout << "Files: " << files.size() << std::endl;
  std::cout << "Uncompressed (B): " << total_bytes << std::endl;

  const size_t chunk_size = 1 << 16;
  BatchDataCPU input_data_cpu(files, chunk_size);
  std::cout << "Chunks: " << input_data_cpu.size() << std::endl;

  // Compression Phase
  cudaEvent_t comp_start, comp_end;
  CUDA_CHECK(cudaEventCreate(&comp_start));
  CUDA_CHECK(cudaEventCreate(&comp_end));
  CUDA_CHECK(cudaEventRecord(comp_start, 0));
  nvcompStatus_t status;
  size_t max_out_bytes;
  status = nvcompBatchedGdeflateCompressGetMaxOutputChunkSize(chunk_size, nvcompBatchedGdeflateDefaultOpts, &max_out_bytes);
  if (status != nvcompSuccess)
    throw std::runtime_error("ERROR: Failed to get max output chunk size");
  BatchDataCPU compress_data_cpu(max_out_bytes, input_data_cpu.size());
  gdeflate::compressCPU(input_data_cpu.ptrs(), input_data_cpu.sizes(), chunk_size, input_data_cpu.size(),
                         compress_data_cpu.ptrs(), compress_data_cpu.sizes());
  CUDA_CHECK(cudaEventRecord(comp_end, 0));
  CUDA_CHECK(cudaEventSynchronize(comp_end));
  float comp_time_ms;
  CUDA_CHECK(cudaEventElapsedTime(&comp_time_ms, comp_start, comp_end));
  metrics.compTimeMs = comp_time_ms;
  metrics.compThroughputGBs = ((double)total_bytes / comp_time_ms) * 1e-6;
  CUDA_CHECK(cudaEventDestroy(comp_start));
  CUDA_CHECK(cudaEventDestroy(comp_end));

  size_t* compressed_sizes_host = compress_data_cpu.sizes();
  size_t comp_bytes = 0;
  for (size_t i = 0; i < compress_data_cpu.size(); ++i)
    comp_bytes += compressed_sizes_host[i];
  metrics.compBytes = comp_bytes;
  metrics.compressionRatio = (double)total_bytes / comp_bytes;
  std::cout << "Compressed size (B): " << comp_bytes << ", Compression ratio: "
            << std::fixed << std::setprecision(2) << metrics.compressionRatio << std::endl;

  // Decompression Phase
  BatchData compress_data(compress_data_cpu, true);
  BatchData decomp_data(input_data_cpu, false);
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  cudaEvent_t decomp_start, decomp_end;
  CUDA_CHECK(cudaEventCreate(&decomp_start));
  CUDA_CHECK(cudaEventCreate(&decomp_end));
  size_t decomp_temp_bytes;
  status = nvcompBatchedGdeflateDecompressGetTempSize(compress_data.size(), chunk_size, &decomp_temp_bytes);
  if (status != nvcompSuccess)
    throw std::runtime_error("ERROR: Failed to get decompression temp size");
  void* d_decomp_temp;
  CUDA_CHECK(cudaMalloc(&d_decomp_temp, decomp_temp_bytes));
  size_t* d_decomp_sizes;
  CUDA_CHECK(cudaMalloc(&d_decomp_sizes, decomp_data.size() * sizeof(size_t)));
  nvcompStatus_t* d_statuses;
  CUDA_CHECK(cudaMalloc(&d_statuses, decomp_data.size() * sizeof(nvcompStatus_t)));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  status = nvcompBatchedGdeflateDecompressAsync(compress_data.ptrs(), compress_data.sizes(),
                                                decomp_data.sizes(), d_decomp_sizes,
                                                compress_data.size(), d_decomp_temp, decomp_temp_bytes,
                                                decomp_data.ptrs(), d_statuses, stream);
  if (status != nvcompSuccess)
    throw std::runtime_error("ERROR: Decompression failed");
  if (!(input_data_cpu == decomp_data))
    throw std::runtime_error("ERROR: Decompressed data does not match input");
  else
    std::cout << "Decompression validated :)" << std::endl;
  CUDA_CHECK(cudaEventRecord(decomp_start, stream));
  status = nvcompBatchedGdeflateDecompressAsync(compress_data.ptrs(), compress_data.sizes(),
                                                decomp_data.sizes(), d_decomp_sizes,
                                                compress_data.size(), d_decomp_temp, decomp_temp_bytes,
                                                decomp_data.ptrs(), d_statuses, stream);
  CUDA_CHECK(cudaEventRecord(decomp_end, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  float decomp_time_ms;
  CUDA_CHECK(cudaEventElapsedTime(&decomp_time_ms, decomp_start, decomp_end));
  metrics.decompTimeMs = decomp_time_ms;
  metrics.decompThroughputGBs = ((double)total_bytes / decomp_time_ms) * 1e-6;
  CUDA_CHECK(cudaFree(d_decomp_temp));
  CUDA_CHECK(cudaFree(d_decomp_sizes));
  CUDA_CHECK(cudaFree(d_statuses));
  CUDA_CHECK(cudaEventDestroy(decomp_start));
  CUDA_CHECK(cudaEventDestroy(decomp_end));
  CUDA_CHECK(cudaStreamDestroy(stream));
  std::cout << "Whole-dataset compression throughput (GB/s): " << metrics.compThroughputGBs << std::endl;
  std::cout << "Whole-dataset decompression throughput (GB/s): " << metrics.decompThroughputGBs << std::endl;
  return metrics;
}

// -----------------------------------------------------------------------------
// (C) Component-Based Compression/Decompression (gdeflate mode)
// -----------------------------------------------------------------------------
struct ComponentResult {
  size_t compSize;
  std::vector<char> compData;
  float compTimeMs;
  float decompTimeMs;
};

ComponentResult compress_decompress_component(const std::vector<char>& compData) {
  ComponentResult result;
  result.compData = compData;
  size_t comp_total_bytes = compData.size();
  std::cout << "Component uncompressed size: " << comp_total_bytes << " bytes" << std::endl;
  std::vector<std::vector<char>> files = { compData };
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  cudaEvent_t comp_comp_start, comp_comp_end;
  CUDA_CHECK(cudaEventCreate(&comp_comp_start));
  CUDA_CHECK(cudaEventCreate(&comp_comp_end));
  CUDA_CHECK(cudaEventRecord(comp_comp_start, stream));
  const size_t chunk_size = 1 << 16;
  BatchDataCPU input_data_cpu(files, chunk_size);
  nvcompStatus_t status;
  size_t max_out_bytes;
  status = nvcompBatchedGdeflateCompressGetMaxOutputChunkSize(chunk_size, nvcompBatchedGdeflateDefaultOpts, &max_out_bytes);
  if (status != nvcompSuccess)
    throw std::runtime_error("Error: Failed to get max output size for component");
  BatchDataCPU compress_data_cpu(max_out_bytes, input_data_cpu.size());
  gdeflate::compressCPU(input_data_cpu.ptrs(), input_data_cpu.sizes(), chunk_size, input_data_cpu.size(),
                         compress_data_cpu.ptrs(), compress_data_cpu.sizes());
  CUDA_CHECK(cudaEventRecord(comp_comp_end, stream));
  CUDA_CHECK(cudaEventSynchronize(comp_comp_end));
  float comp_comp_time_ms;
  CUDA_CHECK(cudaEventElapsedTime(&comp_comp_time_ms, comp_comp_start, comp_comp_end));
  result.compTimeMs = comp_comp_time_ms;
  double comp_comp_throughput = ((double)comp_total_bytes / comp_comp_time_ms) * 1e-6;
  std::cout << "Component compression throughput (GB/s): " << comp_comp_throughput << std::endl;
  CUDA_CHECK(cudaEventDestroy(comp_comp_start));
  CUDA_CHECK(cudaEventDestroy(comp_comp_end));
  size_t* compressed_sizes_host = compress_data_cpu.sizes();
  size_t compSize = 0;
  for (size_t i = 0; i < input_data_cpu.size(); i++)
    compSize += compressed_sizes_host[i];
  result.compSize = compSize;
  std::cout << "Component compressed size: " << compSize << " bytes" << std::endl;
  if (compSize >= comp_total_bytes) {
    std::cout << "Compressed size (" << compSize << " bytes) is not smaller than uncompressed size ("
              << comp_total_bytes << " bytes). Skipping decompression." << std::endl;
    result.decompTimeMs = 0.0f;
    CUDA_CHECK(cudaStreamDestroy(stream));
    return result;
  }
  BatchData compress_data(compress_data_cpu, true);
  BatchData decomp_data(input_data_cpu, false);
  size_t decomp_temp_bytes;
  status = nvcompBatchedGdeflateDecompressGetTempSize(compress_data.size(), chunk_size, &decomp_temp_bytes);
  if (status != nvcompSuccess)
    throw std::runtime_error("Error: Failed to get temp decompression size for component");
  void* d_decomp_temp;
  CUDA_CHECK(cudaMalloc(&d_decomp_temp, decomp_temp_bytes));
  size_t* d_decomp_sizes;
  CUDA_CHECK(cudaMalloc(&d_decomp_sizes, decomp_data.size() * sizeof(size_t)));
  nvcompStatus_t* d_statuses;
  CUDA_CHECK(cudaMalloc(&d_statuses, decomp_data.size() * sizeof(nvcompStatus_t)));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  status = nvcompBatchedGdeflateDecompressAsync(compress_data.ptrs(), compress_data.sizes(),
                                                decomp_data.sizes(), d_decomp_sizes,
                                                compress_data.size(), d_decomp_temp,
                                                decomp_temp_bytes, decomp_data.ptrs(),
                                                d_statuses, stream);
  if (status != nvcompSuccess)
    throw std::runtime_error("Error: Component decompression validation failed");
  CUDA_CHECK(cudaStreamSynchronize(stream));
  cudaEvent_t comp_decomp_start, comp_decomp_end;
  CUDA_CHECK(cudaEventCreate(&comp_decomp_start));
  CUDA_CHECK(cudaEventCreate(&comp_decomp_end));
  CUDA_CHECK(cudaEventRecord(comp_decomp_start, stream));
  status = nvcompBatchedGdeflateDecompressAsync(compress_data.ptrs(), compress_data.sizes(),
                                                decomp_data.sizes(), d_decomp_sizes,
                                                compress_data.size(), d_decomp_temp,
                                                decomp_temp_bytes, decomp_data.ptrs(),
                                                d_statuses, stream);
  CUDA_CHECK(cudaEventRecord(comp_decomp_end, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  float comp_decomp_time_ms;
  CUDA_CHECK(cudaEventElapsedTime(&comp_decomp_time_ms, comp_decomp_start, comp_decomp_end));
  result.decompTimeMs = comp_decomp_time_ms;
  double comp_decomp_throughput = ((double)comp_total_bytes / comp_decomp_time_ms) * 1e-6;
  std::cout << "Component decompression throughput (GB/s): " << comp_decomp_throughput << std::endl;
  CUDA_CHECK(cudaEventDestroy(comp_decomp_start));
  CUDA_CHECK(cudaEventDestroy(comp_decomp_end));
  CUDA_CHECK(cudaFree(d_decomp_temp));
  CUDA_CHECK(cudaFree(d_decomp_sizes));
  CUDA_CHECK(cudaFree(d_statuses));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return result;
}

// -----------------------------------------------------------------------------
// (D) Whole-Dataset and Component-Based GZIP Modes (New)
// -----------------------------------------------------------------------------
struct WholeDatasetGzipMetrics {
  size_t totalBytes;
  size_t compBytes;
  double compressionRatio;
  float compTimeMs;
  float decompTimeMs;
  double compThroughputGBs;
  double decompThroughputGBs;
};

struct ComponentGzipMetrics {
  std::string configStr;
  size_t totalUncompressed;
  size_t totalCompressed;
  float overallCompTimeMs;
  float overallDecompTimeMs;
  double compThroughputGBs;
  double decompThroughputGBs;
  double compressionRatio;
};
WholeDatasetGzipMetrics run_whole_dataset_gzip(const std::vector<std::vector<char>>& files) {
  WholeDatasetGzipMetrics metrics;
  const size_t chunk_size = 1 << 16; // 64 KB

  // Step 1: Create a padded version of the input files (each file size becomes a multiple of chunk_size)
  std::vector<std::vector<char>> paddedFiles = files;
  for (auto &file : paddedFiles) {
    size_t remainder = file.size() % chunk_size;
    if (remainder != 0) {
      size_t pad = chunk_size - remainder;
      file.insert(file.end(), pad, 0); // pad with zeros
    }
  }
  // Create a CPU batch from the padded files.
  BatchDataCPU input_data_cpu(paddedFiles, chunk_size);

  // Compute total original (unpadded) bytes.
  size_t total_bytes = 0;
  for (const auto &file : files)
    total_bytes += file.size();
  metrics.totalBytes = total_bytes;

  // Step 2: CPU Compression using zlib in GZIP mode.
  size_t numChunks = input_data_cpu.size();
  std::vector<std::vector<char>> gzipCompressedChunks(numChunks);
  std::vector<size_t> compSizes(numChunks);
  for (size_t i = 0; i < numChunks; i++) {
    const char* src = reinterpret_cast<const char*>(input_data_cpu.ptrs()[i]);
    size_t srcSize = input_data_cpu.sizes()[i]; // Should equal chunk_size (possibly with extra zeros)
    uLongf compBufferSize = compressBound(srcSize);
    std::vector<char> compBuffer(compBufferSize);

    // Set up z_stream for gzip compression (using deflateInit2 with windowBits 15|16)
    z_stream zs;
    memset(&zs, 0, sizeof(zs));
    zs.next_in = reinterpret_cast<Bytef*>(const_cast<char*>(src));
    zs.avail_in = srcSize;
    zs.next_out = reinterpret_cast<Bytef*>(compBuffer.data());
    zs.avail_out = compBufferSize;
    int ret = deflateInit2(&zs, 9, Z_DEFLATED, 15 | 16, 8, Z_DEFAULT_STRATEGY);
    if (ret != Z_OK)
      throw std::runtime_error("ERROR: deflateInit2 failed on chunk " + std::to_string(i));
    ret = deflate(&zs, Z_FINISH);
    if (ret != Z_STREAM_END)
      throw std::runtime_error("ERROR: deflate did not return Z_STREAM_END on chunk " + std::to_string(i));
    ret = deflateEnd(&zs);
    if (ret != Z_OK)
      throw std::runtime_error("ERROR: deflateEnd failed on chunk " + std::to_string(i));
    compBuffer.resize(zs.total_out);
    gzipCompressedChunks[i] = std::move(compBuffer);
    compSizes[i] = zs.total_out;
  }

  // (Optional: you can measure compression time here using CUDA events.)

  // Step 3: Copy the compressed chunks to device memory.
  std::vector<void*> d_compChunks(numChunks, nullptr);
  for (size_t i = 0; i < numChunks; i++) {
    CUDA_CHECK(cudaMalloc(&d_compChunks[i], compSizes[i]));
    CUDA_CHECK(cudaMemcpy(d_compChunks[i], gzipCompressedChunks[i].data(), compSizes[i], cudaMemcpyHostToDevice));
  }
  std::vector<const void*> compPtrs(numChunks);
  for (size_t i = 0; i < numChunks; i++) {
    compPtrs[i] = d_compChunks[i];
  }

  // Step 4: GPU Decompression using NVCOMP GZIP API.
  // Create a BatchData object for the decompression output (using same dimensions as input_data_cpu)
  BatchData decomp_data(input_data_cpu, false);

  // Create CUDA stream and events for timing.
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  cudaEvent_t decomp_start, decomp_end;
  CUDA_CHECK(cudaEventCreate(&decomp_start));
  CUDA_CHECK(cudaEventCreate(&decomp_end));

  // Query temporary storage size needed for decompression.
  size_t decomp_temp_bytes = 0;
  nvcompStatus_t status = nvcompBatchedGzipDecompressGetTempSize(numChunks, chunk_size, &decomp_temp_bytes);
  if (status != nvcompSuccess)
    throw std::runtime_error("ERROR: Failed to get NVCOMP GZIP decompression temp size");

  // Allocate temporary device memory.
  void* d_decomp_temp;
  CUDA_CHECK(cudaMalloc(&d_decomp_temp, decomp_temp_bytes));
  size_t* d_decomp_sizes;
  CUDA_CHECK(cudaMalloc(&d_decomp_sizes, numChunks * sizeof(size_t)));
  nvcompStatus_t* d_statuses;
  CUDA_CHECK(cudaMalloc(&d_statuses, numChunks * sizeof(nvcompStatus_t)));

  // Record the start time.
  CUDA_CHECK(cudaEventRecord(decomp_start, stream));

  // Launch asynchronous decompression.
  status = nvcompBatchedGzipDecompressAsync(compPtrs.data(), compSizes.data(),
                                            decomp_data.sizes(), d_decomp_sizes,
                                            numChunks, d_decomp_temp, decomp_temp_bytes,
                                            decomp_data.ptrs(), d_statuses, stream);
  if (status != nvcompSuccess)
    throw std::runtime_error("ERROR: GZIP GPU decompression async call failed");

  // Record the end time and synchronize.
  CUDA_CHECK(cudaEventRecord(decomp_end, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  float decomp_time_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&decomp_time_ms, decomp_start, decomp_end));
  metrics.decompTimeMs = decomp_time_ms;
  metrics.decompThroughputGBs = ((double)total_bytes / decomp_time_ms) * 1e-6;

  // Clean up temporary device memory.
  CUDA_CHECK(cudaFree(d_decomp_temp));
  CUDA_CHECK(cudaFree(d_decomp_sizes));
  CUDA_CHECK(cudaFree(d_statuses));
  CUDA_CHECK(cudaEventDestroy(decomp_start));
  CUDA_CHECK(cudaEventDestroy(decomp_end));
  CUDA_CHECK(cudaStreamDestroy(stream));

  // Step 5: Free the device memory for the compressed chunks.
  for (size_t i = 0; i < numChunks; i++) {
    CUDA_CHECK(cudaFree(d_compChunks[i]));
  }

  // Step 6: Validate the decompressed output.
  // Concatenate the decompressed chunks (each is chunk_size) and then truncate to the original total_bytes.
  std::vector<char> decompConcatenated;
  for (size_t i = 0; i < numChunks; i++) {
    const char* chunk = reinterpret_cast<const char*>(input_data_cpu.ptrs()[i]);
    decompConcatenated.insert(decompConcatenated.end(), chunk, chunk + input_data_cpu.sizes()[i]);
  }
  decompConcatenated.resize(total_bytes);
  std::vector<char> originalConcatenated;
  for (const auto &file : files)
    originalConcatenated.insert(originalConcatenated.end(), file.begin(), file.end());
  if (memcmp(originalConcatenated.data(), decompConcatenated.data(), total_bytes) != 0)
    throw std::runtime_error("ERROR: Whole-dataset GZIP decompressed data does not match input");
  else
    std::cout << "Whole-dataset GZIP decompression validated :)" << std::endl;

  return metrics;
}

WholeDatasetGzipMetrics run_whole_dataset_gzip1(const std::vector<std::vector<char>>& files) {
  WholeDatasetGzipMetrics metrics;
  const size_t chunk_size = 1 << 16; // 64 KB

  // --- Step 1: Create a padded version of the input files ---
  // (nvCOMP’s GZIP decompression requires every chunk to be the same uncompressed size)
  std::vector<std::vector<char>> paddedFiles = files;
  for (auto &file : paddedFiles) {
    size_t remainder = file.size() % chunk_size;
    if (remainder != 0) {
      size_t pad = chunk_size - remainder;
      file.insert(file.end(), pad, 0); // pad with zeros
    }
  }
  // Build our CPU-side batch from the padded files.
  BatchDataCPU input_data_cpu(paddedFiles, chunk_size);

  // Compute the total original (unpadded) bytes for throughput and ratio metrics.
  size_t total_bytes = 0;
  for (const auto &file : files)
    total_bytes += file.size();
  metrics.totalBytes = total_bytes;

  // --- Step 2: CPU Compression using zlib in GZIP mode ---
  cudaEvent_t comp_start, comp_end;
  CUDA_CHECK(cudaEventCreate(&comp_start));
  CUDA_CHECK(cudaEventCreate(&comp_end));
  CUDA_CHECK(cudaEventRecord(comp_start, 0));
  size_t numChunks = input_data_cpu.size();
  std::vector<std::vector<char>> gzipCompressedChunks(numChunks);
  std::vector<size_t> compSizes(numChunks);
  for (size_t i = 0; i < numChunks; i++) {
    const char* src = reinterpret_cast<const char*>(input_data_cpu.ptrs()[i]);
    size_t srcSize = input_data_cpu.sizes()[i]; // this is now exactly chunk_size for each chunk (except that paddedFiles[?] has extra zeros)
    uLongf compBufferSize = compressBound(srcSize);
    std::vector<char> compBuffer(compBufferSize);
    // Set up z_stream for GZIP compression (using deflateInit2 with 15|16)
    z_stream zs;
    memset(&zs, 0, sizeof(zs));
    zs.next_in = reinterpret_cast<Bytef*>(const_cast<char*>(src));
    zs.avail_in = srcSize;
    zs.next_out = reinterpret_cast<Bytef*>(compBuffer.data());
    zs.avail_out = compBufferSize;
    int ret = deflateInit2(&zs, 9, Z_DEFLATED, 15 | 16, 8, Z_DEFAULT_STRATEGY);
    if (ret != Z_OK)
      throw std::runtime_error("ERROR: deflateInit2 failed on chunk " + std::to_string(i));
    ret = deflate(&zs, Z_FINISH);
    if (ret != Z_STREAM_END)
      throw std::runtime_error("ERROR: deflate did not return Z_STREAM_END on chunk " + std::to_string(i));
    ret = deflateEnd(&zs);
    if (ret != Z_OK)
      throw std::runtime_error("ERROR: deflateEnd failed on chunk " + std::to_string(i));
    compBuffer.resize(zs.total_out);
    gzipCompressedChunks[i] = std::move(compBuffer);
    compSizes[i] = zs.total_out;
  }
  CUDA_CHECK(cudaEventRecord(comp_end, 0));
  CUDA_CHECK(cudaEventSynchronize(comp_end));
  float comp_time_ms;
  CUDA_CHECK(cudaEventElapsedTime(&comp_time_ms, comp_start, comp_end));
  metrics.compTimeMs = comp_time_ms;
  metrics.compThroughputGBs = ((double)total_bytes / comp_time_ms) * 1e-6;
  CUDA_CHECK(cudaEventDestroy(comp_start));
  CUDA_CHECK(cudaEventDestroy(comp_end));

  size_t comp_bytes = 0;
  for (size_t i = 0; i < numChunks; i++)
    comp_bytes += compSizes[i];
  metrics.compBytes = comp_bytes;
  metrics.compressionRatio = (double)total_bytes / comp_bytes;

  // --- Step 3: Copy compressed chunks to device memory ---
  std::vector<void*> d_compChunks(numChunks, nullptr);
  for (size_t i = 0; i < numChunks; i++) {
    CUDA_CHECK(cudaMalloc(&d_compChunks[i], compSizes[i]));
    CUDA_CHECK(cudaMemcpy(d_compChunks[i], gzipCompressedChunks[i].data(), compSizes[i], cudaMemcpyHostToDevice));
  }
  // Build an array of device pointers for nvCOMP.
  std::vector<const void*> compPtrs(numChunks);
  for (size_t i = 0; i < numChunks; i++)
    compPtrs[i] = d_compChunks[i];

  // --- Step 4: GPU Decompression using nvcomp/gzip ---
  BatchData decomp_data(input_data_cpu, false);
  // For nvcomp GZIP, all chunks are assumed to decompress to the same size.
  // (We pass chunk_size as the uncompressed size.)
  const size_t batchSize = 1;
  size_t numBatches = (numChunks + batchSize - 1) / batchSize;
  size_t tempSize = 0;
  nvcompStatus_t status = nvcompBatchedGzipDecompressGetTempSize(batchSize, chunk_size, &tempSize);
  if (status != nvcompSuccess)
    throw std::runtime_error("ERROR: Failed to get nvcomp GZIP decompression temp size");
  void* d_decomp_temp;
  CUDA_CHECK(cudaMalloc(&d_decomp_temp, tempSize));
  size_t* d_decomp_sizes;
  CUDA_CHECK(cudaMalloc(&d_decomp_sizes, batchSize * sizeof(size_t)));
  nvcompStatus_t* d_statuses;
  CUDA_CHECK(cudaMalloc(&d_statuses, batchSize * sizeof(nvcompStatus_t)));
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  float totalBatchTime_ms = 0.0f;
  for (size_t batch = 0; batch < numBatches; batch++) {
    size_t batchStart = batch * batchSize;
    size_t currentBatchSize = std::min(batchSize, numChunks - batchStart);
    cudaEvent_t batchStartEvent, batchEndEvent;
    CUDA_CHECK(cudaEventCreate(&batchStartEvent));
    CUDA_CHECK(cudaEventCreate(&batchEndEvent));
    CUDA_CHECK(cudaEventRecord(batchStartEvent, stream));
    status = nvcompBatchedGzipDecompressAsync(compPtrs.data() + batchStart,
                                              compSizes.data() + batchStart,
                                              decomp_data.sizes() + batchStart,
                                              d_decomp_sizes,
                                              currentBatchSize,
                                              d_decomp_temp,
                                              tempSize,
                                              decomp_data.ptrs() + batchStart,
                                              d_statuses,
                                              stream);
    if (status != nvcompSuccess)
      throw std::runtime_error("ERROR: GZIP GPU decompression failed in batch " + std::to_string(batch));
    CUDA_CHECK(cudaEventRecord(batchEndEvent, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));  // <-- This is where the error (700) appears if a kernel fails.
    float batchTime_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&batchTime_ms, batchStartEvent, batchEndEvent));
    totalBatchTime_ms += batchTime_ms;
    CUDA_CHECK(cudaEventDestroy(batchStartEvent));
    CUDA_CHECK(cudaEventDestroy(batchEndEvent));
  }
  metrics.decompTimeMs = totalBatchTime_ms;
  metrics.decompThroughputGBs = ((double)total_bytes / totalBatchTime_ms) * 1e-6;
  CUDA_CHECK(cudaFree(d_decomp_temp));
  CUDA_CHECK(cudaFree(d_decomp_sizes));
  CUDA_CHECK(cudaFree(d_statuses));
  CUDA_CHECK(cudaStreamDestroy(stream));

  // --- Step 5: Free the device memory for compressed chunks ---
  for (size_t i = 0; i < numChunks; i++) {
    CUDA_CHECK(cudaFree(d_compChunks[i]));
  }

  // --- Step 6: Validate the decompressed output ---
  // Because we padded the input, we must compare only the original (unpadded) bytes.
  // Concatenate the decompressed chunks (which are each of size chunk_size) and then
  // compare the first total_bytes with the original data.
  std::vector<char> decompConcatenated;
  for (size_t i = 0; i < numChunks; i++) {
    const char* chunk = reinterpret_cast<const char*>(input_data_cpu.ptrs()[i]);  // using input_data_cpu to get order
    // Note: In your actual BatchData implementation, you might have a helper to concatenate the data.
    decompConcatenated.insert(decompConcatenated.end(), chunk, chunk + input_data_cpu.sizes()[i]);
  }
  // Truncate to original size.
  decompConcatenated.resize(total_bytes);
  std::vector<char> originalConcatenated;
  for (const auto &file : files)
    originalConcatenated.insert(originalConcatenated.end(), file.begin(), file.end());
  if (memcmp(originalConcatenated.data(), decompConcatenated.data(), total_bytes) != 0)
    throw std::runtime_error("ERROR: Whole-dataset GZIP decompressed data does not match input");
  else
    std::cout << "Whole-dataset GZIP decompression validated :)" << std::endl;

  return metrics;
}

// -----------------------------------------------------------------------------
// For component-based GZIP, use deflateInit2 to produce a GZIP stream
// -----------------------------------------------------------------------------
ComponentResult compress_decompress_component_gzip(const std::vector<char>& compData) {
  ComponentResult result;
  result.compData = compData;
  size_t comp_total_bytes = compData.size();

  // Use deflateInit2/deflate to produce a GZIP stream
  auto compStart = std::chrono::high_resolution_clock::now();
  uLong srcSize = comp_total_bytes;
  uLongf compBufferSize = compressBound(srcSize);
  std::vector<char> compBuffer(compBufferSize);
  z_stream zs;
  memset(&zs, 0, sizeof(zs));
  zs.next_in = reinterpret_cast<Bytef*>(const_cast<char*>(compData.data()));
  zs.avail_in = srcSize;
  zs.next_out = reinterpret_cast<Bytef*>(compBuffer.data());
  zs.avail_out = compBufferSize;
  int ret = deflateInit2(&zs, 9, Z_DEFLATED, 15 | 16, 8, Z_DEFAULT_STRATEGY);
  if (ret != Z_OK)
    throw std::runtime_error("ERROR: deflateInit2 failed in component GZIP compression");
  ret = deflate(&zs, Z_FINISH);
  if (ret != Z_STREAM_END)
    throw std::runtime_error("ERROR: deflate did not return Z_STREAM_END in component GZIP compression");
  ret = deflateEnd(&zs);
  if (ret != Z_OK)
    throw std::runtime_error("ERROR: deflateEnd failed in component GZIP compression");
  auto compEnd = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float, std::milli> compDuration = compEnd - compStart;
  result.compTimeMs = compDuration.count();
  compBuffer.resize(zs.total_out);
  result.compSize = zs.total_out;

  // GPU decompression using nvcomp/gzip:
  size_t tempSize = 0;
  nvcompStatus_t status = nvcompBatchedGzipDecompressGetTempSize(1, comp_total_bytes, &tempSize);
  if (status != nvcompSuccess) {
    std::cerr << "Failed to get nvcomp GZIP decompression temp size for component." << std::endl;
    result.decompTimeMs = 0.0f;
    return result;
  }
  void* d_decomp_temp;
  CUDA_CHECK(cudaMalloc(&d_decomp_temp, tempSize));
  size_t* d_decomp_sizes;
  CUDA_CHECK(cudaMalloc(&d_decomp_sizes, sizeof(size_t)));
  nvcompStatus_t* d_statuses;
  CUDA_CHECK(cudaMalloc(&d_statuses, sizeof(nvcompStatus_t)));
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  const void* compPtrs[1] = { compBuffer.data() };
  size_t compSizes[1] = { compBuffer.size() };
  std::vector<char> decompBuffer(comp_total_bytes);
  void* decompPtrs[1] = { decompBuffer.data() };
  size_t decompSizes[1] = { comp_total_bytes };
  cudaEvent_t decomp_start, decomp_end;
  CUDA_CHECK(cudaEventCreate(&decomp_start));
  CUDA_CHECK(cudaEventCreate(&decomp_end));
  CUDA_CHECK(cudaEventRecord(decomp_start, stream));
  status = nvcompBatchedGzipDecompressAsync(compPtrs, compSizes, decompSizes, d_decomp_sizes,
                                             1, d_decomp_temp, tempSize, decompPtrs, d_statuses, stream);
  if (status != nvcompSuccess) {
    std::cerr << "GZIP GPU decompression failed for component." << std::endl;
    result.decompTimeMs = 0.0f;
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_decomp_temp));
    CUDA_CHECK(cudaFree(d_decomp_sizes));
    CUDA_CHECK(cudaFree(d_statuses));
    return result;
  }
  CUDA_CHECK(cudaEventRecord(decomp_end, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  float decomp_time_ms;
  CUDA_CHECK(cudaEventElapsedTime(&decomp_time_ms, decomp_start, decomp_end));
  result.decompTimeMs = decomp_time_ms;
  CUDA_CHECK(cudaEventDestroy(decomp_start));
  CUDA_CHECK(cudaEventDestroy(decomp_end));
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(d_decomp_temp));
  CUDA_CHECK(cudaFree(d_decomp_sizes));
  CUDA_CHECK(cudaFree(d_statuses));
  if (memcmp(compData.data(), decompBuffer.data(), comp_total_bytes) != 0)
    std::cerr << "GZIP component decompression validation failed." << std::endl;
  result.compData.assign(decompBuffer.begin(), decompBuffer.end());
  return result;
}

ComponentGzipMetrics run_component_gzip(const std::vector<uint8_t>& globalByteArray,
                                          const std::vector<std::vector<size_t>>& chosenConfig,
                                          int numThreads) {
  ComponentGzipMetrics metrics;
  metrics.configStr = configToString(chosenConfig);
  metrics.totalUncompressed = globalByteArray.size();
  metrics.totalCompressed = 0;
  metrics.overallCompTimeMs = 0.0f;
  metrics.overallDecompTimeMs = 0.0f;
  std::vector<std::vector<uint8_t>> decomposedComponents;
  splitBytesIntoComponentsNested(globalByteArray, decomposedComponents, chosenConfig, numThreads);
  std::cout << "Dataset decomposed into " << decomposedComponents.size() << " components (GZIP decomposed mode)." << std::endl;
  std::vector<std::vector<uint8_t>> decomposedComponentData;
  int compIndex = 0;
  for (const auto &component : decomposedComponents) {
    std::vector<char> compData(component.begin(), component.end());
    ComponentResult result = compress_decompress_component_gzip(compData);
    metrics.totalCompressed += result.compSize;
    metrics.overallCompTimeMs += result.compTimeMs;
    metrics.overallDecompTimeMs += result.decompTimeMs;
    std::vector<uint8_t> compDecomp(result.compData.begin(), result.compData.end());
    decomposedComponentData.push_back(compDecomp);
    double compThroughput = (result.compTimeMs > 0) ? ((double)compData.size() / result.compTimeMs) * 1e-6 : 0.0;
    double decompThroughput = (result.decompTimeMs > 0) ? ((double)compData.size() / result.decompTimeMs) * 1e-6 : 0.0;
    double compRatio = (result.compSize > 0) ? ((double)compData.size() / result.compSize) : 0.0;
    std::cout << "GZIP Component " << compIndex << " uncompressed: " << compData.size()
              << " bytes, compressed: " << result.compSize
              << " bytes, ratio: " << std::fixed << std::setprecision(2) << compRatio
              << ", comp throughput: " << compThroughput
              << " GB/s, decomp throughput: " << decompThroughput << " GB/s" << std::endl;
    compIndex++;
  }
  metrics.compressionRatio = (metrics.totalCompressed > 0) ? ((double)metrics.totalUncompressed / metrics.totalCompressed) : 0.0;
  metrics.compThroughputGBs = (metrics.overallCompTimeMs > 0) ? ((double)metrics.totalUncompressed / metrics.overallCompTimeMs) * 1e-6 : 0.0;
  metrics.decompThroughputGBs = (metrics.overallDecompTimeMs > 0) ? ((double)metrics.totalUncompressed / metrics.overallDecompTimeMs) * 1e-6 : 0.0;
  return metrics;
}

// -----------------------------------------------------------------------------
// Global Map for Component Configurations
// -----------------------------------------------------------------------------
std::map<std::string, std::vector<std::vector<std::vector<size_t>>>> datasetComponentMap = {
  {"default", {{{1}, {2}, {3}, {4}}}}
  // Add other dataset configurations as needed.
};

// -----------------------------------------------------------------------------
// (D) Main
// -----------------------------------------------------------------------------
int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <datasetPath> <precisionBits (32|64)>" << std::endl;
    return EXIT_FAILURE;
  }
  std::string datasetPath = argv[1];
  int precisionBits = std::stoi(argv[2]);
  std::vector<uint8_t> globalByteArray;
  size_t rowCount = 0;
  if (precisionBits == 64) {
    auto tmp = loadTSVDatasetdouble(datasetPath);
    if (tmp.first.empty()) {
      std::cerr << "Failed to load dataset from " << datasetPath << std::endl;
      return EXIT_FAILURE;
    }
    globalByteArray = convertDoubleToBytes(tmp.first);
    rowCount = tmp.second;
    std::cout << "Loaded " << rowCount << " rows (64-bit) with " << tmp.first.size() << " total values." << std::endl;
  } else if (precisionBits == 32) {
    auto tmp = loadTSVDataset(datasetPath);
    if (tmp.first.empty()) {
      std::cerr << "Failed to load dataset from " << datasetPath << std::endl;
      return EXIT_FAILURE;
    }
    globalByteArray = convertFloatToBytes(tmp.first);
    rowCount = tmp.second;
    std::cout << "Loaded " << rowCount << " rows (32-bit) with " << tmp.first.size() << " total values." << std::endl;
  } else {
    std::cerr << "Unsupported precision: " << precisionBits << ". Use 32 or 64." << std::endl;
    return EXIT_FAILURE;
  }
  size_t totalBytes = globalByteArray.size();
  std::cout << "Total dataset bytes: " << totalBytes << std::endl;
  auto pos = datasetPath.find_last_of("/\\");
  std::string datasetName = (pos == std::string::npos) ? datasetPath : datasetPath.substr(pos + 1);
  pos = datasetName.find_last_of('.');
  if (pos != std::string::npos)
    datasetName = datasetName.substr(0, pos);
  std::string csvFilename = "/home/jamalids/Documents/" + datasetName + ".csv";
  std::ofstream csvFile(csvFilename);
  if (!csvFile.is_open()) {
    std::cerr << "Failed to open CSV file for writing: " << csvFilename << std::endl;
    return EXIT_FAILURE;
  }
  csvFile << "Dataset,Config,Level,ComponentIndex,UncompressedBytes,CompressedBytes,CompressionRatio,CompTimeMs,DecompTimeMs,CompThroughputGBs,DecompThroughputGBs\n";

  // (I) Whole-Dataset Compression/Decompression (gdeflate mode)
  std::vector<char> dataAsChar(globalByteArray.begin(), globalByteArray.end());
  std::vector<std::vector<char>> files = { dataAsChar };
  WholeDatasetMetrics wholeMetrics = run_whole_dataset(files);
  csvFile << datasetName << ",N/A,WholeDataset," << ","
          << wholeMetrics.totalBytes << ","
          << wholeMetrics.compBytes << ","
          << std::fixed << std::setprecision(2) << wholeMetrics.compressionRatio << ","
          << wholeMetrics.compTimeMs << ","
          << wholeMetrics.decompTimeMs << ","
          << wholeMetrics.compThroughputGBs << ","
          << wholeMetrics.decompThroughputGBs << "\n";

  // (II) Component-Based Compression/Decompression (gdeflate mode)
  std::vector<std::vector<std::vector<size_t>>> configOptions;
  if (datasetComponentMap.find(datasetName) != datasetComponentMap.end())
    configOptions = datasetComponentMap[datasetName];
  else
    configOptions = datasetComponentMap["default"];
  int numThreads = 10;
  for (size_t cfgIdx = 0; cfgIdx < configOptions.size(); cfgIdx++) {
    const std::vector<std::vector<size_t>>& chosenConfig = configOptions[cfgIdx];
    std::string configStr = configToString(chosenConfig);
    std::cout << "Processing configuration " << cfgIdx << ": " << configStr << std::endl;
    std::vector<std::vector<uint8_t>> decomposedComponents;
    splitBytesIntoComponentsNested(globalByteArray, decomposedComponents, chosenConfig, numThreads);
    std::cout << "Dataset decomposed into " << decomposedComponents.size() << " components." << std::endl;
    std::vector<std::vector<uint8_t>> decomposedComponentData;
    size_t totalCompCompressedSize = 0;
    float overallCompTimeMs = 0.0f;
    float overallDecompTimeMs = 0.0f;
    for (size_t compIdx = 0; compIdx < decomposedComponents.size(); compIdx++) {
      std::vector<char> compData(decomposedComponents[compIdx].begin(), decomposedComponents[compIdx].end());
      size_t compUncompressed = compData.size();
      ComponentResult result = compress_decompress_component(compData);
      totalCompCompressedSize += result.compSize;
      overallCompTimeMs += result.compTimeMs;
      overallDecompTimeMs += result.decompTimeMs;
      std::vector<uint8_t> compDecomp(result.compData.begin(), result.compData.end());
      decomposedComponentData.push_back(compDecomp);
      double compThroughput = result.compTimeMs > 0 ? ((double)compUncompressed / result.compTimeMs) * 1e-6 : 0.0;
      double decompThroughput = result.decompTimeMs > 0 ? ((double)compUncompressed / result.decompTimeMs) * 1e-6 : 0.0;
      double compRatio = result.compSize > 0 ? ((double)compUncompressed / result.compSize) : 0.0;
      csvFile << datasetName << "," << configStr << ",Component," << compIdx << ","
              << compUncompressed << "," << result.compSize << ","
              << std::fixed << std::setprecision(2) << compRatio << ","
              << result.compTimeMs << "," << result.decompTimeMs << ","
              << compThroughput << "," << decompThroughput << "\n";
    }
    double overallCompressionRatio = totalCompCompressedSize > 0 ? ((double)totalBytes / totalCompCompressedSize) : 0.0;
    double overallCompThroughput = overallCompTimeMs > 0 ? ((double)totalBytes / overallCompTimeMs) * 1e-6 : 0.0;
    double overallDecompThroughput = overallDecompTimeMs > 0 ? ((double)totalBytes / overallDecompTimeMs) * 1e-6 : 0.0;
    csvFile << datasetName << "," << configStr << ",Overall," << ","
            << totalBytes << "," << totalCompCompressedSize << ","
            << std::fixed << std::setprecision(2) << overallCompressionRatio << ","
            << overallCompTimeMs << "," << overallDecompTimeMs << ","
            << overallCompThroughput << "," << overallDecompThroughput << "\n";
    std::vector<uint8_t> reassembled(globalByteArray.size());
    reassembleBytesFromComponentsNestedlz4(decomposedComponentData, reassembled.data(), reassembled.size(), chosenConfig, numThreads);
    if (reassembled == globalByteArray)
      std::cout << "Reassembled data matches the original dataset." << std::endl;
    else
      std::cerr << "Error: Reassembled data does NOT match the original dataset!" << std::endl;
  }

  // (III) Whole-Dataset GZIP Mode
  std::cout << "Running alternate GZIP whole-dataset mode..." << std::endl;
  WholeDatasetGzipMetrics gzipWholeMetrics = run_whole_dataset_gzip(files);
  csvFile << datasetName << ",GZIPWholeDataset,WholeDatasetGZIP," << ","
          << gzipWholeMetrics.totalBytes << "," << gzipWholeMetrics.compBytes << ","
          << std::fixed << std::setprecision(2) << gzipWholeMetrics.compressionRatio << ","
          << gzipWholeMetrics.compTimeMs << "," << gzipWholeMetrics.decompTimeMs << ","
          << gzipWholeMetrics.compThroughputGBs << "," << gzipWholeMetrics.decompThroughputGBs << "\n";

  // (IV) Component-Based (Decomposed) GZIP Mode
  std::cout << "Running alternate GZIP decomposed mode..." << std::endl;
  for (size_t cfgIdx = 0; cfgIdx < configOptions.size(); cfgIdx++) {
    const std::vector<std::vector<size_t>>& chosenConfig = configOptions[cfgIdx];
    ComponentGzipMetrics compGzipMetrics = run_component_gzip(globalByteArray, chosenConfig, numThreads);
    csvFile << datasetName << "," << compGzipMetrics.configStr << ",ComponentGZIP," << ","
            << compGzipMetrics.totalUncompressed << "," << compGzipMetrics.totalCompressed << ","
            << std::fixed << std::setprecision(2) << compGzipMetrics.compressionRatio << ","
            << compGzipMetrics.overallCompTimeMs << "," << compGzipMetrics.overallDecompTimeMs << ","
            << compGzipMetrics.compThroughputGBs << "," << compGzipMetrics.decompThroughputGBs << "\n";
  }

  csvFile.close();
  std::cout << "Results saved to " << csvFilename << std::endl;
  return EXIT_SUCCESS;
}
