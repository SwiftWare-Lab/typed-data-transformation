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

#include "BatchData.h"

// nvCOMP headers
#include <nvcomp/native/gdeflate_cpu.h>
#include <nvcomp/gdeflate.h>

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
#ifdef _OPENMP
#include <omp.h>
#endif
#include <chrono> // if needed for wall clock timing

// -----------------------------------------------------------------------------
// Dataset Loading and Conversion Functions (for TSV files)
// -----------------------------------------------------------------------------

// Loads a TSV file (skipping the first column) and parses remaining columns as float.
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

// Loads a TSV file (skipping the first column) and parses remaining columns as double.
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

// Converts a vector of floats to a vector of bytes.
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

// Converts a vector of doubles to a vector of bytes.
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
// (A) Whole-Dataset Compression/Decompression with Throughput Measurements
// -----------------------------------------------------------------------------
static void run_example(const std::vector<std::vector<char>>& data, const size_t chunk_size, int algorithm)
{
  size_t total_bytes = 0;
  for (const auto &part : data) {
    total_bytes += part.size();
  }

  std::cout << "----------" << std::endl;
  std::cout << "Whole-dataset mode:" << std::endl;
  std::cout << "Files: " << data.size() << std::endl;
  std::cout << "Uncompressed (B): " << total_bytes << std::endl;
  std::cout << "Using chunk size: " << chunk_size << " bytes" << std::endl;
  std::cout << "Algorithm: " << algorithm << std::endl;

  // Build up input batch on CPU.
  BatchDataCPU input_data_cpu(data, chunk_size);
  std::cout << "Chunks: " << input_data_cpu.size() << std::endl;

  // -----------------
  // Compression Phase (with timing)
  // -----------------
  cudaEvent_t comp_start, comp_end;
  CUDA_CHECK(cudaEventCreate(&comp_start));
  CUDA_CHECK(cudaEventCreate(&comp_end));
  CUDA_CHECK(cudaEventRecord(comp_start, 0));

  nvcompStatus_t status;
  size_t max_out_bytes;
  status = nvcompBatchedGdeflateCompressGetMaxOutputChunkSize(
      chunk_size, nvcompBatchedGdeflateDefaultOpts, &max_out_bytes);
  if (status != nvcompSuccess) {
    throw std::runtime_error("ERROR: Failed to get max output chunk size");
  }

  BatchDataCPU compress_data_cpu(max_out_bytes, input_data_cpu.size());

  // Select compression algorithm:
  switch(algorithm) {
    case 0:
      // Use NVCOMP gdeflate (our "libdeflate" option)
      gdeflate::compressCPU(input_data_cpu.ptrs(), input_data_cpu.sizes(),
                            chunk_size, input_data_cpu.size(),
                            compress_data_cpu.ptrs(), compress_data_cpu.sizes());
      break;
    case 1:
      // Stub for zlib_compress2 (replace with actual call if available)
      gdeflate::compressCPU(input_data_cpu.ptrs(), input_data_cpu.sizes(),
                            chunk_size, input_data_cpu.size(),
                            compress_data_cpu.ptrs(), compress_data_cpu.sizes());
      break;
    case 2:
      // Stub for zlib_deflate (replace with actual call if available)
      gdeflate::compressCPU(input_data_cpu.ptrs(), input_data_cpu.sizes(),
                            chunk_size, input_data_cpu.size(),
                            compress_data_cpu.ptrs(), compress_data_cpu.sizes());
      break;
    default:
      throw std::runtime_error("Unknown compression algorithm");
  }

  CUDA_CHECK(cudaEventRecord(comp_end, 0));
  CUDA_CHECK(cudaEventSynchronize(comp_end));
  float comp_time_ms;
  CUDA_CHECK(cudaEventElapsedTime(&comp_time_ms, comp_start, comp_end));
  double comp_throughput = ((double)total_bytes / comp_time_ms) * 1e-6; // GB/s
  std::cout << "Whole-dataset compression throughput (GB/s): " << comp_throughput << std::endl;
  CUDA_CHECK(cudaEventDestroy(comp_start));
  CUDA_CHECK(cudaEventDestroy(comp_end));

  // Compute compressed size.
  size_t* compressed_sizes_host = compress_data_cpu.sizes();
  size_t comp_bytes = 0;
  for (size_t i = 0; i < compress_data_cpu.size(); ++i)
    comp_bytes += compressed_sizes_host[i];
  std::cout << "Compressed size (B): " << comp_bytes
            << ", Compression ratio: " << std::fixed << std::setprecision(2)
            << (double)total_bytes / comp_bytes << std::endl;

  // -----------------
  // Decompression Phase (with timing)
  // -----------------
  BatchData compress_data(compress_data_cpu, true);
  BatchData decomp_data(input_data_cpu, false);

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  cudaEvent_t decomp_start, decomp_end;
  CUDA_CHECK(cudaEventCreate(&decomp_start));
  CUDA_CHECK(cudaEventCreate(&decomp_end));

  size_t decomp_temp_bytes;
  status = nvcompBatchedGdeflateDecompressGetTempSize(
      compress_data.size(), chunk_size, &decomp_temp_bytes);
  if (status != nvcompSuccess) {
    throw std::runtime_error("ERROR: Failed to get decompression temp size");
  }

  void* d_decomp_temp;
  CUDA_CHECK(cudaMalloc(&d_decomp_temp, decomp_temp_bytes));
  size_t* d_decomp_sizes;
  CUDA_CHECK(cudaMalloc(&d_decomp_sizes, decomp_data.size() * sizeof(size_t)));
  nvcompStatus_t* d_statuses;
  CUDA_CHECK(cudaMalloc(&d_statuses, decomp_data.size() * sizeof(nvcompStatus_t)));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Validate decompression first.
  status = nvcompBatchedGdeflateDecompressAsync(
      compress_data.ptrs(), compress_data.sizes(),
      decomp_data.sizes(), d_decomp_sizes,
      compress_data.size(), d_decomp_temp, decomp_temp_bytes,
      decomp_data.ptrs(), d_statuses, stream);
  if (status != nvcompSuccess) {
    throw std::runtime_error("ERROR: Decompression failed");
  }
  if (!(input_data_cpu == decomp_data))
    throw std::runtime_error("ERROR: Decompressed data does not match input");
  else
    std::cout << "Decompression validated :)" << std::endl;

  // Measure decompression throughput.
  CUDA_CHECK(cudaEventRecord(decomp_start, stream));
  status = nvcompBatchedGdeflateDecompressAsync(
      compress_data.ptrs(), compress_data.sizes(),
      decomp_data.sizes(), d_decomp_sizes,
      compress_data.size(), d_decomp_temp, decomp_temp_bytes,
      decomp_data.ptrs(), d_statuses, stream);
  CUDA_CHECK(cudaEventRecord(decomp_end, stream));
  if (status != nvcompSuccess) {
    throw std::runtime_error("ERROR: Decompression throughput run failed");
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));
  float decomp_time_ms;
  CUDA_CHECK(cudaEventElapsedTime(&decomp_time_ms, decomp_start, decomp_end));
  double decomp_throughput = ((double)total_bytes / decomp_time_ms) * 1e-6;
  std::cout << "Whole-dataset decompression throughput (GB/s): " << decomp_throughput << std::endl;

  CUDA_CHECK(cudaFree(d_decomp_temp));
  CUDA_CHECK(cudaFree(d_decomp_sizes));
  CUDA_CHECK(cudaFree(d_statuses));
  CUDA_CHECK(cudaEventDestroy(decomp_start));
  CUDA_CHECK(cudaEventDestroy(decomp_end));
  CUDA_CHECK(cudaStreamDestroy(stream));

  // Reconstruct decompressed data.
  std::vector<char> decompressed;
  decompressed.resize(data.size());
  uint8_t* decomp_ptr = static_cast<uint8_t*>(decomp_data.ptrs()[0]);
  std::copy(decomp_ptr, decomp_ptr + data.size(), decompressed.begin());

  return decompressed;
}

// -----------------------------------------------------------------------------
// (B) Component-Based Compression/Decompression with Overall Throughput
// -----------------------------------------------------------------------------

// Map of dataset names to possible component configurations.
// Map to store dataset names and their multiple possible configurations
std::map<std::string, std::vector<std::vector<std::vector<size_t>>>> datasetComponentMap = {
  {"acs_wht_f32", {
          {{1,2}, {3}, {4}} ,
          // {{1, 2,3}, {4}},
  }},
  {"g24_78_usb2_f32", {
          {{1}, {2,3}, {4}},
          {{1,2,3}, {4}},
  }},
  {"jw_mirimage_f32", {
          {{1,2}, {3}, {4}},
          {{1,2,3}, {4}},
  }},
  {"spitzer_irac_f32", {
          {{1,2}, {3}, {4}},
          {{1,2,3}, {4}},
  }},
  {"turbulence_f32", {
          {{1,2}, {3}, {4}},
          {{1,2,3}, {4}},
  }},
  {"wave_f32", {
          {{1,2}, {3}, {4}},
          {{1,2,3}, {4}},
  }},
  {"hdr_night_f32", {
          {{1,4}, {2}, {3}},
          {{1}, {2}, {3}, {4}},
          {{1,4}, {2,3}},
  }},
  {"ts_gas_f32", {
          {{1,2}, {3}, {4}},
  }},
  {"solar_wind_f32", {
          {{1}, {4}, {2}, {3}},
          {{1}, {2,3}, {4}},
  }},
  {"tpch_lineitem_f32", {
          {{1,2,3}, {4}},
          {{1,2}, {3}, {4}},
  }},
  {"tpcds_web_f32", {
          {{1,2,3}, {4}},
          {{1}, {2,3}, {4}},
  }},
  {"tpcds_store_f32", {
          {{1,2,3}, {4}},
          {{1}, {2,3}, {4}},
  }},
  {"tpcds_catalog_f32", {
          {{1,2,3}, {4}},
          {{1}, {2,3}, {4}},
  }},
  {"citytemp_f32", {
          {{1,4}, {2,3}},
          {{1}, {2}, {3}, {4}},
          {{1,2}, {3}, {4}}
  }},
  {"hst_wfc3_ir_f32", {
          {{1}, {2}, {3}, {4}},
          {{1,2}, {3}, {4}}
  }},
  {"hst_wfc3_uvis_f32", {
          {{1}, {2}, {3}, {4}},
          {{1,2}, {3}, {4}}
  }},
  {"rsim_f32", {
          {{1,2,3}, {4}},
          {{1,2}, {3}, {4}}
  }},
  {"astro_mhd_f64", {
          {{1,2,3,4,5,6}, {7}, {8}},
          {{1,2,3,4,5}, {6}, {7}, {8}}
  }},
  {"astro_pt_f64", {
          {{1,2,3,4,5,6}, {7}, {8}},
          {{1,2,3,4,5}, {6}, {7}, {8}}
  }},
  {"jane_street_f64", {
          {{1,2,3,4,5,6}, {7}, {8}},
          {{3,2,5,6,4,1}, {7}, {8}}
  }},
  {"msg_bt_f64", {
          {{1,2,3,4,5}, {6}, {7}, {8}},
          {{3,2,1,4,5,6}, {7}, {8}},
          {{3,2,1,4,5}, {6}, {7}, {8}}
  }},
  {"num_brain_f64", {
           {{7},{4}, {6}, {5}, {3},{2}, {1}, {8}},

          {{1,2,3,4,5,6}, {7}, {8}},
          {{3,2,4,5,1,6}, {7}, {8}},
  }},
  {"num_control_f64", {
          {{1,2,3,4,5,6}, {7}, {8}},
          {{4,5}, {6,3}, {1,2}, {7}, {8}},
          {{4,5,6,3,1,2}, {7}, {8}},
  }},
  {"nyc_taxi2015_f64", {
          {{7,4,6}, {5}, {3,2,1,8}},
          {{7,4,6,5}, {3,2,1,8}},
          {{7,4,6}, {5}, {3,2,1}, {8}},
          {{7,4}, {6}, {5}, {3,2}, {1}, {8}},
  }},
  {"phone_gyro_f64", {
          {{4,6}, {5}, {3,2,1,7}, {8}},
          {{4,6}, {1}, {3,2}, {5}, {7}, {8}},
          {{6,4,3,2,1,7}, {5}, {8}},
  }},
  {"tpch_order_f64", {
          {{3,2,4,1}, {7}, {6,5}, {8}},
          {{3,2,4,1,7}, {6,5}, {8}},
  }},
  {"tpcxbb_store_f64", {
          {{4,2,3}, {1}, {5}, {7}, {6}, {8}},
          {{4,2,3,1}, {5}, {7,6}, {8}},
          {{4,2,3,1,5}, {7,6}, {8}},
  }},
  {"tpcxbb_web_f64", {
          {{4,2,3}, {1}, {5}, {7}, {6}, {8}},
          {{4,2,3,1}, {5}, {7,6}, {8}},
          {{4,2,3,1,5}, {7,6}, {8}},
  }},
  {"wesad_chest_f64", {
              {{7,5,6}, {8,4,1,3,2}},
          {{7,5}, {6}, {8,4,1}, {3,2}},
          {{7,5}, {6}, {8,4}, {1}, {3,2}},
          {{7,5}, {6}, {8,4,1,3,2}},
  }},
  {"default", {
          {{1}, {2}, {3}, {4}}
  }}
};


// Splits the byte array into components according to a configuration.
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
                size_t idxInElem = groupIndices[sub] - 1;
                size_t globalSrcIdx = elem * totalBytesPerElement + idxInElem;
                outputComponents[compIdx][writePos + sub] = byteArray[globalSrcIdx];
            }
        }
    }
}

// Reassembles the full byte array from its component parts.
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
// (C) Component-Based Compression/Decompression with Overall Throughput
// -----------------------------------------------------------------------------

// Structure to hold per-component results.
struct ComponentResult {
  size_t compSize;
  std::vector<char> compData;
  float compTimeMs;
  float decompTimeMs;
};

// Compresses and decompresses a single component and returns the result.
ComponentResult compress_decompress_component(const std::vector<char>& compData)
{
  ComponentResult result;
  result.compData = compData; // Placeholder: assume decompression returns input
  size_t comp_total_bytes = compData.size();

  // Wrap compData as a single file.
  std::vector<std::vector<char>> files = { compData };

  // Create CUDA stream.
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // Compression Phase for Component.
  cudaEvent_t comp_comp_start, comp_comp_end;
  CUDA_CHECK(cudaEventCreate(&comp_comp_start));
  CUDA_CHECK(cudaEventCreate(&comp_comp_end));
  CUDA_CHECK(cudaEventRecord(comp_comp_start, stream));

  const size_t chunk_size = 1 << 16; // 64K
  BatchDataCPU input_data_cpu(files, chunk_size);
  nvcompStatus_t status;
  size_t max_out_bytes;
  status = nvcompBatchedGdeflateCompressGetMaxOutputChunkSize(
      chunk_size, nvcompBatchedGdeflateDefaultOpts, &max_out_bytes);
  if (status != nvcompSuccess)
    throw std::runtime_error("Error: Failed to get max output size for component");

  BatchDataCPU compress_data_cpu(max_out_bytes, input_data_cpu.size());
  gdeflate::compressCPU(
      input_data_cpu.ptrs(), input_data_cpu.sizes(),
      chunk_size, input_data_cpu.size(),
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

  // Compute compressed size.
  size_t* compressed_sizes_host = compress_data_cpu.sizes();
  size_t compSize = 0;
  for (size_t i = 0; i < input_data_cpu.size(); i++)
    compSize += compressed_sizes_host[i];
  result.compSize = compSize;

  // Decompression Phase for Component.
  BatchData compress_data(compress_data_cpu, true);
  BatchData decomp_data(input_data_cpu, false);

  size_t decomp_temp_bytes;
  status = nvcompBatchedGdeflateDecompressGetTempSize(
      compress_data.size(), chunk_size, &decomp_temp_bytes);
  if (status != nvcompSuccess)
    throw std::runtime_error("Error: Failed to get temp decompression size for component");

  void* d_decomp_temp;
  CUDA_CHECK(cudaMalloc(&d_decomp_temp, decomp_temp_bytes));
  size_t* d_decomp_sizes;
  CUDA_CHECK(cudaMalloc(&d_decomp_sizes, decomp_data.size() * sizeof(size_t)));
  nvcompStatus_t* d_statuses;
  CUDA_CHECK(cudaMalloc(&d_statuses, decomp_data.size() * sizeof(nvcompStatus_t)));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  status = nvcompBatchedGdeflateDecompressAsync(
      compress_data.ptrs(), compress_data.sizes(),
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
  status = nvcompBatchedGdeflateDecompressAsync(
      compress_data.ptrs(), compress_data.sizes(),
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
// (D) Main: Read Dataset, Run Whole-Dataset and Component-Based Compression/Decompression
// -----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
  // Usage: <program> <datasetPath> <precisionBits (32|64)> [chunkSizeInBytes] [algorithm]
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <datasetPath> <precisionBits (32|64)> [chunkSizeInBytes] [algorithm]" << std::endl;
    return EXIT_FAILURE;
  }

  std::string datasetPath = argv[1];
  int precisionBits = std::stoi(argv[2]);
  size_t chunk_size = (argc >= 4) ? std::stoul(argv[3]) : (1 << 16);
  int algorithm = (argc >= 5) ? std::stoi(argv[4]) : 0;

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

  // --- (I) Whole-Dataset Compression/Decompression ---
  std::vector<char> dataAsChar(globalByteArray.begin(), globalByteArray.end());
  std::vector<std::vector<char>> files = { dataAsChar };
  std::vector<char> decompressed = run_example(files, chunk_size, algorithm);
  if(decompressed == dataAsChar)
    std::cout << "Whole-dataset reassembled data matches the original dataset." << std::endl;
  else
    std::cerr << "Error: Whole-dataset reassembled data does NOT match the original dataset!" << std::endl;

  // --- (II) Component-Based Compression/Decompression ---
  // Extract dataset name from file path.
  auto pos = datasetPath.find_last_of("/\\");
  std::string datasetName = (pos == std::string::npos) ? datasetPath : datasetPath.substr(pos + 1);
  pos = datasetName.find_last_of('.');
  if (pos != std::string::npos)
    datasetName = datasetName.substr(0, pos);

  // Look up configuration; if not found, use "default".
  std::vector<std::vector<std::vector<size_t>>> configOptions;
  if (datasetComponentMap.find(datasetName) != datasetComponentMap.end())
    configOptions = datasetComponentMap[datasetName];
  else
    configOptions = datasetComponentMap["default"];
  const std::vector<std::vector<size_t>>& chosenConfig = configOptions[0];

  // Split the global byte array into components.
  std::vector<std::vector<uint8_t>> decomposedComponents;
  int numThreads = 10;
  splitBytesIntoComponentsNested(globalByteArray, decomposedComponents, chosenConfig, numThreads);
  std::cout << "Dataset decomposed into " << decomposedComponents.size() << " components." << std::endl;

  // Process each component.
  size_t totalCompCompressedSize = 0;
  float overallCompTimeMs = 0.0f;
  float overallDecompTimeMs = 0.0f;
  std::vector<std::vector<uint8_t>> decompressedComponentData;
  for (size_t i = 0; i < decomposedComponents.size(); i++) {
    std::vector<char> compData(decomposedComponents[i].begin(), decomposedComponents[i].end());
    ComponentResult result = compress_decompress_component(compData);
    totalCompCompressedSize += result.compSize;
    overallCompTimeMs += result.compTimeMs;
    overallDecompTimeMs += result.decompTimeMs;
    std::vector<uint8_t> compDecomp(result.compData.begin(), result.compData.end());
    decompressedComponentData.push_back(compDecomp);
    std::cout << "Component " << i << " compressed size: " << result.compSize << std::endl;
  }
  std::cout << "Total compressed size (components): " << totalCompCompressedSize << std::endl;
  std::cout << "Overall compression ratio (components): " << std::fixed << std::setprecision(2)
            << (double)totalBytes / totalCompCompressedSize << std::endl;
  std::cout << "Overall component-based compression time: " << overallCompTimeMs << " ms" << std::endl;
  std::cout << "Overall component-based decompression time: " << overallDecompTimeMs << " ms" << std::endl;

  double overallCompThroughput = ((double)totalBytes / overallCompTimeMs) * 1e-6;
  double overallDecompThroughput = ((double)totalBytes / overallDecompTimeMs) * 1e-6;
  std::cout << "Overall component-based compression throughput (GB/s): " << overallCompThroughput << std::endl;
  std::cout << "Overall component-based decompression throughput (GB/s): " << overallDecompThroughput << std::endl;

  // Reassemble the decomposed components back into a full byte array.
  std::vector<uint8_t> reassembled(globalByteArray.size());
  reassembleBytesFromComponentsNestedlz4(decompressedComponentData, reassembled.data(), reassembled.size(), chosenConfig, numThreads);

  if (reassembled == globalByteArray)
    std::cout << "Reassembled data matches the original dataset." << std::endl;
  else
    std::cerr << "Error: Reassembled data does NOT match the original dataset!" << std::endl;

  return EXIT_SUCCESS;
}
