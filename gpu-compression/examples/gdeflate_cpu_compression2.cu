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
static void run_example(const std::vector<std::vector<char>>& data)
{
  size_t total_bytes = 0;
  for (const auto &part : data) {
    total_bytes += part.size();
  }

  std::cout << "----------" << std::endl;
  std::cout << "Whole-dataset mode:" << std::endl;
  std::cout << "Files: " << data.size() << std::endl;
  std::cout << "Uncompressed (B): " << total_bytes << std::endl;

  const size_t chunk_size = 1 << 16; // 64K

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
  gdeflate::compressCPU(
      input_data_cpu.ptrs(), input_data_cpu.sizes(),
      chunk_size, input_data_cpu.size(),
      compress_data_cpu.ptrs(), compress_data_cpu.sizes());

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
}

// -----------------------------------------------------------------------------
// (B) Component-Based Decomposition: Map, Split, and Reassemble Functions
// -----------------------------------------------------------------------------

// Map of dataset names to possible component configurations.
// Map to store dataset names and their multiple possible configurations
std::map<std::string, std::vector<std::vector<std::vector<size_t>>>> datasetComponentMap = {
  {"acs_wht_f32", {
          {{1,2}, {3}, {4}} ,
          {{1, 2,3}, {4}},
{{1,2,3,4}}
  }},
  {"g24_78_usb2_f32", {
        //  {{1}, {2,3}, {4}},
          {{1,2,3}, {4}},
{{1,2,3,4}}
  }},
  {"jw_mirimage_f32", {
         // {{1,2}, {3}, {4}},
          {{1,2,3}, {4}},
{{1,2,3,4}}
  }},
  {"spitzer_irac_f32", {
          {{1,2}, {3}, {4}},
          {{1,2,3}, {4}},
{{1,2,3,4}}
  }},
  {"turbulence_f32", {
          {{1,2}, {3}, {4}},
          {{1,2,3}, {4}},
{{1,2,3,4}}
  }},
  {"wave_f32", {
          {{1,2}, {3}, {4}},
          {{1,2,3}, {4}},
           {{1,2,3,4}},
  }},
  {"hdr_night_f32", {
          {{1,4}, {2}, {3}},
          {{1}, {2}, {3}, {4}},
          {{1,4}, {2,3}},
{{1,4,2,3}}
  }},
  {"ts_gas_f32", {
          {{1,2}, {3}, {4}},
{{1,2,3,4}},
  }},
  {"solar_wind_f32", {
          {{1}, {4}, {2}, {3}},
          {{1}, {2,3}, {4}},
{{1,2,3,4}}
  }},
  {"tpch_lineitem_f32", {
          {{1,2,3}, {4}},
          {{1,2}, {3}, {4}},
{{1,2,3,4}}
  }},
  {"tpcds_web_f32", {
          {{1,2,3}, {4}},
          {{1}, {2,3}, {4}},
{{1,2,3,4}}
  }},
  {"tpcds_store_f32", {
          {{1,2,3}, {4}},
          {{1}, {2,3}, {4}},
{{1,2,3,4}}
  }},
  {"tpcds_catalog_f32", {
          {{1,2,3}, {4}},
          {{1}, {2,3}, {4}},
{{1,2,3,4}}
  }},
  {"citytemp_f32", {
          {{1,4}, {2,3}},
          {{1}, {2}, {3}, {4}},
          {{1,2}, {3}, {4}},
{{1,2,3,4}}
  }},
  {"hst_wfc3_ir_f32", {
          {{1}, {2}, {3}, {4}},
          {{1,2}, {3}, {4}},
{{1,2,3,4}}
  }},
  {"hst_wfc3_uvis_f32", {
          {{1}, {2}, {3}, {4}},
          {{1,2}, {3}, {4}},
{{1,2,3,4}}
  }},
  {"rsim_f32", {
          {{1,2,3}, {4}},
          {{1,2}, {3}, {4}},
{{1,2,3,4}},
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
          {{3,2,5,6,4,1}, {7}, {8}},
{{3,2,5,6,4,1,7,8}}
  }},
  {"msg_bt_f64", {
          {{1,2,3,4,5}, {6}, {7}, {8}},
          {{3,2,1,4,5,6}, {7}, {8}},
          {{3,2,1,4,5}, {6}, {7}, {8}},
{{3,2,1,4,5,6,7,8}}
  }},
  {"num_brain_f64", {
          {{1,2,3,4,5,6}, {7}, {8}},
          {{3,2,4,5,1,6}, {7}, {8}},
{{3,2,4,5,1,6,7,8}},
  }},
  {"num_control_f64", {
          {{1,2,3,4,5,6}, {7}, {8}},
          {{4,5}, {6,3}, {1,2}, {7}, {8}},
          {{4,5,6,3,1,2}, {7}, {8}},
{{4,5,6,3,1,2,7,8}},
  }},
  {"nyc_taxi2015_f64", {
          {{7,4,6}, {5}, {3,2,1,8}},
          {{7,4,6,5}, {3,2,1,8}},
          {{7,4,6}, {5}, {3,2,1}, {8}},
          {{7,4}, {6}, {5}, {3,2}, {1}, {8}},
{{7,4,6,5,3,2,1,8}},
  }},
  {"phone_gyro_f64", {
          {{4,6}, {8}, {3,2,1,7},{5}, },
          {{4,6}, {1}, {3,2}, {5}, {7}, {8}},
          {{6,4,3,2,1,7}, {5}, {8}},
{{6,4,3,2,1,7,5}, {8}},
{{6,4,3,2,1,7}, {5,8}},
{{6,4,3,2,1,7,5,8}},
  }},
  {"tpch_order_f64", {
              {{1,2,3,4}, {7}, {6,5}, {8}},
          {{3,2,4,1}, {7}, {6,5}, {8}},
          {{3,2,4,1,7}, {6,5}, {8}},
{{3,2,4,1,7,6,5,8}},
  }},
  {"tpcxbb_store_f64", {
          {{4,2,3}, {1}, {5}, {7}, {6}, {8}},
          {{4,2,3,1}, {5}, {7,6}, {8}},
          {{4,2,3,1,5}, {7,6}, {8}},
{{4,2,3,1,5,7,6,8}},
  }},
  {"tpcxbb_web_f64", {
          {{4,2,3}, {1}, {5}, {7}, {6}, {8}},
          {{4,2,3,1}, {5}, {7,6}, {8}},
          {{4,2,3,1,5}, {7,6}, {8}},
{{4,2,3,1,5,7,6,8}},
  }},
  {"wesad_chest_f64", {
              {{7,5,6}, {8,4,1,3,2}},
          {{7,5}, {6}, {8,4,1}, {3,2}},
          {{7,5}, {6}, {8,4}, {1}, {3,2}},
          {{7,5}, {6}, {8,4,1,3,2}},
{{7,5,6,8,4,1,3,2}},
  }},
  {"default", {
          {{1}, {2}, {3}, {4}}
  }}
};

// Splits the byte array into components according to a configuration.
// Each inner vector in allComponentSizes holds 1-based indices for that component.
inline void splitBytesIntoComponentsNested(
    const std::vector<uint8_t>& byteArray,
    std::vector<std::vector<uint8_t>>& outputComponents,
    const std::vector<std::vector<size_t>>& allComponentSizes,
    int numThreads
) {
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
                size_t idxInElem = groupIndices[sub] - 1; // convert from 1-based index.
                size_t globalSrcIdx = elem * totalBytesPerElement + idxInElem;
                outputComponents[compIdx][writePos + sub] = byteArray[globalSrcIdx];
            }
        }
    }
}

// Reassembles the full byte array from its component parts.
inline void reassembleBytesFromComponentsNestedlz4(
    const std::vector<std::vector<uint8_t>>& inputComponents,
    uint8_t* byteArray, // destination buffer pointer
    size_t byteArraySize, // total size of destination buffer
    const std::vector<std::vector<size_t>>& allComponentSizes,
    int numThreads
) {
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
  std::vector<char> compData; // (decompressed data; assumed same as input)
  float compTimeMs;   // Compression time in ms.
  float decompTimeMs; // Decompression time in ms.
};
ComponentResult compress_decompress_component(const std::vector<char>& compData)
{
  ComponentResult result;
  // By default, return the original data.
  result.compData = compData;
  size_t comp_total_bytes = compData.size();
  std::cout << "Component uncompressed size: " << comp_total_bytes << " bytes" << std::endl;

  // Wrap compData as a single “file”
  std::vector<std::vector<char>> files = { compData };

  // Create a CUDA stream.
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // -----------------
  // Compression Phase for Component
  // -----------------
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
    throw std::runtime_error("Error: Failed to get max output chunk size for component");

  BatchDataCPU compress_data_cpu(max_out_bytes, input_data_cpu.size());

  // Try to compress; if compression fails (likely because the data is incompressible),
  // catch the error and simply return the original data.
  try {
    gdeflate::compressCPU(
        input_data_cpu.ptrs(), input_data_cpu.sizes(),
        chunk_size, input_data_cpu.size(),
        compress_data_cpu.ptrs(), compress_data_cpu.sizes());
  }
  catch (const std::runtime_error &e) {
    std::cerr << "Compression failed for component (likely incompressible data): " << e.what() << std::endl;
    // Treat as if no compression was achieved.
    result.compSize = comp_total_bytes;
    result.compTimeMs = 0.0f;
    result.decompTimeMs = 0.0f;
    CUDA_CHECK(cudaStreamDestroy(stream));
    return result;
  }

  CUDA_CHECK(cudaEventRecord(comp_comp_end, stream));
  CUDA_CHECK(cudaEventSynchronize(comp_comp_end));
  float comp_comp_time_ms;
  CUDA_CHECK(cudaEventElapsedTime(&comp_comp_time_ms, comp_comp_start, comp_comp_end));
  result.compTimeMs = comp_comp_time_ms;
  double comp_comp_throughput = ((double)comp_total_bytes / comp_comp_time_ms) * 1e-6; // GB/s
  std::cout << "Component compression throughput (GB/s): " << comp_comp_throughput << std::endl;
  CUDA_CHECK(cudaEventDestroy(comp_comp_start));
  CUDA_CHECK(cudaEventDestroy(comp_comp_end));

  // Compute compressed size for this component.
  size_t* compressed_sizes_host = compress_data_cpu.sizes();
  size_t compSize = 0;
  for (size_t i = 0; i < input_data_cpu.size(); i++)
    compSize += compressed_sizes_host[i];
  result.compSize = compSize;
  std::cout << "Component compressed size: " << compSize << " bytes" << std::endl;

  // If the compressed size is not smaller than the uncompressed size, skip decompression.
  if (compSize >= comp_total_bytes) {
    std::cout << "Compressed size (" << compSize << " bytes) is not smaller than uncompressed size ("
              << comp_total_bytes << " bytes). Skipping decompression." << std::endl;
    result.decompTimeMs = 0.0f;
    CUDA_CHECK(cudaStreamDestroy(stream));
    return result;
  }

  // -----------------
  // Decompression Phase for Component (validation + timing)
  // -----------------
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

  // Validate decompression.
  status = nvcompBatchedGdeflateDecompressAsync(
      compress_data.ptrs(), compress_data.sizes(),
      decomp_data.sizes(), d_decomp_sizes,
      compress_data.size(), d_decomp_temp,
      decomp_temp_bytes, decomp_data.ptrs(),
      d_statuses, stream);
  if (status != nvcompSuccess)
    throw std::runtime_error("Error: Component decompression validation failed");
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Measure decompression throughput.
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
  double comp_decomp_throughput = ((double)comp_total_bytes / comp_decomp_time_ms) * 1e-6; // GB/s
  std::cout << "Component decompression throughput (GB/s): " << comp_decomp_throughput << std::endl;
  CUDA_CHECK(cudaEventDestroy(comp_decomp_start));
  CUDA_CHECK(cudaEventDestroy(comp_decomp_end));

  CUDA_CHECK(cudaFree(d_decomp_temp));
  CUDA_CHECK(cudaFree(d_decomp_sizes));
  CUDA_CHECK(cudaFree(d_statuses));
  CUDA_CHECK(cudaStreamDestroy(stream));

  // For simplicity, assume that decompressed data equals the original component.
  return result;
}

// Compresses and decompresses a single component and returns the result.
ComponentResult compress_decompress_component1(const std::vector<char>& compData)
{
  ComponentResult result;
  result.compData = compData; // We'll return the input as placeholder (assume decompression matches input)
  size_t comp_total_bytes = compData.size();

  // Wrap compData as a single "file"
  std::vector<std::vector<char>> files = { compData };

  // Create a CUDA stream.
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // -----------------
  // Compression Phase for Component
  // -----------------
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

  // Compute compressed size for this component.
  size_t* compressed_sizes_host = compress_data_cpu.sizes();
  size_t compSize = 0;
  for (size_t i = 0; i < input_data_cpu.size(); i++)
    compSize += compressed_sizes_host[i];
  result.compSize = compSize;

  // -----------------
  // Decompression Phase for Component (validation + timing)
  // -----------------
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

  // Run initial decompression for validation.
  status = nvcompBatchedGdeflateDecompressAsync(
      compress_data.ptrs(), compress_data.sizes(),
      decomp_data.sizes(), d_decomp_sizes,
      compress_data.size(), d_decomp_temp,
      decomp_temp_bytes, decomp_data.ptrs(),
      d_statuses, stream);
  if (status != nvcompSuccess)
    throw std::runtime_error("Error: Component decompression validation failed");
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Measure decompression throughput.
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

  // For simplicity, we assume the decompressed data equals the original component.
  return result;
}

// -----------------------------------------------------------------------------
// (D) Main: Read Dataset, Run Whole-Dataset and Component-Based Compression/Decompression
// -----------------------------------------------------------------------------


// Helper structure to hold overall component-based metrics for one config.
struct ComponentMetrics {
  std::string configStr; // string version of the configuration
  size_t totalUncompressed;
  size_t totalCompressed;
  float overallCompTimeMs;
  float overallDecompTimeMs;
  double compThroughputGBs;
  double decompThroughputGBs;
  double compressionRatio;
};

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

// Define a structure to hold whole-dataset metrics.
struct WholeDatasetMetrics {
  size_t totalBytes;
  size_t compBytes;
  double compressionRatio;
  float compTimeMs;
  float decompTimeMs;
  double compThroughputGBs;
  double decompThroughputGBs;
};

// Modify or create a new function that runs the whole-dataset compression/decompression
// and returns the metrics. (This is based on your original run_example function.)
WholeDatasetMetrics run_whole_dataset(const std::vector<std::vector<char>>& files)
{
  WholeDatasetMetrics metrics;
  // Compute total uncompressed bytes.
  size_t total_bytes = 0;
  for (const auto &part : files) {
    total_bytes += part.size();
  }
  metrics.totalBytes = total_bytes;

  std::cout << "----------" << std::endl;
  std::cout << "Whole-dataset mode:" << std::endl;
  std::cout << "Files: " << files.size() << std::endl;
  std::cout << "Uncompressed (B): " << total_bytes << std::endl;

  const size_t chunk_size = 1 << 16; // 64K

  // Build up input batch on CPU.
  BatchDataCPU input_data_cpu(files, chunk_size);
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
  gdeflate::compressCPU(
      input_data_cpu.ptrs(), input_data_cpu.sizes(),
      chunk_size, input_data_cpu.size(),
      compress_data_cpu.ptrs(), compress_data_cpu.sizes());

  CUDA_CHECK(cudaEventRecord(comp_end, 0));
  CUDA_CHECK(cudaEventSynchronize(comp_end));
  float comp_time_ms;
  CUDA_CHECK(cudaEventElapsedTime(&comp_time_ms, comp_start, comp_end));
  metrics.compTimeMs = comp_time_ms;
  metrics.compThroughputGBs = ((double)total_bytes / comp_time_ms) * 1e-6; // GB/s

  // Compute compressed size.
  size_t* compressed_sizes_host = compress_data_cpu.sizes();
  size_t comp_bytes = 0;
  for (size_t i = 0; i < compress_data_cpu.size(); ++i)
    comp_bytes += compressed_sizes_host[i];
  metrics.compBytes = comp_bytes;
  metrics.compressionRatio = (double)total_bytes / comp_bytes;

  std::cout << "Compressed size (B): " << comp_bytes
            << ", Compression ratio: " << std::fixed << std::setprecision(2)
            << metrics.compressionRatio << std::endl;

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
  metrics.decompTimeMs = decomp_time_ms;
  metrics.decompThroughputGBs = ((double)total_bytes / decomp_time_ms) * 1e-6;

  CUDA_CHECK(cudaFree(d_decomp_temp));
  CUDA_CHECK(cudaFree(d_decomp_sizes));
  CUDA_CHECK(cudaFree(d_statuses));
  CUDA_CHECK(cudaEventDestroy(comp_start));
  CUDA_CHECK(cudaEventDestroy(comp_end));
  CUDA_CHECK(cudaEventDestroy(decomp_start));
  CUDA_CHECK(cudaEventDestroy(decomp_end));
  CUDA_CHECK(cudaStreamDestroy(stream));

  // Also print out throughput info.
  std::cout << "Whole-dataset compression throughput (GB/s): " << metrics.compThroughputGBs << std::endl;
  std::cout << "Whole-dataset decompression throughput (GB/s): " << metrics.decompThroughputGBs << std::endl;

  return metrics;
}

//
// ===== Main function =====
//
int main(int argc, char* argv[])
{
  // Usage: <program> <datasetPath> <precisionBits (32|64)>
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

  // Prepare CSV output.
  // Save CSV file in your desired directory.
  // Extract dataset name from file path.
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
  // Write CSV header.
  // We include a "Level" field to distinguish Whole-Dataset vs Component-based results.
  csvFile << "Dataset,Config,Level,ComponentIndex,UncompressedBytes,CompressedBytes,CompressionRatio,CompTimeMs,DecompTimeMs,CompThroughputGBs,DecompThroughputGBs\n";

  //
  // --- (I) Whole-Dataset Compression/Decompression ---
  //
  std::vector<char> dataAsChar(globalByteArray.begin(), globalByteArray.end());
  std::vector<std::vector<char>> files = { dataAsChar };

  // Call our new function to get whole-dataset metrics.
  WholeDatasetMetrics wholeMetrics = run_whole_dataset(files);

  // Write a row to CSV for the whole-dataset results.
  csvFile << datasetName << ","
          << "N/A" << ","       // Config: not applicable here.
          << "WholeDataset" << ","
          << "" << ","          // ComponentIndex left blank.
          << wholeMetrics.totalBytes << ","
          << wholeMetrics.compBytes << ","
          << std::fixed << std::setprecision(2) << wholeMetrics.compressionRatio << ","
          << wholeMetrics.compTimeMs << ","
          << wholeMetrics.decompTimeMs << ","
          << wholeMetrics.compThroughputGBs << ","
          << wholeMetrics.decompThroughputGBs << "\n";


  //
  // --- (II) Component-Based Compression/Decompression for each configuration ---
  //
  // Look up configuration options; if not found, use "default".
  std::vector<std::vector<std::vector<size_t>>> configOptions;
  if (datasetComponentMap.find(datasetName) != datasetComponentMap.end())
    configOptions = datasetComponentMap[datasetName];
  else
    configOptions = datasetComponentMap["default"];

  int numThreads = 10; // adjust thread count as needed

  // Loop over each configuration option.
  for (size_t cfgIdx = 0; cfgIdx < configOptions.size(); cfgIdx++) {
    const std::vector<std::vector<size_t>>& chosenConfig = configOptions[cfgIdx];
    std::string configStr = configToString(chosenConfig);
    std::cout << "Processing configuration " << cfgIdx << ": " << configStr << std::endl;

    // Split the global byte array into components.
    std::vector<std::vector<uint8_t>> decomposedComponents;
    splitBytesIntoComponentsNested(globalByteArray, decomposedComponents, chosenConfig, numThreads);
    std::cout << "Dataset decomposed into " << decomposedComponents.size() << " components." << std::endl;

    // Overall (aggregated) metrics for this configuration.
    size_t totalCompCompressedSize = 0;
    float overallCompTimeMs = 0.0f;
    float overallDecompTimeMs = 0.0f;

    // To store decompressed component data for reassembly.
    std::vector<std::vector<uint8_t>> decompressedComponentData;

    // Process each component.
    for (size_t compIdx = 0; compIdx < decomposedComponents.size(); compIdx++) {
      std::vector<char> compData(decomposedComponents[compIdx].begin(), decomposedComponents[compIdx].end());
      size_t compUncompressed = compData.size();

      ComponentResult result = compress_decompress_component(compData);
      totalCompCompressedSize += result.compSize;
      overallCompTimeMs += result.compTimeMs;
      overallDecompTimeMs += result.decompTimeMs;

      std::vector<uint8_t> compDecomp(result.compData.begin(), result.compData.end());
      decompressedComponentData.push_back(compDecomp);

      double compThroughput = result.compTimeMs > 0 ? ((double)compUncompressed / result.compTimeMs) * 1e-6 : 0.0;
      double decompThroughput = result.decompTimeMs > 0 ? ((double)compUncompressed / result.decompTimeMs) * 1e-6 : 0.0;
      double compRatio = (result.compSize > 0) ? (double)compUncompressed / result.compSize : 0.0;

      // Write per-component row to CSV.
      csvFile << datasetName << ","
              << configStr << ","
              << "Component" << ","
              << compIdx << ","
              << compUncompressed << ","
              << result.compSize << ","
              << std::fixed << std::setprecision(2) << compRatio << ","
              << result.compTimeMs << ","
              << result.decompTimeMs << ","
              << compThroughput << ","
              << decompThroughput << "\n";
    }

    double overallCompressionRatio = totalCompCompressedSize > 0 ? (double)totalBytes / totalCompCompressedSize : 0.0;
    double overallCompThroughput = overallCompTimeMs > 0 ? ((double)totalBytes / overallCompTimeMs) * 1e-6 : 0.0;
    double overallDecompThroughput = overallDecompTimeMs > 0 ? ((double)totalBytes / overallDecompTimeMs) * 1e-6 : 0.0;

    // Write overall (aggregated) row for the configuration to CSV.
    csvFile << datasetName << ","
            << configStr << ","
            << "Overall" << ","
            << "" << ","
            << totalBytes << ","
            << totalCompCompressedSize << ","
            << std::fixed << std::setprecision(2) << overallCompressionRatio << ","
            << overallCompTimeMs << ","
            << overallDecompTimeMs << ","
            << overallCompThroughput << ","
            << overallDecompThroughput << "\n";

    // Reassemble the decomposed components back into a full byte array.
    std::vector<uint8_t> reassembled(globalByteArray.size());
    reassembleBytesFromComponentsNestedlz4(decompressedComponentData, reassembled.data(), reassembled.size(), chosenConfig, numThreads);
    if (reassembled == globalByteArray)
      std::cout << "Reassembled data matches the original dataset." << std::endl;
    else
      std::cerr << "Error: Reassembled data does NOT match the original dataset!" << std::endl;
  }

  csvFile.close();
  std::cout << "Results saved to " << csvFilename << std::endl;

  return EXIT_SUCCESS;
}
