// gdeflate_cpu_compression.cu

/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024 NVIDIA CORPORATION
 * All rights reserved. SPDX-LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material
 * without an express license agreement from NVIDIA CORPORATION or its affiliates
 * is strictly prohibited.
 */

#include <nvcomp/deflate.h>
#include <nvcomp.h>
#include <zlib.h>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

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
#include <numeric>
#include <functional>
#include <cstring>
#ifdef _OPENMP
#include <omp.h>
#endif

//---------------------------------------------------------
// CUDA error checking macro
//---------------------------------------------------------
#ifndef CUDA_CHECK
#define CUDA_CHECK(func)                                                       \
  do {                                                                         \
    cudaError_t rt = (func);                                                   \
    if (rt != cudaSuccess) {                                                   \
      std::cerr << "CUDA API call failure \"" #func "\" with error: "          \
                << cudaGetErrorString(rt) << " at " << __FILE__ << ":"         \
                << __LINE__ << std::endl;                                      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)
#endif

//==============================
// CPU side: Compress one chunk using zlib raw deflate.
//==============================
static std::vector<uint8_t> cpuDeflateCompressOneChunk(const uint8_t* src_ptr, size_t src_size)
{
  // Minimal demonstration of raw DEFLATE using zlib.
  std::vector<uint8_t> outBuffer;
  outBuffer.resize(compressBound(src_size)); // oversize buffer

  z_stream strm;
  std::memset(&strm, 0, sizeof(z_stream));
  // Use negative windowBits for raw deflate, e.g. -15
  if (deflateInit2(&strm, Z_DEFAULT_COMPRESSION, Z_DEFLATED, -15, 8,
                   Z_DEFAULT_STRATEGY) != Z_OK)
  {
    throw std::runtime_error("deflateInit2() failed");
  }

  strm.avail_in  = static_cast<uInt>(src_size);
  strm.next_in   = const_cast<Bytef*>(reinterpret_cast<const Bytef*>(src_ptr));
  strm.avail_out = static_cast<uInt>(outBuffer.size());
  strm.next_out  = reinterpret_cast<Bytef*>(outBuffer.data());

  int ret = deflate(&strm, Z_FINISH);
  if (ret != Z_STREAM_END) {
    deflateEnd(&strm);
    throw std::runtime_error("deflate() did not return Z_STREAM_END");
  }
  size_t compSize = strm.total_out;
  deflateEnd(&strm);

  outBuffer.resize(compSize);
  return outBuffer;
}

// Forward-declare GPU BatchData
class BatchData;

//==============================
// BatchDataCPU class definition
//==============================
class BatchDataCPU
{
public:
  /**
   * @brief Constructor that splits the input files into 64KB (or user-specified) chunks
   *        and copies the data into the CPU batch buffers.
   */
  BatchDataCPU(const std::vector<std::vector<char>>& host_data, size_t chunk_size)
    : m_ptrs(), m_sizes(), m_data(), m_size(0)
  {
    // First pass: count how many chunks in total
    // and the sum of their sizes
    size_t totalBytes = 0;
    for (auto& file : host_data) {
      size_t fileSize = file.size();
      // number of chunks for this file
      size_t numChunks = (fileSize + chunk_size - 1) / chunk_size;
      m_size += numChunks;
      totalBytes += fileSize;
    }

    // We'll fill m_sizes with the "actual" sizes per chunk
    m_sizes.reserve(m_size);

    // second pass: figure out exact chunk sizes
    for (auto& file : host_data) {
      size_t fileSize = file.size();
      const size_t numChunks = (fileSize + chunk_size - 1) / chunk_size;
      size_t remaining = fileSize;
      for (size_t c = 0; c < numChunks; c++) {
        size_t thisChunkSize = std::min<size_t>(remaining, chunk_size);
        m_sizes.push_back(thisChunkSize);
        remaining -= thisChunkSize;
      }
    }

    // total size of all chunks combined
    size_t totalChunkBytes = 0;
    for (auto sz : m_sizes) {
      totalChunkBytes += sz;
    }
    m_data.resize(totalChunkBytes);

    // create pointer array and copy chunk data from each file
    m_ptrs.resize(m_size);

    size_t currentOffset = 0;
    size_t chunkIndex = 0;
    for (auto& file : host_data) {
      size_t fileSize = file.size();
      const char* srcPtr = file.data();
      size_t remaining = fileSize;

      while (remaining > 0) {
        size_t thisChunkSize = std::min<size_t>(remaining, chunk_size);
        m_ptrs[chunkIndex] = m_data.data() + currentOffset;

        // copy the file bytes into chunk
        std::memcpy(m_ptrs[chunkIndex], srcPtr, thisChunkSize);

        currentOffset += thisChunkSize;
        srcPtr        += thisChunkSize;
        remaining     -= thisChunkSize;
        chunkIndex++;
      }
    }
  }

  // 2) Constructor for fixed-size allocation
  BatchDataCPU(size_t max_output_size, size_t batch_size)
    : m_ptrs(), m_sizes(), m_data(), m_size(batch_size)
  {
    m_data.resize(max_output_size * m_size);
    m_sizes.resize(m_size, max_output_size);
    m_ptrs.resize(m_size);
    for (size_t i = 0; i < m_size; ++i) {
      m_ptrs[i] = m_data.data() + max_output_size * i;
    }
  }

  // 3) Constructor from device pointers with optional copy
  BatchDataCPU(const void* const* in_ptrs,
               const size_t* in_sizes,
               const uint8_t* /*in_data*/,
               size_t in_size,
               bool copy_data = false)
    : m_ptrs(), m_sizes(), m_data(), m_size(in_size)
  {
    m_sizes.resize(m_size);
    CUDA_CHECK(cudaMemcpy(m_sizes.data(), in_sizes,
                          m_size * sizeof(*in_sizes),
                          cudaMemcpyDeviceToHost));

    const size_t total =
        std::accumulate(m_sizes.begin(), m_sizes.end(),
                        static_cast<size_t>(0));
    m_data.resize(total);
    m_ptrs.resize(m_size);

    size_t offset = 0;
    for (size_t i = 0; i < m_size; ++i) {
      m_ptrs[i] = m_data.data() + offset;
      offset += m_sizes[i];
    }

    if (copy_data) {
      std::vector<void*> hostPtrs(m_size);
      CUDA_CHECK(cudaMemcpy(hostPtrs.data(), in_ptrs,
                            m_size * sizeof(void*),
                            cudaMemcpyDeviceToHost));
      for (size_t i = 0; i < m_size; i++) {
        CUDA_CHECK(cudaMemcpy(m_ptrs[i], hostPtrs[i],
                              m_sizes[i], cudaMemcpyDeviceToHost));
      }
    }
  }

  // 4) Move Constructor
  BatchDataCPU(BatchDataCPU&& other) = default;

  // 5) Construct from GPU BatchData (defined later)
  BatchDataCPU(const BatchData& batch_data, bool copy_data = false);

  // 6) Copy Constructor
  BatchDataCPU(const BatchDataCPU& other)
    : m_ptrs(), m_sizes(other.m_sizes), m_data(other.m_data), m_size(other.m_size)
  {
    m_ptrs.resize(m_size);
    size_t offset = 0;
    for (size_t i = 0; i < m_size; ++i) {
      m_ptrs[i] = m_data.data() + offset;
      offset += m_sizes[i];
    }
  }

  // 7) Copy Assignment Operator
  BatchDataCPU& operator=(const BatchDataCPU& other)
  {
    if (this != &other) {
      m_size  = other.m_size;
      m_sizes = other.m_sizes;
      m_data  = other.m_data;
      m_ptrs.resize(m_size);

      size_t offset = 0;
      for (size_t i = 0; i < m_size; ++i) {
        m_ptrs[i] = m_data.data() + offset;
        offset += m_sizes[i];
      }
    }
    return *this;
  }

  // Accessors
  uint8_t* data()             { return m_data.data(); }
  const uint8_t* data() const { return m_data.data(); }

  void** ptrs()                     { return m_ptrs.data(); }
  const void* const* ptrs() const   { return m_ptrs.data(); }

  size_t* sizes()                   { return m_sizes.data(); }
  const size_t* sizes() const       { return m_sizes.data(); }

  size_t size() const { return m_size; }

private:
  std::vector<void*>   m_ptrs;
  std::vector<size_t>  m_sizes;
  std::vector<uint8_t> m_data;
  size_t               m_size;
};

inline bool operator==(const BatchDataCPU& lhs, const BatchDataCPU& rhs)
{
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (size_t i = 0; i < lhs.size(); ++i) {
    if (lhs.sizes()[i] != rhs.sizes()[i]) {
      return false;
    }
    const uint8_t* lhs_ptr = reinterpret_cast<const uint8_t*>(lhs.ptrs()[i]);
    const uint8_t* rhs_ptr = reinterpret_cast<const uint8_t*>(rhs.ptrs()[i]);
    for (size_t j = 0; j < lhs.sizes()[i]; ++j) {
      if (lhs_ptr[j] != rhs_ptr[j]) {
        return false;
      }
    }
  }
  return true;
}

//==============================
// GPU BatchData class definition
//==============================
class BatchData
{
public:
  // Construct a BatchData from a CPU batch (copy data from host to device)
  BatchData(const BatchDataCPU& cpu, bool copy_data)
  {
    m_batch_size = cpu.size();

    // Allocate device array for sizes
    CUDA_CHECK(cudaMalloc(&d_sizes, m_batch_size * sizeof(size_t)));
    CUDA_CHECK(cudaMemcpy(d_sizes, cpu.sizes(),
                          m_batch_size * sizeof(size_t),
                          cudaMemcpyHostToDevice));

    // Allocate device memory for each chunk
    std::vector<void*> host_ptrs(m_batch_size);
    for (size_t i = 0; i < m_batch_size; i++) {
      size_t s = cpu.sizes()[i];
      void* d_chunk = nullptr;
      CUDA_CHECK(cudaMalloc(&d_chunk, s));
      if (copy_data) {
        CUDA_CHECK(cudaMemcpy(d_chunk, cpu.ptrs()[i],
                              s, cudaMemcpyHostToDevice));
      }
      host_ptrs[i] = d_chunk;
    }
    // Allocate device array for pointers
    CUDA_CHECK(cudaMalloc(&d_ptrs, m_batch_size * sizeof(void*)));
    CUDA_CHECK(cudaMemcpy(d_ptrs, host_ptrs.data(),
                          m_batch_size * sizeof(void*),
                          cudaMemcpyHostToDevice));
  }

  ~BatchData()
  {
    if (d_ptrs) {
      std::vector<void*> host_ptrs(m_batch_size);
      CUDA_CHECK(cudaMemcpy(host_ptrs.data(),
                            d_ptrs,
                            m_batch_size * sizeof(void*),
                            cudaMemcpyDeviceToHost));
      for (size_t i = 0; i < m_batch_size; i++) {
        if (host_ptrs[i]) {
          cudaFree(host_ptrs[i]);
        }
      }
      cudaFree(d_ptrs);
    }
    if (d_sizes) {
      cudaFree(d_sizes);
    }
  }

  size_t size() const { return m_batch_size; }

  // For nvcomp, we cast this to (void* const*) when calling the API.
  const void* const* ptrs() const { return (const void* const*)d_ptrs; }
  const size_t* sizes()    const { return d_sizes; }

private:
  void*   d_ptrs      = nullptr;
  size_t* d_sizes     = nullptr;
  size_t  m_batch_size = 0;
};

//----------------------------
// Definition: CPU from GPU BatchData
//----------------------------
BatchDataCPU::BatchDataCPU(const BatchData& batch_data, bool copy_data)
  : m_size(batch_data.size())
{
  m_sizes.resize(m_size);
  CUDA_CHECK(cudaMemcpy(m_sizes.data(), batch_data.sizes(),
                        m_size * sizeof(size_t),
                        cudaMemcpyDeviceToHost));

  size_t total_data = 0;
  for (auto sz : m_sizes) {
    total_data += sz;
  }
  m_data.resize(total_data);
  m_ptrs.resize(m_size);

  size_t offset = 0;
  for (size_t i = 0; i < m_size; i++) {
    m_ptrs[i] = m_data.data() + offset;
    offset += m_sizes[i];
  }
  if (copy_data) {
    std::vector<void*> hostPtrs(m_size);
    CUDA_CHECK(cudaMemcpy(hostPtrs.data(), batch_data.ptrs(),
                          m_size * sizeof(void*),
                          cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < m_size; i++) {
      CUDA_CHECK(cudaMemcpy(m_ptrs[i], hostPtrs[i],
                            m_sizes[i], cudaMemcpyDeviceToHost));
    }
  }
}

//==============================
// Additional Utility
//==============================
inline float measureCudaTime(std::function<void(cudaStream_t)> kernelFunc,
                             cudaStream_t stream)
{
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start, stream));
  kernelFunc(stream);
  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms;
}

__global__ void compareBuffers(const uint8_t* a, const uint8_t* b,
                               int* invalid, size_t n)
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = gridDim.x * blockDim.x;
  while (i < n) {
    if (a[i] != b[i]) {
      *invalid = 1;
    }
    i += stride;
  }
}

//==============================
// Whole-Dataset Metrics
//==============================
struct WholeDatasetMetrics
{
  size_t totalBytes;
  size_t compBytes;
  double compressionRatio;
  float  compTimeMs;
  float  decompTimeMs;
  double compThroughputGBs;
  double decompThroughputGBs;
};

//==============================
// Splitting out-of-band (like before)
inline void splitBytesIntoComponentsNested(
    const std::vector<uint8_t>& byteArray,
    std::vector<std::vector<uint8_t>>& outputComponents,
    const std::vector<std::vector<size_t>>& allComponentSizes,
    int numThreads)
{
  size_t totalBytesPerElement = 0;
  for (const auto &group : allComponentSizes) {
    totalBytesPerElement += group.size();
  }

  size_t numElements = byteArray.size() / totalBytesPerElement;
  outputComponents.resize(allComponentSizes.size());
  for (size_t i = 0; i < allComponentSizes.size(); i++) {
    outputComponents[i].resize(numElements * allComponentSizes[i].size());
  }

#ifdef _OPENMP
#pragma omp parallel for num_threads(numThreads)
#endif
  for (size_t elem = 0; elem < numElements; elem++) {
    for (size_t compIdx = 0; compIdx < allComponentSizes.size(); compIdx++) {
      const auto& groupIndices = allComponentSizes[compIdx];
      size_t groupSize = groupIndices.size();
      size_t writePos  = elem * groupSize;
      for (size_t sub = 0; sub < groupSize; sub++) {
        size_t idxInElem    = groupIndices[sub] - 1;
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
  for (const auto &group : allComponentSizes) {
    totalBytesPerElement += group.size();
  }

  size_t numElements = byteArraySize / totalBytesPerElement;
#ifdef _OPENMP
#pragma omp parallel for num_threads(numThreads)
#endif
  for (size_t compIdx = 0; compIdx < inputComponents.size(); compIdx++) {
    const auto& groupIndices = allComponentSizes[compIdx];
    const auto& componentData = inputComponents[compIdx];
    size_t groupSize = groupIndices.size();
    for (size_t elem = 0; elem < numElements; elem++) {
      size_t readPos = elem * groupSize;
      for (size_t sub = 0; sub < groupSize; sub++) {
        size_t idxInElem   = groupIndices[sub] - 1;
        size_t globalIndex = elem * totalBytesPerElement + idxInElem;
        byteArray[globalIndex] = componentData[readPos + sub];
      }
    }
  }
}

//==============================
// Whole dataset test function
//==============================
WholeDatasetMetrics run_whole_dataset_deflate(const std::vector<std::vector<char>>& files)
{
  WholeDatasetMetrics metrics{};
  size_t total_bytes = 0;
  for (const auto& part : files) {
    total_bytes += part.size();
  }
  metrics.totalBytes = total_bytes;
  std::cout << "----------\nWhole-dataset mode (CPU zlib + GPU deflate):\n"
            << "Files: " << files.size() << "\n"
            << "Uncompressed (B): " << total_bytes << "\n";

  // 64KB chunk
  const size_t chunk_size = 1 << 16;
  // Actually copy file data into chunked CPU buffers
  BatchDataCPU input_data_cpu(files, chunk_size);
  std::cout << "Chunks: " << input_data_cpu.size() << "\n";

  // (1) CPU zlib compression
  float comp_time_ms = 0.0f;
  size_t total_comp_bytes = 0;
  std::vector<std::vector<uint8_t>> comp_chunks(input_data_cpu.size());

  {
    cudaEvent_t startE, stopE;
    CUDA_CHECK(cudaEventCreate(&startE));
    CUDA_CHECK(cudaEventCreate(&stopE));
    CUDA_CHECK(cudaEventRecord(startE, 0));

    for (size_t i = 0; i < input_data_cpu.size(); i++) {
      size_t src_size = input_data_cpu.sizes()[i];
      const uint8_t* src_ptr =
          reinterpret_cast<const uint8_t*>(input_data_cpu.ptrs()[i]);
      std::vector<uint8_t> cdata =
          cpuDeflateCompressOneChunk(src_ptr, src_size);
      total_comp_bytes += cdata.size();
      comp_chunks[i] = std::move(cdata);
    }

    CUDA_CHECK(cudaEventRecord(stopE, 0));
    CUDA_CHECK(cudaEventSynchronize(stopE));
    CUDA_CHECK(cudaEventElapsedTime(&comp_time_ms, startE, stopE));
    CUDA_CHECK(cudaEventDestroy(startE));
    CUDA_CHECK(cudaEventDestroy(stopE));

    metrics.compBytes = total_comp_bytes;
    metrics.compressionRatio =
        (double)total_bytes / (double)total_comp_bytes;
    std::cout << "Compressed size (B): " << total_comp_bytes
              << ", Ratio: " << metrics.compressionRatio << "\n";
  }

  metrics.compTimeMs = comp_time_ms;
  metrics.compThroughputGBs =
      (comp_time_ms > 0.0f)
          ? ((double)total_bytes / comp_time_ms) * 1e-6
          : 0.0;

  // (2) GPU deflate decompression
  float decomp_time_ms = 0.0f;
  {
    size_t batch_count = input_data_cpu.size();

    // Pack variable-length compressed chunks to uniform batch
    size_t largest_chunk = 0;
    for (size_t i = 0; i < batch_count; i++) {
      if (comp_chunks[i].size() > largest_chunk) {
        largest_chunk = comp_chunks[i].size();
      }
    }
    // Build a CPU structure for the compressed data
    BatchDataCPU compress_data_cpu(largest_chunk, batch_count);
    for (size_t i = 0; i < batch_count; i++) {
      size_t csize = comp_chunks[i].size();
      compress_data_cpu.sizes()[i] = csize;
      std::memcpy(compress_data_cpu.ptrs()[i],
                  comp_chunks[i].data(), csize);
    }

    // Copy compressed data to GPU
    BatchData compress_data(compress_data_cpu, true);

    // Allocate GPU buffer for decompression output
    BatchData decomp_data(input_data_cpu, false);

    cudaStream_t stream_gpu;
    CUDA_CHECK(cudaStreamCreate(&stream_gpu));

    size_t decomp_temp_bytes = 0;
    nvcompStatus_t status =
      nvcompBatchedDeflateDecompressGetTempSize(batch_count, chunk_size,
                                                &decomp_temp_bytes);
    if (status != nvcompSuccess) {
      throw std::runtime_error("nvcompBatchedDeflateDecompressGetTempSize() failed");
    }

    void* d_decomp_temp = nullptr;
    CUDA_CHECK(cudaMalloc(&d_decomp_temp, decomp_temp_bytes));

    size_t* d_decomp_sizes = nullptr;
    CUDA_CHECK(cudaMalloc(&d_decomp_sizes, batch_count * sizeof(size_t)));

    nvcompStatus_t* d_statuses = nullptr;
    CUDA_CHECK(cudaMalloc(&d_statuses, batch_count * sizeof(nvcompStatus_t)));

    cudaEvent_t startE, stopE;
    CUDA_CHECK(cudaEventCreate(&startE));
    CUDA_CHECK(cudaEventCreate(&stopE));
    CUDA_CHECK(cudaEventRecord(startE, stream_gpu));

    // Decompress
    status = nvcompBatchedDeflateDecompressAsync(
        compress_data.ptrs(),
        compress_data.sizes(),
        input_data_cpu.sizes(),  // actual uncompressed sizes
        d_decomp_sizes,
        batch_count,
        d_decomp_temp,
        decomp_temp_bytes,
        (void* const*)decomp_data.ptrs(),  // cast to match signature
        d_statuses,
        stream_gpu);

    if (status != nvcompSuccess) {
      throw std::runtime_error("nvcompBatchedDeflateDecompressAsync() failed");
    }

    CUDA_CHECK(cudaEventRecord(stopE, stream_gpu));
    CUDA_CHECK(cudaStreamSynchronize(stream_gpu));
    CUDA_CHECK(cudaEventElapsedTime(&decomp_time_ms, startE, stopE));
    CUDA_CHECK(cudaEventDestroy(startE));
    CUDA_CHECK(cudaEventDestroy(stopE));

    CUDA_CHECK(cudaFree(d_decomp_temp));
    CUDA_CHECK(cudaFree(d_decomp_sizes));
    CUDA_CHECK(cudaFree(d_statuses));
    CUDA_CHECK(cudaStreamDestroy(stream_gpu));

    // Validate
    BatchDataCPU final_decomp_cpu(decomp_data, true);
    if (!(final_decomp_cpu == input_data_cpu)) {
      throw std::runtime_error(
          "ERROR: Decompressed data does not match original!");
    } else {
      std::cout << "Decompression validated :)\n";
    }
  }
  metrics.decompTimeMs = decomp_time_ms;
  metrics.decompThroughputGBs =
      (decomp_time_ms > 0.0f)
          ? ((double)total_bytes / decomp_time_ms) * 1e-6
          : 0.0;

  std::cout << "Whole-dataset Compression Throughput (GB/s): "
            << metrics.compThroughputGBs << "\n"
            << "Whole-dataset Decompression Throughput (GB/s): "
            << metrics.decompThroughputGBs << "\n";

  return metrics;
}

//==============================
// Component-Based Flow
//==============================
struct ComponentResult
{
  size_t compSize;
  std::vector<char> compData;
  float compTimeMs;
  float decompTimeMs;
};

ComponentResult compress_decompress_component_deflate(const std::vector<char>& compData)
{
  ComponentResult result{};
  result.compData = compData;
  const size_t comp_total_bytes = compData.size();
  std::cout << "Component uncompressed size: " << comp_total_bytes << " bytes\n";

  const size_t chunk_size = 1 << 16;
  std::vector<std::vector<char>> files = { compData };
  BatchDataCPU input_data_cpu(files, chunk_size);
  size_t batch_count = input_data_cpu.size();

  // (1) CPU zlib compression
  float comp_time_ms = 0.0f;
  size_t total_comp_bytes = 0;
  std::vector<std::vector<uint8_t>> comp_chunks(batch_count);

  {
    cudaEvent_t startE, stopE;
    CUDA_CHECK(cudaEventCreate(&startE));
    CUDA_CHECK(cudaEventCreate(&stopE));
    CUDA_CHECK(cudaEventRecord(startE, 0));

    for (size_t i = 0; i < batch_count; i++) {
      size_t src_size = input_data_cpu.sizes()[i];
      const uint8_t* src_ptr =
          reinterpret_cast<const uint8_t*>(input_data_cpu.ptrs()[i]);
      std::vector<uint8_t> cdata =
          cpuDeflateCompressOneChunk(src_ptr, src_size);
      total_comp_bytes += cdata.size();
      comp_chunks[i] = std::move(cdata);
    }

    CUDA_CHECK(cudaEventRecord(stopE, 0));
    CUDA_CHECK(cudaEventSynchronize(stopE));
    CUDA_CHECK(cudaEventElapsedTime(&comp_time_ms, startE, stopE));
    CUDA_CHECK(cudaEventDestroy(startE));
    CUDA_CHECK(cudaEventDestroy(stopE));
  }

  result.compTimeMs = comp_time_ms;
  result.compSize   = total_comp_bytes;

  double comp_throughput =
      (comp_time_ms > 0.0f)
          ? ((double)comp_total_bytes / comp_time_ms) * 1e-6
          : 0.0;
  std::cout << "Component Compression Throughput (GB/s): "
            << comp_throughput << "\n";

  if (total_comp_bytes >= comp_total_bytes) {
    std::cout << "Compressed size >= uncompressed. Skipping decompression.\n";
    result.decompTimeMs = 0.0f;
    return result;
  }

  // (2) GPU decompression
  float decomp_time_ms = 0.0f;
  {
    auto buildCPUfromChunks =
      [&](const std::vector<std::vector<uint8_t>>& chunks) -> BatchDataCPU
    {
      // find the largest chunk
      size_t largest_chunk = 0;
      for (auto &c : chunks) {
        if (c.size() > largest_chunk) {
          largest_chunk = c.size();
        }
      }
      BatchDataCPU ret(largest_chunk, chunks.size());
      for (size_t i = 0; i < chunks.size(); i++) {
        size_t csize = chunks[i].size();
        ret.sizes()[i] = csize;
        std::memcpy(ret.ptrs()[i],
                    chunks[i].data(), csize);
      }
      return ret;
    };

    // pack the variable-length compressed chunks
    BatchDataCPU compress_data_cpu = buildCPUfromChunks(comp_chunks);
    // copy compressed data to GPU
    BatchData compress_data(compress_data_cpu, true);

    // allocate GPU buffer for decompressed
    BatchData decomp_data(input_data_cpu, false);

    nvcompStatus_t status;
    size_t temp_bytes = 0;
    status = nvcompBatchedDeflateDecompressGetTempSize(batch_count,
                                                       chunk_size,
                                                       &temp_bytes);
    if (status != nvcompSuccess) {
      throw std::runtime_error("nvcompBatchedDeflateDecompressGetTempSize() failed for component");
    }

    void* d_temp = nullptr;
    CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));
    size_t* d_decomp_sizes = nullptr;
    CUDA_CHECK(cudaMalloc(&d_decomp_sizes, batch_count * sizeof(size_t)));

    cudaEvent_t startE, stopE;
    CUDA_CHECK(cudaEventCreate(&startE));
    CUDA_CHECK(cudaEventCreate(&stopE));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    CUDA_CHECK(cudaEventRecord(startE, stream));

    // Decompress
    status = nvcompBatchedDeflateDecompressAsync(
        compress_data.ptrs(),
        compress_data.sizes(),
        input_data_cpu.sizes(),
        d_decomp_sizes,
        batch_count,
        d_temp,
        temp_bytes,
        (void* const*)decomp_data.ptrs(),
        nullptr,
        stream);

    if (status != nvcompSuccess) {
      throw std::runtime_error("nvcompBatchedDeflateDecompressAsync() failed for component");
    }

    CUDA_CHECK(cudaEventRecord(stopE, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaEventElapsedTime(&decomp_time_ms, startE, stopE));

    CUDA_CHECK(cudaEventDestroy(startE));
    CUDA_CHECK(cudaEventDestroy(stopE));
    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaFree(d_temp));
    CUDA_CHECK(cudaFree(d_decomp_sizes));

    result.decompTimeMs = decomp_time_ms;
    double dec_throughput =
        (decomp_time_ms > 0.0f)
            ? ((double)comp_total_bytes / decomp_time_ms) * 1e-6
            : 0.0;
    std::cout << "Component Decompression Throughput (GB/s): "
              << dec_throughput << "\n";

    // Copy decompressed data back to CPU
    BatchDataCPU final_decomp_cpu(decomp_data, true);
    result.compData.resize(final_decomp_cpu.size() * chunk_size);

    size_t offset = 0;
    for (size_t i = 0; i < final_decomp_cpu.size(); i++) {
      size_t csize = final_decomp_cpu.sizes()[i];
      std::memcpy(&result.compData[offset],
                  final_decomp_cpu.ptrs()[i],
                  csize);
      offset += csize;
    }
    result.compData.resize(offset);
  }

  return result;
}

//==============================
// Dataset loading
//==============================
std::pair<std::vector<float>, size_t> loadTSVDataset(const std::string& filePath)
{
  std::vector<float> floatArray;
  std::ifstream file(filePath);
  std::string line;
  size_t rowCount = 0;
  if (file.is_open()) {
    while (std::getline(file, line)) {
      std::stringstream ss(line);
      std::string value;
      // skip first column
      std::getline(ss, value, '\t');
      while (std::getline(ss, value, '\t')) {
        floatArray.push_back(std::stof(value));
      }
      ++rowCount;
    }
    file.close();
  } else {
    throw std::runtime_error("Unable to open file: " + filePath);
  }
  return { floatArray, rowCount };
}

std::pair<std::vector<double>, size_t> loadTSVDatasetdouble(const std::string& filePath)
{
  std::vector<double> doubleArray;
  std::ifstream file(filePath);
  std::string line;
  size_t rowCount = 0;
  if (file.is_open()) {
    while (std::getline(file, line)) {
      std::stringstream ss(line);
      std::string value;
      // skip first column
      std::getline(ss, value, '\t');
      while (std::getline(ss, value, '\t')) {
        doubleArray.push_back(std::stod(value));
      }
      ++rowCount;
    }
    file.close();
  } else {
    throw std::runtime_error("Unable to open file: " + filePath);
  }
  return { doubleArray, rowCount };
}

std::vector<uint8_t> convertFloatToBytes(const std::vector<float>& floatArray)
{
  std::vector<uint8_t> byteArray(floatArray.size() * sizeof(float));
  std::memcpy(byteArray.data(), floatArray.data(), byteArray.size());
  return byteArray;
}

std::vector<uint8_t> convertDoubleToBytes(const std::vector<double>& doubleArray)
{
  std::vector<uint8_t> byteArray(doubleArray.size() * sizeof(double));
  std::memcpy(byteArray.data(), doubleArray.data(), byteArray.size());
  return byteArray;
}

std::string configToString(const std::vector<std::vector<size_t>>& config)
{
  std::stringstream ss;
  ss << "\"";
  for (size_t i = 0; i < config.size(); i++) {
    ss << "[";
    for (size_t j = 0; j < config[i].size(); j++) {
      ss << config[i][j];
      if (j < config[i].size() - 1) {
        ss << ",";
      }
    }
    ss << "]";
    if (i < config.size() - 1) {
      ss << ",";
    }
  }
  ss << "\"";
  return ss.str();
}

//==============================
// Combined Main Driver
//==============================
int main(int argc, char* argv[])
{
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <datasetPath> <precisionBits (32|64)>\n";
    return EXIT_FAILURE;
  }

  std::string datasetPath = argv[1];
  int precisionBits       = std::stoi(argv[2]);

  std::vector<uint8_t> globalByteArray;
  size_t rowCount = 0;
  if (precisionBits == 64) {
    auto tmp = loadTSVDatasetdouble(datasetPath);
    if (tmp.first.empty()) {
      std::cerr << "Failed to load dataset from " << datasetPath << "\n";
      return EXIT_FAILURE;
    }
    globalByteArray = convertDoubleToBytes(tmp.first);
    rowCount        = tmp.second;
    std::cout << "Loaded " << rowCount << " rows (64-bit) with "
              << tmp.first.size() << " total values.\n";

  } else if (precisionBits == 32) {
    auto tmp = loadTSVDataset(datasetPath);
    if (tmp.first.empty()) {
      std::cerr << "Failed to load dataset from " << datasetPath << "\n";
      return EXIT_FAILURE;
    }
    globalByteArray = convertFloatToBytes(tmp.first);
    rowCount        = tmp.second;
    std::cout << "Loaded " << rowCount << " rows (32-bit) with "
              << tmp.first.size() << " total values.\n";

  } else {
    std::cerr << "Unsupported precision: " << precisionBits
              << ". Use 32 or 64.\n";
    return EXIT_FAILURE;
  }

  size_t totalBytes = globalByteArray.size();
  std::cout << "Total dataset bytes: " << totalBytes << "\n";

  // Extract dataset name
  auto pos = datasetPath.find_last_of("/\\");
  std::string datasetName =
      (pos == std::string::npos) ? datasetPath : datasetPath.substr(pos + 1);
  pos = datasetName.find_last_of('.');
  if (pos != std::string::npos) {
    datasetName = datasetName.substr(0, pos);
  }
  std::cout << "Dataset name: " << datasetName << "\n";

  // Adjust the CSV path to a valid location you can write
  std::string csvFilename = datasetName + "_deflate.csv";
  std::ofstream csvFile(csvFilename);
  if (!csvFile.is_open()) {
    std::cerr << "Failed to open CSV file: " << csvFilename << "\n";
    return EXIT_FAILURE;
  }
  csvFile << "Dataset,Config,Level,ComponentIndex,UncompressedBytes,CompressedBytes,"
             "CompressionRatio,CompTimeMs,DecompTimeMs,CompThroughputGBs,DecompThroughputGBs\n";

  // (I) Whole-Dataset Mode
  std::vector<char> dataAsChar(globalByteArray.begin(), globalByteArray.end());
  std::vector<std::vector<char>> files = { dataAsChar };

  WholeDatasetMetrics wholeMetrics = run_whole_dataset_deflate(files);
  csvFile << datasetName << ","
          << "N/A,WholeDataset,"
          << ","
          << wholeMetrics.totalBytes << ","
          << wholeMetrics.compBytes << ","
          << std::fixed << std::setprecision(2)
          << wholeMetrics.compressionRatio << ","
          << wholeMetrics.compTimeMs << ","
          << wholeMetrics.decompTimeMs << ","
          << wholeMetrics.compThroughputGBs << ","
          << wholeMetrics.decompThroughputGBs << "\n";

  // (II) Component-Based Mode
  // For demonstration, we define a map of component configurations
  std::map<std::string,
           std::vector<std::vector<std::vector<size_t>>>> datasetComponentMap = {
      {"acs_wht_f32", {
                         {{1,2}, {3},   {4}},
                         {{1,2,3}, {4}},
                         {{1,2,3,4}}
                      }},
      {"default", {
                    {{1}, {2}, {3}, {4}}
                  }}
  };

  std::vector<std::vector<std::vector<size_t>>> configOptions;
  if (datasetComponentMap.find(datasetName) != datasetComponentMap.end()) {
    configOptions = datasetComponentMap[datasetName];
  } else {
    configOptions = datasetComponentMap["default"];
  }

  int numThreads = 10;
  for (size_t cfgIdx = 0; cfgIdx < configOptions.size(); cfgIdx++) {
    const auto& chosenConfig = configOptions[cfgIdx];
    std::string configStr = configToString(chosenConfig);
    std::cout << "Processing configuration " << cfgIdx
              << ": " << configStr << "\n";

    // Decompose
    std::vector<std::vector<uint8_t>> decomposedComponents;
    splitBytesIntoComponentsNested(globalByteArray,
                                   decomposedComponents,
                                   chosenConfig, numThreads);
    std::cout << "Dataset decomposed into "
              << decomposedComponents.size() << " components.\n";

    size_t totalCompCompressedSize = 0;
    float overallCompTimeMs        = 0.0f;
    float overallDecompTimeMs      = 0.0f;

    // We store the re-inflated components for final reassembly
    std::vector<std::vector<uint8_t>> decomposedComponentData;
    decomposedComponentData.reserve(decomposedComponents.size());

    for (size_t compIdx = 0; compIdx < decomposedComponents.size(); compIdx++) {
      std::vector<char> compData(decomposedComponents[compIdx].begin(),
                                 decomposedComponents[compIdx].end());
      size_t compUncompressed = compData.size();

      ComponentResult result =
          compress_decompress_component_deflate(compData);

      totalCompCompressedSize += result.compSize;
      overallCompTimeMs       += result.compTimeMs;
      overallDecompTimeMs     += result.decompTimeMs;

      std::vector<uint8_t> compDecomp(
          result.compData.begin(),
          result.compData.end());
      decomposedComponentData.push_back(std::move(compDecomp));

      double compThroughput =
          (result.compTimeMs > 0.0f)
              ? ((double)compUncompressed / result.compTimeMs) * 1e-6
              : 0.0;
      double decompThroughput =
          (result.decompTimeMs > 0.0f)
              ? ((double)compUncompressed / result.decompTimeMs) * 1e-6
              : 0.0;
      double compRatio =
          (result.compSize > 0)
              ? (double)compUncompressed / (double)result.compSize
              : 0.0;

      csvFile << datasetName << ","
              << configStr << ","
              << "Component," << compIdx << ","
              << compUncompressed << ","
              << result.compSize << ","
              << std::fixed << std::setprecision(2) << compRatio << ","
              << result.compTimeMs << ","
              << result.decompTimeMs << ","
              << compThroughput << ","
              << decompThroughput << "\n";
    }

    double overallCompressionRatio =
        (totalCompCompressedSize > 0)
            ? ((double)totalBytes / totalCompCompressedSize)
            : 0.0;
    double overallCompThroughput =
        (overallCompTimeMs > 0.0f)
            ? ((double)totalBytes / overallCompTimeMs) * 1e-6
            : 0.0;
    double overallDecompThroughput =
        (overallDecompTimeMs > 0.0f)
            ? ((double)totalBytes / overallDecompTimeMs) * 1e-6
            : 0.0;

    csvFile << datasetName << ","
            << configStr << ","
            << "Overall,"
            << ","
            << totalBytes << ","
            << totalCompCompressedSize << ","
            << std::fixed << std::setprecision(2)
            << overallCompressionRatio << ","
            << overallCompTimeMs << ","
            << overallDecompTimeMs << ","
            << overallCompThroughput << ","
            << overallDecompThroughput << "\n";

    // Reassemble to validate
    std::vector<uint8_t> reassembled(globalByteArray.size());
    reassembleBytesFromComponentsNestedlz4(
        decomposedComponentData, reassembled.data(),
        reassembled.size(), chosenConfig, numThreads);

    if (reassembled == globalByteArray) {
      std::cout << "Reassembled data matches the original dataset.\n";
    } else {
      std::cerr << "Error: Reassembled data does NOT match the original dataset!\n";
    }
  }

  csvFile.close();
  std::cout << "Results saved to: " << csvFilename << "\n";
  return EXIT_SUCCESS;
}
