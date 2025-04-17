/****************************************************************************
 * Example of (1) reading dataset, (2) decomposing it into sub-buffers,
 * (3) GPU compressing each sub-buffer with nvcomp deflate,
 * (4) CPU decompressing with libdeflate,
 * (5) reassembling the final data, and validating correctness.
 ****************************************************************************/

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstdint>
#include <stdexcept>
#include <algorithm>    // std::min, etc.
#include <iomanip>      // std::setprecision
#include <cuda_runtime.h>

// nvCOMP
#include <nvcomp/deflate.h>
#include <nvcomp.h>

// libdeflate (for CPU decompression)
#include <libdeflate.h>

// If you have your own "BatchData" type for chunking, you can #include "BatchData.h" here.
// For demonstration, we'll define a minimal chunk-based structure below.

#define CUDA_CHECK(stmt)                                     \
  do {                                                       \
    cudaError_t err = stmt;                                  \
    if (err != cudaSuccess) {                                \
      std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                << " at " << __FILE__ << ":" << __LINE__     \
                << std::endl;                                \
      throw std::runtime_error("CUDA_CHECK failed.");         \
    }                                                        \
  } while (0)

/******************************************************************************
 * Minimal chunk-based structure (like your BatchData) for the GPU compression
 ******************************************************************************/
class ChunkedData
{
public:
  ChunkedData(const std::vector<char>& hostData, size_t chunkSize)
  {
    // Break hostData into lumps of chunkSize
    size_t offset = 0;
    while (offset < hostData.size()) {
      size_t bytes = std::min((size_t)chunkSize, hostData.size() - offset);
      chunkPointersHost.push_back((void*)(hostData.data() + offset));
      chunkSizesHost.push_back(bytes);
      offset += bytes;
    }
  }

  // Return arrays for nvcompBatchedDeflateCompressAsync
  void** ptrsHost() { return chunkPointersHost.data(); }
  size_t* sizesHost() { return chunkSizesHost.data(); }
  size_t numChunks() const { return chunkPointersHost.size(); }

private:
  std::vector<void*>  chunkPointersHost;
  std::vector<size_t> chunkSizesHost;
};

/******************************************************************************
 * Functions to do the data decomposition & reassembly
 * (Simplified placeholders)
 ******************************************************************************/

// Example: suppose each "element" is 4 bytes, and you want to group them
// with a certain config.  For demonstration, we'll treat the entire dataset
// as a sequence of "elements."  We'll keep your function signature form:
void splitBytesIntoComponentsNested(
    const std::vector<uint8_t>& byteArray,
    std::vector<std::vector<uint8_t>>& outputComponents,
    const std::vector<std::vector<size_t>>& allComponentSizes,
    int numThreads = 1)
{
  // E.g., each "element" is sum(allComponentSizes[i]) bytes.  We'll do
  // a trivial approach. In real code, you'd do the "1-based index" approach
  // that you described, but we keep it simpler here for demonstration.
  // We'll assume allComponentSizes might say e.g. { {1,2}, {3}, {4} }.

  size_t totalBytesPerElement = 0;
  for (auto &group : allComponentSizes) {
    totalBytesPerElement += group.size();
  }

  // Number of "elements"
  size_t numElements = byteArray.size() / totalBytesPerElement;
  outputComponents.resize(allComponentSizes.size());

  // Allocate output
  for (size_t i = 0; i < allComponentSizes.size(); i++) {
    size_t groupSize = allComponentSizes[i].size();
    outputComponents[i].resize(numElements * groupSize);
  }

  // We'll do a naive single-thread example.  If you want OMP, fine.
  for (size_t elem = 0; elem < numElements; elem++) {
    for (size_t compIdx = 0; compIdx < allComponentSizes.size(); compIdx++) {
      const auto &groupIndices = allComponentSizes[compIdx];
      size_t groupSize = groupIndices.size();
      size_t writePos = elem * groupSize;
      for (size_t sub = 0; sub < groupSize; sub++) {
        // 1-based => sub -1, etc.  We'll do a dummy approach:
        size_t idxInElem = groupIndices[sub] - 1; // (1-based -> 0-based)
        size_t globalSrcIdx = elem * totalBytesPerElement + idxInElem;
        outputComponents[compIdx][writePos + sub] = byteArray[globalSrcIdx];
      }
    }
  }
}

// Reassembly
void reassembleBytesFromComponentsNested(
    const std::vector<std::vector<uint8_t>>& inputComponents,
    uint8_t* byteArray,
    size_t byteArraySize,
    const std::vector<std::vector<size_t>>& allComponentSizes,
    int numThreads = 1)
{
  size_t totalBytesPerElement = 0;
  for (auto &group : allComponentSizes) {
    totalBytesPerElement += group.size();
  }
  size_t numElements = byteArraySize / totalBytesPerElement;

  for (size_t compIdx = 0; compIdx < inputComponents.size(); compIdx++) {
    const auto &groupIndices = allComponentSizes[compIdx];
    size_t groupSize = groupIndices.size();
    const auto &componentData = inputComponents[compIdx];
    for (size_t elem = 0; elem < numElements; elem++) {
      size_t readPos = elem * groupSize;
      for (size_t sub = 0; sub < groupSize; sub++) {
        size_t idxInElem = groupIndices[sub] - 1;  // again converting 1-based
        size_t globalIndex = elem * totalBytesPerElement + idxInElem;
        byteArray[globalIndex] = componentData[readPos + sub];
      }
    }
  }
}

/******************************************************************************
 * GPU-based compression with nvcomp. We'll define a function that compresses
 * a single sub-buffer (component).
 ******************************************************************************/
std::vector<char> compressWithNvcompDeflate(const std::vector<char>& inData)
{
  // We'll do a "BatchData" style approach for the entire inData as single
  // "file" chunked to 64KB lumps.
  size_t chunkSize = 64 * 1024;
  ChunkedData chunked(inData, chunkSize);
  size_t numChunks = chunked.numChunks();
  std::cout << "component has " << inData.size() << " uncompressed bytes in " << numChunks << " chunks.\n";

  // 1) get temp and output chunk sizes
  size_t temp_bytes;
  nvcompStatus_t status = nvcompBatchedDeflateCompressGetTempSize(numChunks, chunkSize, &temp_bytes);
  if (status != nvcompSuccess) {
    throw std::runtime_error("Error: getTempSize for deflate compress");
  }

  size_t max_out_bytes;
  status = nvcompBatchedDeflateCompressGetMaxOutputChunkSize(chunkSize, &max_out_bytes);
  if (status != nvcompSuccess) {
    throw std::runtime_error("Error: getMaxOutputChunkSize");
  }

  // 2) allocate device memory
  void* d_in_data;
  CUDA_CHECK(cudaMalloc(&d_in_data, inData.size()));
  CUDA_CHECK(cudaMemcpy(d_in_data, inData.data(), inData.size(), cudaMemcpyHostToDevice));

  // build arrays for chunk pointers in device memory
  std::vector<void*>  chunk_ptrs_host(numChunks);
  std::vector<size_t> chunk_sizes_host(numChunks);
  {
    // Re-chunk the data in device form
    size_t offset = 0;
    for (size_t i = 0; i < numChunks; i++) {
      size_t size_i = chunked.sizesHost()[i];
      chunk_ptrs_host[i] = (char*)d_in_data + offset;
      chunk_sizes_host[i] = size_i;
      offset += size_i;
    }
  }
  // copy these arrays to device
  void** d_in_ptrs;
  CUDA_CHECK(cudaMalloc(&d_in_ptrs, numChunks * sizeof(void*)));
  CUDA_CHECK(cudaMemcpy(d_in_ptrs, chunk_ptrs_host.data(), numChunks * sizeof(void*), cudaMemcpyHostToDevice));

  size_t* d_in_sizes;
  CUDA_CHECK(cudaMalloc(&d_in_sizes, numChunks * sizeof(size_t)));
  CUDA_CHECK(cudaMemcpy(d_in_sizes, chunk_sizes_host.data(), numChunks * sizeof(size_t), cudaMemcpyHostToDevice));

  // allocate output
  void* d_temp;
  CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));

  void** d_out_ptrs;
  CUDA_CHECK(cudaMalloc(&d_out_ptrs, numChunks * sizeof(void*)));
  // each chunk can be up to max_out_bytes
  size_t total_out_mem = numChunks * max_out_bytes;
  void* d_out_data;
  CUDA_CHECK(cudaMalloc(&d_out_data, total_out_mem));

  // set up the chunk out ptr array
  std::vector<void*> out_ptrs_host(numChunks);
  size_t offset_out = 0;
  for (size_t i = 0; i < numChunks; i++) {
    out_ptrs_host[i] = (char*)d_out_data + offset_out;
    offset_out += max_out_bytes; // each chunk up to max_out_bytes
  }
  CUDA_CHECK(cudaMemcpy(d_out_ptrs, out_ptrs_host.data(), numChunks*sizeof(void*), cudaMemcpyHostToDevice));

  size_t* d_out_sizes;
  CUDA_CHECK(cudaMalloc(&d_out_sizes, numChunks * sizeof(size_t)));

  // Now compress:
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  status = nvcompBatchedDeflateCompressAsync(
      d_in_ptrs, d_in_sizes, chunkSize,
      numChunks,
      d_temp, temp_bytes,
      d_out_ptrs, d_out_sizes,
      stream
  );
  if (status != nvcompSuccess) {
    throw std::runtime_error("Error: nvcompBatchedDeflateCompressAsync");
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // get the actual compressed sizes
  std::vector<size_t> out_sizes_host(numChunks);
  CUDA_CHECK(cudaMemcpy(out_sizes_host.data(), d_out_sizes, numChunks*sizeof(size_t), cudaMemcpyDeviceToHost));

  // sum them
  size_t total_comp_bytes = 0;
  for (auto sz : out_sizes_host) {
    total_comp_bytes += sz;
  }

  // copy the compressed data back to host
  std::vector<char> compDataHost(total_comp_bytes);
  {
    size_t out_offset = 0;
    size_t base = 0;
    for (size_t i = 0; i < numChunks; i++) {
      size_t sz_i = out_sizes_host[i];
      CUDA_CHECK(cudaMemcpy(compDataHost.data() + out_offset,
                            (char*)d_out_ptrs[i],
                            sz_i,
                            cudaMemcpyDeviceToHost));
      out_offset += sz_i;
    }
  }
  std::cout << "Compressed from " << inData.size()
            << " to " << total_comp_bytes
            << " bytes. ratio: "
            << std::fixed << std::setprecision(2)
            << (double)inData.size() / total_comp_bytes << std::endl;

  // cleanup
  CUDA_CHECK(cudaFree(d_in_data));
  CUDA_CHECK(cudaFree(d_in_ptrs));
  CUDA_CHECK(cudaFree(d_in_sizes));
  CUDA_CHECK(cudaFree(d_temp));
  CUDA_CHECK(cudaFree(d_out_data));
  CUDA_CHECK(cudaFree(d_out_ptrs));
  CUDA_CHECK(cudaFree(d_out_sizes));
  CUDA_CHECK(cudaStreamDestroy(stream));

  return compDataHost;
}

/******************************************************************************
 * CPU Decompression with libdeflate (each sub-buffer)
 ******************************************************************************/
std::vector<uint8_t> decompressLibdeflate(const std::vector<char>& compData, size_t uncompressedSize)
{
  std::vector<uint8_t> outBuf(uncompressedSize);

  // create a libdeflate decompressor
  struct libdeflate_decompressor* decompressor = libdeflate_alloc_decompressor();
  if (!decompressor) {
    throw std::runtime_error("libdeflate_alloc_decompressor failed");
  }

  size_t actual_out = 0;
  libdeflate_result res = libdeflate_deflate_decompress(
      decompressor,
      compData.data(),
      compData.size(),
      outBuf.data(),
      outBuf.size(),
      &actual_out
  );
  libdeflate_free_decompressor(decompressor);
  if (res != LIBDEFLATE_SUCCESS) {
    throw std::runtime_error("Error: libdeflate_deflate_decompress returned " + std::to_string(res));
  }
  if (actual_out != uncompressedSize) {
    std::cerr << "Warning: uncompressed size mismatch. expected " << uncompressedSize << " got " << actual_out << std::endl;
  }
  return outBuf;
}


/******************************************************************************
 * MAIN
 *  1) read dataset into globalByteArray
 *  2) define a decomposition config
 *  3) decompose
 *  4) GPU compress each sub-buffer
 *  5) CPU decompress each sub-buffer
 *  6) reassemble
 ******************************************************************************/
int main(int argc, char** argv)
{
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <input.bin>\n";
    return 1;
  }
  std::string inputFile = argv[1];

  // read entire file into memory as bytes
  std::ifstream fin(inputFile, std::ios::binary);
  if (!fin.is_open()) {
    std::cerr << "Cannot open " << inputFile << std::endl;
    return 1;
  }
  std::vector<uint8_t> globalByteArray;
  {
    fin.seekg(0, std::ios::end);
    size_t fileSize = fin.tellg();
    fin.seekg(0, std::ios::beg);
    globalByteArray.resize(fileSize);
    fin.read((char*)globalByteArray.data(), fileSize);
  }
  fin.close();

  size_t totalBytes = globalByteArray.size();
  std::cout << "Loaded " << totalBytes << " bytes from " << inputFile << std::endl;

  // Suppose we define a decomposition config like: { {1,2}, {3}, {4} }
  // meaning in each "element," the first sub-buffer is bytes #1,2, the second #3, the third #4.
  // We'll assume each element is 4 bytes. We'll also assume that globalByteArray length
  // is multiple of 4. This is purely illustrative.

  std::vector<std::vector<size_t>> chosenConfig = {
    {1,2}, {3}, {4}
  };
  // If you want a single config that lumps all bytes into one sub-buffer, you'd do { {1,2,3,4} }.

  // 1) Decompose
  std::vector<std::vector<uint8_t>> decomposedComponents;
  splitBytesIntoComponentsNested(globalByteArray, decomposedComponents, chosenConfig, /*threads=*/1);

  // 2) For each sub-buffer, GPU compress
  std::vector<std::vector<char>> compressedBuffers(decomposedComponents.size());
  for (size_t c = 0; c < decomposedComponents.size(); c++) {
    std::vector<char> inChar(decomposedComponents[c].begin(), decomposedComponents[c].end());
    compressedBuffers[c] = compressWithNvcompDeflate(inChar);
  }

  // 3) CPU decompress each sub-buffer with libdeflate
  std::vector<std::vector<uint8_t>> decompressedComponents(decomposedComponents.size());
  for (size_t c = 0; c < decomposedComponents.size(); c++) {
    size_t subSize = decomposedComponents[c].size();
    decompressedComponents[c] = decompressLibdeflate(compressedBuffers[c], subSize);
    // Compare or trust it.  If mismatch => error
    if (decompressedComponents[c] != decomposedComponents[c]) {
      std::cerr << "Error: mismatch on component " << c << std::endl;
      return 1;
    }
  }

  // 4) Reassemble
  std::vector<uint8_t> reassembled(totalBytes);
  reassembleBytesFromComponentsNested(decompressedComponents, reassembled.data(), reassembled.size(), chosenConfig, 1);

  // 5) Check final
  if (reassembled == globalByteArray) {
    std::cout << "Success: Reassembled data matches the original :)\n";
  } else {
    std::cerr << "ERROR: mismatch after final reassembly\n";
  }

  return 0;
}
