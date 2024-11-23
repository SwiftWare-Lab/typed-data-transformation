#ifndef LZ4_PARALLEL_H
#define LZ4_PARALLEL_H

#include <vector>
#include <iostream>
#include <cstring>
#include <lz4.h>
#include <lz4hc.h>
#include "profiling_info.h"

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

// Compress with LZ4
size_t compressWithLZ4(const std::vector<uint8_t>& data, std::vector<uint8_t>& compressedData, int compressionLevel) {
    int maxCompressedSize = LZ4_compressBound(data.size());
    compressedData.resize(maxCompressedSize);

    int compressedSize;
    if (compressionLevel > 0) {
        // Use high-compression mode
        compressedSize = LZ4_compress_HC(reinterpret_cast<const char*>(data.data()),
                                         reinterpret_cast<char*>(compressedData.data()),
                                         data.size(), maxCompressedSize, compressionLevel);
    } else {
        // Use fast compression mode
        compressedSize = LZ4_compress_default(reinterpret_cast<const char*>(data.data()),
                                              reinterpret_cast<char*>(compressedData.data()),
                                              data.size(), maxCompressedSize);
    }

    if (compressedSize <= 0) {
        std::cerr << "LZ4 compression error." << std::endl;
        return 0;
    }

    compressedData.resize(compressedSize);
    return compressedSize;
}

// Decompress with LZ4
size_t decompressWithLZ4(const std::vector<uint8_t>& compressedData, std::vector<uint8_t>& decompressedData, size_t originalSize) {
    decompressedData.resize(originalSize);
    int decompressedSize = LZ4_decompress_safe(reinterpret_cast<const char*>(compressedData.data()),
                                               reinterpret_cast<char*>(decompressedData.data()),
                                               compressedData.size(), originalSize);
    if (decompressedSize < 0) {
        std::cerr << "LZ4 decompression error." << std::endl;
        return 0;
    }

    return decompressedSize;
}

// Full compression without decomposition
size_t lz4Compression(const std::vector<uint8_t>& data, ProfilingInfo &pi, std::vector<uint8_t>& compressedData) {
    auto start = std::chrono::high_resolution_clock::now();
    size_t compressedSize = compressWithLZ4(data, compressedData, 0); // Fast compression
    auto end = std::chrono::high_resolution_clock::now();

    pi.type = "LZ4 Full Compression";

    return compressedSize;
}

// Full decompression without decomposition
void lz4Decompression(const std::vector<uint8_t>& compressedData, std::vector<uint8_t>& decompressedData, ProfilingInfo &pi, size_t originalSize) {
    auto start = std::chrono::high_resolution_clock::now();
    decompressWithLZ4(compressedData, decompressedData, originalSize);
    auto end = std::chrono::high_resolution_clock::now();
  // Verify decompressed data
  if (!verifyDataMatch(globalByteArray, decompressedData)) {
    std::cerr << "Error: Decompressed data doesn't match the original." << std::endl;
  }

}
// Sequential compression with decomposition that takes dynamic byte sizes as parameters
size_t lz4DecomposedSequential(const std::vector<uint8_t>& data, ProfilingInfo &pi,
                                std::vector<uint8_t>& compressedLeading,
                                std::vector<uint8_t>& compressedContent,
                                std::vector<uint8_t>& compressedTrailing,
                                size_t leadingBytes, size_t contentBytes, size_t trailingBytes) {
    std::vector<uint8_t> leading, content, trailing;
    splitBytesIntoComponents(data, leading, content, trailing, leadingBytes, contentBytes, trailingBytes);

    auto start = std::chrono::high_resolution_clock::now();
    size_t compressedSize_l = compressWithLZ4(leading, compressedLeading, 3);
    auto end = std::chrono::high_resolution_clock::now();
    pi.leading_time = std::chrono::duration<double>(end - start).count();

    start = std::chrono::high_resolution_clock::now();
    size_t compressedSize_c = compressWithLZ4(content, compressedContent, 3);
    end = std::chrono::high_resolution_clock::now();
    pi.content_time = std::chrono::duration<double>(end - start).count();

    start = std::chrono::high_resolution_clock::now();
    size_t compressedSize_t = compressWithLZ4(trailing, compressedTrailing, 3);
    end = std::chrono::high_resolution_clock::now();
    pi.trailing_time = std::chrono::duration<double>(end - start).count();

    pi.type = "Sequential Decomposition";
    return compressedSize_l + compressedSize_c + compressedSize_t;
}

// Sequential decompression with decomposition that  takes dynamic byte sizes as parameters
void lz4DecomposedSequentialDecompression(const std::vector<uint8_t>& compressedLeading,
                                           const std::vector<uint8_t>& compressedContent,
                                           const std::vector<uint8_t>& compressedTrailing,
                                           ProfilingInfo &pi,
                                           size_t leadingBytes, size_t contentBytes, size_t trailingBytes) {
    size_t totalSize = globalByteArray.size();  // Ensure globalByteArray is appropriately defined and accessible
    size_t floatCount = totalSize / (leadingBytes + contentBytes + trailingBytes);

    std::vector<uint8_t> reconstructedData(totalSize);

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<uint8_t> tempLeading(floatCount * leadingBytes);
    decompressWithLZ4(compressedLeading, tempLeading, floatCount * leadingBytes);
    for (size_t i = 0; i < floatCount; ++i) {
        for (size_t j = 0; j < leadingBytes; ++j) {
            reconstructedData[i * (leadingBytes + contentBytes + trailingBytes) + j] = tempLeading[i * leadingBytes + j];
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    pi.leading_time = std::chrono::duration<double>(end - start).count();

    start = std::chrono::high_resolution_clock::now();
    std::vector<uint8_t> tempContent(floatCount * contentBytes);
    decompressWithLZ4(compressedContent, tempContent, floatCount * contentBytes);
    for (size_t i = 0; i < floatCount; ++i) {
        for (size_t j = 0; j < contentBytes; ++j) {
            reconstructedData[i * (leadingBytes + contentBytes + trailingBytes) + leadingBytes + j] = tempContent[i * contentBytes + j];
        }
    }
    end = std::chrono::high_resolution_clock::now();
    pi.content_time = std::chrono::duration<double>(end - start).count();

    start = std::chrono::high_resolution_clock::now();
    std::vector<uint8_t> tempTrailing(floatCount * trailingBytes);
    decompressWithLZ4(compressedTrailing, tempTrailing, floatCount * trailingBytes);
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
size_t lz4DecomposedParallel(const std::vector<uint8_t>& data, ProfilingInfo &pi,
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
            compressedSize_l = compressWithLZ4(leading, compressedLeading, 3);
            auto end = std::chrono::high_resolution_clock::now();
            pi.leading_time = std::chrono::duration<double>(end - start).count();
        }

        #pragma omp section
        {
            auto start = std::chrono::high_resolution_clock::now();
            compressedSize_c = compressWithLZ4(content, compressedContent, 3);
            auto end = std::chrono::high_resolution_clock::now();
            pi.content_time = std::chrono::duration<double>(end - start).count();
        }

        #pragma omp section
        {
            auto start = std::chrono::high_resolution_clock::now();
            compressedSize_t = compressWithLZ4(trailing, compressedTrailing, 3);
            auto end = std::chrono::high_resolution_clock::now();
            pi.trailing_time = std::chrono::duration<double>(end - start).count();
        }
    }

    pi.type = "Parallel Decomposition";
    return (compressedLeading.size() + compressedContent.size() + compressedTrailing.size());
}



// Decompression function with dynamic byte segmentation and OpenMP parallelization
void zstdDecomposedParallelDecompression(const std::vector<uint8_t>& compressedLeading,
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
            decompressWithLZ4(compressedLeading, tempLeading, floatCount * leadingBytes);
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
            decompressWithLZ4(compressedContent, tempContent, floatCount * contentBytes);
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
            decompressWithLZ4(compressedTrailing, tempTrailing, floatCount * trailingBytes);
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

#endif // LZ4_PARALLEL_H
