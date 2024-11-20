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
#include <cmath>




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
// Compress with Zstd using 5 threads
size_t compressWithZstd1(const std::vector<uint8_t>& data, std::vector<uint8_t>& compressedData, int compressionLevel) {
  size_t const cBuffSize = ZSTD_compressBound(data.size());
  compressedData.resize(cBuffSize);

  // Create a compression context
  ZSTD_CCtx* cctx = ZSTD_createCCtx();
  if (cctx == NULL) {
    std::cerr << "Failed to create ZSTD_CCtx" << std::endl;
    return 0;
  }

  // Set the number of threads to 5
  ZSTD_CCtx_setParameter(cctx, ZSTD_c_nbWorkers, 0);

  // Compress using the compression context
  size_t const cSize = ZSTD_compressCCtx(cctx, compressedData.data(), cBuffSize, data.data(), data.size(), compressionLevel);
  if (ZSTD_isError(cSize)) {
    std::cerr << "Zstd compression error: " << ZSTD_getErrorName(cSize) << std::endl;
    ZSTD_freeCCtx(cctx);
    return 0;
  }

  // Resize the compressed data to the actual compressed size
  compressedData.resize(cSize);

  // Free the compression context
  ZSTD_freeCCtx(cctx);
  return cSize;
}
size_t compressWithZstd2(const std::vector<uint8_t>& data, std::vector<uint8_t>& compressedData, int compressionLevel) {
  unsigned numThreads = 10;  // Number of threads
  int additionalInt = 2;    // This is hypothetical as it's unclear what this integer represents without your specific Zstd documentation.

  size_t const cBuffSize = ZSTD_compressBound(data.size());
  compressedData.resize(cBuffSize);

  // Create a compression context
  ZSTD_CCtx* cctx = ZSTD_createCCtx();
  if (cctx == NULL) {
    std::cerr << "Failed to create ZSTD_CCtx" << std::endl;
    return 0;
  }

  // Set the compression level
  if (ZSTD_isError(ZSTD_CCtx_setParameter(cctx, ZSTD_c_compressionLevel, compressionLevel))) {
    std::cerr << "Failed to set compression level" << std::endl;
    ZSTD_freeCCtx(cctx);
    return 0;
  }

  // Set the number of threads
  if (ZSTD_isError(ZSTD_CCtx_setParameter(cctx, ZSTD_c_nbWorkers, numThreads))) {
    std::cerr << "Failed to set number of threads" << std::endl;
    ZSTD_freeCCtx(cctx);
    return 0;
  }

  // Compress with an additional integer argument
  size_t const cSize = ZSTD_compressCCtx(cctx, compressedData.data(), cBuffSize, data.data(), data.size(), additionalInt);
  if (ZSTD_isError(cSize)) {
    std::cerr << "Zstd compression error: " << ZSTD_getErrorName(cSize) << std::endl;
    ZSTD_freeCCtx(cctx);
    return 0;
  }

  // Resize the compressed data to the actual compressed size
  compressedData.resize(cSize);

  // Free the compression context
  ZSTD_freeCCtx(cctx);
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
// Sequential compression with decomposition that takes dynamic byte sizes as parameters
size_t zstdDecomposedSequential(const std::vector<uint8_t>& data, ProfilingInfo &pi,
                                std::vector<uint8_t>& compressedLeading,
                                std::vector<uint8_t>& compressedContent,
                                std::vector<uint8_t>& compressedTrailing,
                                size_t leadingBytes, size_t contentBytes, size_t trailingBytes) {
    std::vector<uint8_t> leading, content, trailing;
    splitBytesIntoComponents(data, leading, content, trailing, leadingBytes, contentBytes, trailingBytes);

    auto start = std::chrono::high_resolution_clock::now();
    size_t compressedSize_l = compressWithZstd(leading, compressedLeading, 3);
    auto end = std::chrono::high_resolution_clock::now();
    pi.leading_time = std::chrono::duration<double>(end - start).count();

    start = std::chrono::high_resolution_clock::now();
    size_t compressedSize_c = compressWithZstd(content, compressedContent, 3);
    end = std::chrono::high_resolution_clock::now();
    pi.content_time = std::chrono::duration<double>(end - start).count();

    start = std::chrono::high_resolution_clock::now();
    size_t compressedSize_t = compressWithZstd(trailing, compressedTrailing, 3);
    end = std::chrono::high_resolution_clock::now();
    pi.trailing_time = std::chrono::duration<double>(end - start).count();

    pi.type = "Sequential Decomposition";
    return compressedSize_l + compressedSize_c + compressedSize_t;
}

// Sequential decompression with decomposition that  takes dynamic byte sizes as parameters
void zstdDecomposedSequentialDecompression(const std::vector<uint8_t>& compressedLeading,
                                           const std::vector<uint8_t>& compressedContent,
                                           const std::vector<uint8_t>& compressedTrailing,
                                           ProfilingInfo &pi,
                                           size_t leadingBytes, size_t contentBytes, size_t trailingBytes) {
    size_t totalSize = globalByteArray.size();  // Ensure globalByteArray is appropriately defined and accessible
    size_t floatCount = totalSize / (leadingBytes + contentBytes + trailingBytes);

    std::vector<uint8_t> reconstructedData(totalSize);

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<uint8_t> tempLeading(floatCount * leadingBytes);
    decompressWithZstd(compressedLeading, tempLeading, floatCount * leadingBytes);
    for (size_t i = 0; i < floatCount; ++i) {
        for (size_t j = 0; j < leadingBytes; ++j) {
            reconstructedData[i * (leadingBytes + contentBytes + trailingBytes) + j] = tempLeading[i * leadingBytes + j];
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    pi.leading_time = std::chrono::duration<double>(end - start).count();

    start = std::chrono::high_resolution_clock::now();
    std::vector<uint8_t> tempContent(floatCount * contentBytes);
    decompressWithZstd(compressedContent, tempContent, floatCount * contentBytes);
    for (size_t i = 0; i < floatCount; ++i) {
        for (size_t j = 0; j < contentBytes; ++j) {
            reconstructedData[i * (leadingBytes + contentBytes + trailingBytes) + leadingBytes + j] = tempContent[i * contentBytes + j];
        }
    }
    end = std::chrono::high_resolution_clock::now();
    pi.content_time = std::chrono::duration<double>(end - start).count();

    start = std::chrono::high_resolution_clock::now();
    std::vector<uint8_t> tempTrailing(floatCount * trailingBytes);
    decompressWithZstd(compressedTrailing, tempTrailing, floatCount * trailingBytes);
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
std::tuple<size_t, size_t, size_t, size_t> zstdDecomposedParallel(const std::vector<uint8_t>& data, ProfilingInfo &pi,
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
            compressedSize_l = compressWithZstd(leading, compressedLeading, 3);
            auto end = std::chrono::high_resolution_clock::now();
            pi.leading_time = std::chrono::duration<double>(end - start).count();
        }

        #pragma omp section
        {
            auto start = std::chrono::high_resolution_clock::now();
            compressedSize_c = compressWithZstd(content, compressedContent, 3);
            auto end = std::chrono::high_resolution_clock::now();
            pi.content_time = std::chrono::duration<double>(end - start).count();
        }

        #pragma omp section
        {
            auto start = std::chrono::high_resolution_clock::now();
            compressedSize_t = compressWithZstd(trailing, compressedTrailing, 3);
            auto end = std::chrono::high_resolution_clock::now();
            pi.trailing_time = std::chrono::duration<double>(end - start).count();
        }
    }

    pi.type = "Parallel Decomposition";
  return {compressedLeading.size() + compressedContent.size() + compressedTrailing.size(),
         compressedLeading.size(), compressedContent.size(), compressedTrailing.size()};
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
            decompressWithZstd(compressedLeading, tempLeading, floatCount * leadingBytes);
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
            decompressWithZstd(compressedContent, tempContent, floatCount * contentBytes);
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
            decompressWithZstd(compressedTrailing, tempTrailing, floatCount * trailingBytes);
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


double calculateCompressionRatio(size_t originalSize, size_t compressedSize) {
  return static_cast<double>(originalSize) / static_cast<double>(compressedSize);
}
// Helper function to convert a byte array into a vector of binary patterns of any size
std::vector<std::string> createBinaryPatterns(const std::vector<uint8_t>& byteArray, size_t patternLength) {
  std::vector<std::string> patterns;

  // Create a string of bits from the byte array
  std::string bitStream;
  for (uint8_t byte : byteArray) {
    for (int i = 7; i >= 0; --i) {  // Extract bits MSB to LSB
      bitStream += (byte & (1 << i)) ? '1' : '0';
    }
  }

  // Split the bit stream into patterns of the given length
  for (size_t i = 0; i + patternLength <= bitStream.size(); i += patternLength) {
    patterns.push_back(bitStream.substr(i, patternLength));
  }

  return patterns;
}

// Function to calculate entropy based on binary patterns
double calculateEntropy(const std::vector<uint8_t>& byteArray, size_t patternLength) {
  // Convert byte array into binary patterns
  std::vector<std::string> patterns = createBinaryPatterns(byteArray, patternLength);

  // Calculate the frequency of each unique pattern
  std::map<std::string, size_t> frequencies;
  for (const std::string& pattern : patterns) {
    frequencies[pattern]++;
  }

  // Calculate probabilities and entropy
  double entropy = 0.0;
  size_t total_count = patterns.size();
  for (const auto& pair : frequencies) {
    double probability = static_cast<double>(pair.second) / total_count;
    entropy -= probability * std::log2(probability);
  }

  return entropy;
}
// Function to calculate correlation between binary patterns
double calculateCorrelation(const std::vector<uint8_t>& byteArray, size_t patternLength) {
  // Convert byte array into binary patterns
  std::vector<std::string> patterns = createBinaryPatterns(byteArray, patternLength);

  if (patterns.size() < 2) {
    return 0.0; // No correlation for single pattern
  }

  // Transition frequency map
  std::unordered_map<std::string, size_t> transitionCounts;
  size_t totalTransitions = 0;

  // Count transitions between patterns
  for (size_t i = 0; i < patterns.size() - 1; ++i) {
    std::string transition = patterns[i] + "->" + patterns[i + 1];
    transitionCounts[transition]++;
    totalTransitions++;
  }

  // Calculate probabilities of transitions and their entropy
  double correlation = 0.0;
  for (const auto& pair : transitionCounts) {
    double probability = static_cast<double>(pair.second) / totalTransitions;
    correlation -= probability * std::log2(probability);
  }

  return correlation;
}

// Function to calculate the frequency distribution of patterns
// std::unordered_map<std::string, double> measurePatternDistribution(const std::vector<uint8_t>& byteArray, size_t patternLength) {
//   // Generate binary patterns from the byte array
//   std::vector<std::string> patterns = createBinaryPatterns(byteArray, patternLength);
//
//   // Count occurrences of each pattern
//   std::unordered_map<std::string, size_t> frequencyMap;
//   for (const std::string& pattern : patterns) {
//     frequencyMap[pattern]++;
//   }
//
//   // Normalize frequencies to probabilities
//   std::unordered_map<std::string, double> probabilityMap;
//   size_t totalPatterns = patterns.size();
//   for (const auto& pair : frequencyMap) {
//     probabilityMap[pair.first] = static_cast<double>(pair.second) / totalPatterns;
//   }
//
//   return probabilityMap;
// }
// Function to extract a specific bit range as a pattern
std::string extractPattern(const std::vector<uint8_t>& byteArray, size_t startBit, size_t bitLength) {
  std::string pattern;
  size_t endBit = startBit + bitLength;
  for (size_t bit = startBit; bit < endBit; ++bit) {
    size_t byteIndex = bit / 8;
    size_t bitIndex = 7 - (bit % 8);  // MSB to LSB
    if (byteArray[byteIndex] & (1 << bitIndex)) {
      pattern += '1';
    } else {
      pattern += '0';
    }
  }
  return pattern;
}

// Function to calculate Shannon entropy
double calculateEntropy(const std::unordered_map<std::string, size_t>& frequency, size_t totalSymbols) {
  double entropy = 0.0;
  for (const auto& entry : frequency) {
    double prob = static_cast<double>(entry.second) / totalSymbols;
    entropy -= prob * std::log2(prob);
  }
  return entropy;
}

// Function to calculate entropy for full data with three times more symbols
double calculateExpandedEntropy(const std::vector<uint8_t>& byteArray,
                                size_t leadingBytes, size_t contentBytes, size_t trailingBytes) {
  size_t segmentSize = leadingBytes + contentBytes + trailingBytes;
  size_t numSegments = byteArray.size() / segmentSize;

  // Frequency map for symbols
  std::unordered_map<std::string, size_t> symbolFrequency;

  // Total symbol count
  size_t totalSymbols = 0;

  // Iterate through the dataset and extract symbols
  for (size_t i = 0; i < numSegments; ++i) {
    size_t base = i * segmentSize;

    // Leading pattern (e.g., A)
    std::string leadingPattern(byteArray.begin() + base, byteArray.begin() + base + leadingBytes);
    symbolFrequency[leadingPattern]++;
    totalSymbols++;

    // Content pattern (e.g., BC)
    std::string contentPattern(byteArray.begin() + base + leadingBytes,
                               byteArray.begin() + base + leadingBytes + contentBytes);
    symbolFrequency[contentPattern]++;
    totalSymbols++;

    // Trailing pattern (e.g., D)
    std::string trailingPattern(byteArray.begin() + base + leadingBytes + contentBytes,
                                byteArray.begin() + base + segmentSize);
    symbolFrequency[trailingPattern]++;
    totalSymbols++;
  }

  // Calculate entropy
  return calculateEntropy(symbolFrequency, totalSymbols);
}
