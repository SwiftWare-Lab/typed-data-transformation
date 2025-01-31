#ifndef FASTLZ_PARALLEL_H
#define FASTLZ_PARALLEL_H

#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cstdint>
#include <omp.h>
#include <numeric>
#include "fastlz.h">
#include"fastlz.c"
#include <immintrin.h> // For optional AVX2 usage in reordering

#include "profiling_info.h"

// Global data (defined elsewhere)
extern std::vector<uint8_t> globalByteArray;

//-----------------------------------------------------------------------------
// Basic FastLZ Compression/Decompression
//-----------------------------------------------------------------------------
inline size_t compressWithFastLZ(
    const std::vector<uint8_t>& data,
    std::vector<uint8_t>& compressedData
) {
    // Slightly larger buffer for worst case
    size_t cBuffSize = static_cast<size_t>(data.size() * 1.05 + 16);
    compressedData.resize(cBuffSize);

    size_t cSize = fastlz_compress(
        data.data(),
        data.size(),
        compressedData.data()
    );

    compressedData.resize(cSize);
    return cSize;
}

inline size_t decompressWithFastLZ(
    const std::vector<uint8_t>& compressedData,
    std::vector<uint8_t>& decompressedData,
    size_t originalSize
) {
    decompressedData.resize(originalSize);
    size_t dSize = fastlz_decompress(
        compressedData.data(),
        compressedData.size(),
        decompressedData.data(),
        originalSize
    );

    if (dSize == 0) {
        std::cerr << "FastLZ decompression error: Invalid input data" << std::endl;
        return 0;
    }
    return dSize;
}

/////////////////////////////////////////////////////////////
//  reassembly
/////////////////////////////////////////////////////////////

//-----------------------------------------------------------------------------
// Optimized In-Place Reordering for Decomposition-Based Compression
// Uses SIMD (AVX2) for faster memory operations and OpenMP for parallel execution
//-----------------------------------------------------------------------------
inline void splitBytesIntoComponentsNested(
    const std::vector<uint8_t>& byteArray,
    std::vector<std::vector<uint8_t>>& outputComponents,
    const std::vector<std::vector<size_t>>& allComponentSizes,
    int numThreads
) {
    size_t totalBytesPerElement = 0;
    for (const auto& group : allComponentSizes) {
        totalBytesPerElement += group.size();
    }

    size_t numElements = byteArray.size() / totalBytesPerElement;

    outputComponents.resize(allComponentSizes.size());



    std::vector<uint8_t> temp(byteArray);

    // Resize
    for (size_t i = 0; i < allComponentSizes.size(); i++) {
        outputComponents[i].resize(numElements * allComponentSizes[i].size());
    }

#pragma omp parallel for num_threads(numThreads) schedule(dynamic)
    for (size_t elem = 0; elem < numElements; elem++) {
        for (size_t compIdx = 0; compIdx < allComponentSizes.size(); compIdx++) {
            const auto& groupIndices = allComponentSizes[compIdx];
            size_t groupSize = groupIndices.size();
            size_t writePos = elem * groupSize;

            size_t sub = 0;
#ifdef __AVX2__

#endif
            for (; sub < groupSize; sub++) {
                size_t idxInElem = groupIndices[sub] - 1;
                size_t globalSrcIdx = elem * totalBytesPerElement + idxInElem;
                outputComponents[compIdx][writePos + sub] = temp[globalSrcIdx];
            }
        }
    }
}

//-----------------------------------------------------------------------------
// Simple reordering
//-----------------------------------------------------------------------------
inline void splitBytesIntoComponentsNested1(
    const std::vector<uint8_t>& byteArray,
    std::vector<std::vector<uint8_t>>& outputComponents,
    const std::vector<std::vector<size_t>>& allComponentSizes,
    int numThreads
) {
    size_t totalBytesPerElement = 0;
    for (const auto& group : allComponentSizes) {
        totalBytesPerElement += group.size();
    }

    size_t numElements = byteArray.size() / totalBytesPerElement;

    outputComponents.resize(allComponentSizes.size());
    for (size_t i = 0; i < allComponentSizes.size(); i++) {
        size_t groupSize = allComponentSizes[i].size();
        outputComponents[i].resize(numElements * groupSize);
    }

#pragma omp parallel for num_threads(numThreads)
    for (size_t compIdx = 0; compIdx < allComponentSizes.size(); compIdx++) {
        const auto& groupIndices = allComponentSizes[compIdx];
        size_t groupSize = groupIndices.size();
        for (size_t elem = 0; elem < numElements; elem++) {
            size_t writePos = elem * groupSize;
            for (size_t sub = 0; sub < groupSize; sub++) {
                size_t idxInElem = groupIndices[sub] - 1;
                size_t globalIndex = elem * totalBytesPerElement + idxInElem;
                outputComponents[compIdx][writePos + sub] = byteArray[globalIndex];
            }
        }
    }
}

///////////////////////////
// Reassembly
///////////////////////////

inline void reassembleBytesFromComponentsNested(
    const std::vector<std::vector<uint8_t>>& inputComponents,
    std::vector<uint8_t>& byteArray,
    const std::vector<std::vector<size_t>>& allComponentSizes,
    int numThreads
) {
    size_t totalBytesPerElement = 0;
    for (const auto& group : allComponentSizes) {
        totalBytesPerElement += group.size();
    }

    size_t numElements = byteArray.size() / totalBytesPerElement;

#pragma omp parallel for num_threads(numThreads) schedule(dynamic)
    for (size_t compIdx = 0; compIdx < inputComponents.size(); compIdx++) {
        const auto& groupIndices = allComponentSizes[compIdx];
        const auto& componentData = inputComponents[compIdx];
        size_t groupSize = groupIndices.size();

        for (size_t elem = 0; elem < numElements; elem++) {
            size_t readPos = elem * groupSize;
            size_t sub = 0;
#ifdef __AVX2__

#endif
            for (; sub < groupSize; sub++) {
                size_t idxInElem = groupIndices[sub] - 1;
                size_t globalIndex = elem * totalBytesPerElement + idxInElem;
                byteArray[globalIndex] = componentData[readPos + sub];
            }
        }
    }
}

inline void reassembleBytesFromComponentsNested1(
    const std::vector<std::vector<uint8_t>>& inputComponents,
    std::vector<uint8_t>& byteArray,
    const std::vector<std::vector<size_t>>& allComponentSizes,
    int numThreads
) {
    size_t totalBytesPerElement = 0;
    for (const auto& group : allComponentSizes) {
        totalBytesPerElement += group.size();
    }

    size_t numElements = byteArray.size() / totalBytesPerElement;

#pragma omp parallel for num_threads(numThreads)
    for (size_t compIdx = 0; compIdx < inputComponents.size(); compIdx++) {
        const auto& groupIndices = allComponentSizes[compIdx];
        const auto& componentData = inputComponents[compIdx];
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

///////////////////////////////////////////
// Full  compression
///////////////////////////////////////////
inline size_t fastlzCompression(
    const std::vector<uint8_t>& data,
    ProfilingInfo &pi,
    std::vector<uint8_t>& compressedData
) {
    auto start = std::chrono::high_resolution_clock::now();
    size_t cSize = compressWithFastLZ(data, compressedData);
    auto end = std::chrono::high_resolution_clock::now();

    pi.type = "FullCompression";
    pi.total_time_compressed = std::chrono::duration<double>(end - start).count();
    pi.com_ratio = cSize == 0 ? 0.0 : static_cast<double>(data.size()) / static_cast<double>(cSize);

    return cSize;
}

inline void fastlzDecompression(
    const std::vector<uint8_t>& compressedData,
    std::vector<uint8_t>& decompressedData,
    ProfilingInfo &pi
) {
    auto start = std::chrono::high_resolution_clock::now();
    size_t dSize = decompressWithFastLZ(compressedData, decompressedData, globalByteArray.size());
    auto end = std::chrono::high_resolution_clock::now();

    pi.total_time_decompressed = std::chrono::duration<double>(end - start).count();
    if (dSize == 0) {
        std::cerr << "[ERROR] Decompression failed." << std::endl;
    }
}

/////////////////////////////////////////////
// Decomposed SEQ
/////////////////////////////////////////////
inline size_t fastlzDecomposedSequential(
    const std::vector<uint8_t>& data,
    ProfilingInfo& pi,
    std::vector<std::vector<uint8_t>>& compressedComponents,
    const std::vector<std::vector<size_t>>& allComponentSizes
) {
    auto startAll = std::chrono::high_resolution_clock::now();

    // 1) Reorder data in one pass
    std::vector<std::vector<uint8_t>> subChunks;
    splitBytesIntoComponentsNested(data, subChunks, allComponentSizes, /*numThreads=*/1);

    // 2) Compress each chunk sequentially
    compressedComponents.resize(subChunks.size());

    size_t totalCompressedSize = 0;
    for (size_t i = 0; i < subChunks.size(); i++) {
        auto startOne = std::chrono::high_resolution_clock::now();
        size_t cSize = compressWithFastLZ(subChunks[i], compressedComponents[i]);
        totalCompressedSize += cSize;
        auto endOne = std::chrono::high_resolution_clock::now();
        double compTime = std::chrono::duration<double>(endOne - startOne).count();
        pi.component_times.push_back(compTime);
    }

    auto endAll = std::chrono::high_resolution_clock::now();
    pi.total_time_compressed = std::chrono::duration<double>(endAll - startAll).count();
    pi.com_ratio = (totalCompressedSize == 0) ? 0.0 : static_cast<double>(data.size()) / static_cast<double>(totalCompressedSize);

    return totalCompressedSize;
}

inline void fastlzDecomposedSequentialDecompression(
    const std::vector<std::vector<uint8_t>>& compressedComponents,
    ProfilingInfo& pi,
    const std::vector<std::vector<size_t>>& allComponentSizes
) {
    auto startAll = std::chrono::high_resolution_clock::now();

    // 1) Compute total bytes per element
    size_t totalBytesPerElement = 0;
    for (auto & group : allComponentSizes) {
        totalBytesPerElement += group.size();
    }
    size_t numElements = globalByteArray.size() / totalBytesPerElement;

    // 2) Decompress each chunk sequentially
    std::vector<std::vector<uint8_t>> decompressedSubChunks(compressedComponents.size());
    std::vector<size_t> chunkSizes;
    chunkSizes.reserve(allComponentSizes.size());
    for (auto & group : allComponentSizes) {
        chunkSizes.push_back(numElements * group.size());
    }

    for (size_t i = 0; i < compressedComponents.size(); i++) {
        auto startOne = std::chrono::high_resolution_clock::now();
        std::vector<uint8_t> temp;
        decompressWithFastLZ(compressedComponents[i], temp, chunkSizes[i]);
        decompressedSubChunks[i] = std::move(temp);
        auto endOne = std::chrono::high_resolution_clock::now();
        pi.component_times.push_back(std::chrono::duration<double>(endOne - startOne).count());
    }

    // 3) Reassemble
    std::vector<uint8_t> finalReconstructed(globalByteArray.size());
    reassembleBytesFromComponentsNested(decompressedSubChunks, finalReconstructed, allComponentSizes, /*numThreads=*/1);

    // 4) Measure total time
    auto endAll = std::chrono::high_resolution_clock::now();
    pi.total_time_decompressed = std::chrono::duration<double>(endAll - startAll).count();

    // 5) Verify correctness
    if (finalReconstructed == globalByteArray) {
        std::cout << "[INFO] Reconstructed data matches the original!\n";
    } else {
        std::cerr << "[ERROR] Reconstructed data does NOT match the original.\n";
    }
}

/////////////////////////////////////////////
// Decomposed PARALLEL
/////////////////////////////////////////////
inline size_t fastlzDecomposedParallel(
    const std::vector<uint8_t>& data,
    ProfilingInfo& pi,
    std::vector<std::vector<uint8_t>>& compressedComponents,
    const std::vector<std::vector<size_t>>& allComponentSizes,
    int numThreads
) {
    auto startAll = std::chrono::high_resolution_clock::now();

    // 1) Reorder data in one pass
    std::vector<std::vector<uint8_t>> subChunks;
    // You can do it in parallel or single-thread. Let's do single-thread for the reorder step.
    splitBytesIntoComponentsNested(data, subChunks, allComponentSizes, /*numThreads=*/1);

    // 2) Compress in parallel
    compressedComponents.resize(subChunks.size());
    size_t totalCompressedSize = 0;

    omp_set_num_threads(numThreads);
#pragma omp parallel for reduction(+:totalCompressedSize)
    for (int i = 0; i < (int)subChunks.size(); i++) {
        auto startOne = std::chrono::high_resolution_clock::now();
        std::vector<uint8_t> compData;
        size_t cSize = compressWithFastLZ(subChunks[i], compData);
        compressedComponents[i] = std::move(compData);
        totalCompressedSize += cSize;
        auto endOne = std::chrono::high_resolution_clock::now();
        double compTime = std::chrono::duration<double>(endOne - startOne).count();
#pragma omp critical
        {
            pi.component_times.push_back(compTime);
        }
    }

    auto endAll = std::chrono::high_resolution_clock::now();
    pi.total_time_compressed = std::chrono::duration<double>(endAll - startAll).count();
    pi.com_ratio = (totalCompressedSize == 0) ? 0.0 : static_cast<double>(data.size()) / static_cast<double>(totalCompressedSize);
    return totalCompressedSize;
}

inline std::vector<uint8_t> fastlzDecomposedParallelDecompression(
    const std::vector<std::vector<uint8_t>>& compressedComponents,
    ProfilingInfo& pi,
    const std::vector<std::vector<size_t>>& allComponentSizes,
    int numThreads
) {
    auto startAll = std::chrono::high_resolution_clock::now();

    // 1) Determine chunk sizes
    size_t totalBytesPerElement = 0;
    for (auto & group : allComponentSizes) {
        totalBytesPerElement += group.size();
    }
    size_t numElements = globalByteArray.size() / totalBytesPerElement;

    std::vector<size_t> chunkSizes;
    chunkSizes.reserve(allComponentSizes.size());
    for (auto & group : allComponentSizes) {
        chunkSizes.push_back(numElements * group.size());
    }

    // 2) Decompress in parallel
    std::vector<std::vector<uint8_t>> decompressedSubChunks(compressedComponents.size());
    omp_set_num_threads(numThreads);
#pragma omp parallel for
    for (int i = 0; i < (int)compressedComponents.size(); i++) {
        auto startOne = std::chrono::high_resolution_clock::now();
        std::vector<uint8_t> temp;
        decompressWithFastLZ(compressedComponents[i], temp, chunkSizes[i]);
        decompressedSubChunks[i] = std::move(temp);
        auto endOne = std::chrono::high_resolution_clock::now();
        double decTime = std::chrono::duration<double>(endOne - startOne).count();
#pragma omp critical
        {
            pi.component_times.push_back(decTime);
        }
    }

    // 3) Reassemble
    std::vector<uint8_t> finalReconstructed(globalByteArray.size());
    reassembleBytesFromComponentsNested(
        decompressedSubChunks,
        finalReconstructed,
        allComponentSizes,
        /*numThreads=*/numThreads
    );

    // 4) Time
    auto endAll = std::chrono::high_resolution_clock::now();
    pi.total_time_decompressed = std::chrono::duration<double>(endAll - startAll).count();

    // 5) Verify correctness
    if (finalReconstructed == globalByteArray) {
        std::cout << "[INFO] Reconstructed data matches the original!\n";
    } else {
        std::cerr << "[ERROR] Reconstructed data does NOT match the original.\n";
    }

    return finalReconstructed;
}

//-----------------------------------------------------------------------------
// Compute Overall Compression Ratio
//-----------------------------------------------------------------------------
inline double calculateCompressionRatio(size_t originalSize, size_t compressedSize) {
    return (compressedSize == 0)
        ? 0.0
        : static_cast<double>(originalSize) / static_cast<double>(compressedSize);
}

#endif 
