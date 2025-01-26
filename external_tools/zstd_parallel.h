#ifndef ZSTD_PARALLEL_H
#define ZSTD_PARALLEL_H

#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <zstd.h>
#include <chrono>
#include <cstdint>
#include <omp.h>
#include <numeric>

#include "profiling_info.h"

// Global data (defined elsewhere)
extern std::vector<uint8_t> globalByteArray;

//-----------------------------------------------------------------------------
// Basic Zstd Compression/Decompression
//-----------------------------------------------------------------------------
inline size_t compressWithZstd(
    const std::vector<uint8_t>& data,
    std::vector<uint8_t>& compressedData,
    int compressionLevel = 3
) {
    size_t cBuffSize = ZSTD_compressBound(data.size());
    compressedData.resize(cBuffSize);

    size_t cSize = ZSTD_compress(
        compressedData.data(),
        cBuffSize,
        data.data(),
        data.size(),
        compressionLevel
    );
    if (ZSTD_isError(cSize)) {
        std::cerr << "Zstd compression error: " << ZSTD_getErrorName(cSize) << std::endl;
        return 0;
    }
    compressedData.resize(cSize);
    return cSize;
}

inline size_t decompressWithZstd(
    const std::vector<uint8_t>& compressedData,
    std::vector<uint8_t>& decompressedData,
    size_t originalSize
) {
    decompressedData.resize(originalSize);
    size_t dSize = ZSTD_decompress(
        decompressedData.data(),
        originalSize,
        compressedData.data(),
        compressedData.size()
    );
    if (ZSTD_isError(dSize)) {
        std::cerr << "Zstd decompression error: " << ZSTD_getErrorName(dSize) << std::endl;
        return 0;
    }
    return dSize;
}

//-----------------------------------------------------------------------------
// Reorder in One Pass with a Nested Vector, e.g. {{1,3},{2},{4}}
// each inner vector is a sub-config specifying 1-based byte indices
//-----------------------------------------------------------------------------
inline void splitBytesIntoComponentsNested(
    const std::vector<uint8_t>& byteArray,
    std::vector<std::vector<uint8_t>>& outputComponents,
    const std::vector<std::vector<size_t>>& allComponentSizes,
    int numThreads
) {
    // 1) total bytes per element
    size_t totalBytesPerElement = 0;
    for (const auto& group : allComponentSizes) {
        totalBytesPerElement += group.size();
    }

    // 2) how many "elements"
    size_t numElements = byteArray.size() / totalBytesPerElement;

    // 3) Allocate output
    outputComponents.resize(allComponentSizes.size());
    for (size_t i = 0; i < allComponentSizes.size(); i++) {
        // group i will hold (numElements * groupSize) bytes
        size_t groupSize = allComponentSizes[i].size();
        outputComponents[i].resize(numElements * groupSize);
    }

    // 4) Reorder in parallel across sub-configs
#pragma omp parallel for num_threads(numThreads)
    for (size_t compIdx = 0; compIdx < allComponentSizes.size(); compIdx++) {
        const auto& groupIndices = allComponentSizes[compIdx];
        size_t groupSize = groupIndices.size();

        for (size_t elem = 0; elem < numElements; elem++) {
            size_t writePos = elem * groupSize;

            for (size_t sub = 0; sub < groupSize; sub++) {
                size_t idxInElem = groupIndices[sub] - 1; // 1-based -> 0-based
                size_t globalIndex = elem * totalBytesPerElement + idxInElem;

                outputComponents[compIdx][writePos + sub] = byteArray[globalIndex];
            }
        }
    }
}

//-----------------------------------------------------------------------------
// Full (no decomposition) compression
//-----------------------------------------------------------------------------
inline size_t zstdCompression(
    const std::vector<uint8_t>& data,
    ProfilingInfo &pi,
    std::vector<uint8_t>& compressedData
) {
    size_t cSize = compressWithZstd(data, compressedData, 3);
    pi.type = "FullCompression";
    return cSize;
}
inline void zstdDecompression(
    const std::vector<uint8_t>& compressedData,
    std::vector<uint8_t>& decompressedData,
    ProfilingInfo &pi
) {
    decompressWithZstd(compressedData, decompressedData, globalByteArray.size());
}

//-----------------------------------------------------------------------------
// Decomposed SEQUENTIAL
// Reorder entire dataset in ONE pass, then compress each sub-chunk sequentially
//-----------------------------------------------------------------------------
inline size_t zstdDecomposedSequential(
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

        size_t cSize = compressWithZstd(subChunks[i], compressedComponents[i], 3);
        totalCompressedSize += cSize;

        auto endOne = std::chrono::high_resolution_clock::now();
        double compTime = std::chrono::duration<double>(endOne - startOne).count();
        pi.component_times.push_back(compTime);
    }

    // record total time
    auto endAll = std::chrono::high_resolution_clock::now();
    pi.total_time_compressed = std::chrono::duration<double>(endAll - startAll).count();

    // compute ratio
    if (totalCompressedSize == 0) {
        pi.com_ratio = 0.0;
    } else {
        pi.com_ratio = static_cast<double>(data.size()) / static_cast<double>(totalCompressedSize);
    }

    return totalCompressedSize;
}

inline void zstdDecomposedSequentialDecompression(
    const std::vector<std::vector<uint8_t>>& compressedComponents,
    ProfilingInfo& pi,
    const std::vector<std::vector<size_t>>& allComponentSizes
) {
    auto startAll = std::chrono::high_resolution_clock::now();

    // 1) figure out how big each chunk was from reorder
    // We replicate the logic from "splitBytesIntoComponentsNested" to deduce sizes
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

    // 2) Decompress each chunk sequentially
    for (size_t i = 0; i < compressedComponents.size(); i++) {
        auto startOne = std::chrono::high_resolution_clock::now();

        std::vector<uint8_t> temp;
        decompressWithZstd(compressedComponents[i], temp, chunkSizes[i]);
        // If needed, do an *inverse reorder* to assemble back into a single array

        auto endOne = std::chrono::high_resolution_clock::now();
        double decTime = std::chrono::duration<double>(endOne - startOne).count();
        pi.component_times.push_back(decTime);
    }

    auto endAll = std::chrono::high_resolution_clock::now();
    pi.total_time_decompressed = std::chrono::duration<double>(endAll - startAll).count();
}

//-----------------------------------------------------------------------------
// Decomposed PARALLEL
// Reorder entire dataset in ONE pass, then compress each sub-chunk in parallel
//-----------------------------------------------------------------------------
inline size_t zstdDecomposedParallel(
    const std::vector<uint8_t>& data,
    ProfilingInfo& pi,
    std::vector<std::vector<uint8_t>>& compressedComponents,
    const std::vector<std::vector<size_t>>& allComponentSizes,
    int numThreads
) {
    auto startAll = std::chrono::high_resolution_clock::now();

    // 1) Reorder data in one pass (in parallel if you want)
    std::vector<std::vector<uint8_t>> subChunks;
    splitBytesIntoComponentsNested(data, subChunks, allComponentSizes, /*numThreads=*/1);

    // 2) Compress each chunk in parallel
    compressedComponents.resize(subChunks.size());
    omp_set_num_threads(numThreads);

    size_t totalCompressedSize = 0;

#pragma omp parallel for reduction(+:totalCompressedSize)
    for (int i = 0; i < (int)subChunks.size(); i++) {
        auto startOne = std::chrono::high_resolution_clock::now();

        std::vector<uint8_t> compData;
        size_t cSize = compressWithZstd(subChunks[i], compData, 3);
        compressedComponents[i] = std::move(compData);
        totalCompressedSize += cSize;

        auto endOne = std::chrono::high_resolution_clock::now();
        double compTime = std::chrono::duration<double>(endOne - startOne).count();

        // store each component's time
#pragma omp critical
        {
            pi.component_times.push_back(compTime);
        }
    }

    // total time
    auto endAll = std::chrono::high_resolution_clock::now();
    pi.total_time_compressed = std::chrono::duration<double>(endAll - startAll).count();

    // compute ratio
    if (totalCompressedSize == 0) {
        pi.com_ratio = 0.0;
    } else {
        pi.com_ratio = double(data.size()) / double(totalCompressedSize);
    }
    return totalCompressedSize;
}

inline std::vector<uint8_t> zstdDecomposedParallelDecompression(
    const std::vector<std::vector<uint8_t>>& compressedComponents,
    ProfilingInfo& pi,
    const std::vector<std::vector<size_t>>& allComponentSizes,
    int numThreads
) {
    auto startAll = std::chrono::high_resolution_clock::now();

    // figure out how large each chunk was
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

    // We'll do a dummy final array
    std::vector<uint8_t> finalReconstructed(globalByteArray.size());

    // Decompress in parallel
    omp_set_num_threads(numThreads);

#pragma omp parallel for
    for (int i = 0; i < (int)compressedComponents.size(); i++) {
        auto startOne = std::chrono::high_resolution_clock::now();
        std::vector<uint8_t> temp;
        decompressWithZstd(compressedComponents[i], temp, chunkSizes[i]);
        // If needed, do inverse reorder to fill finalReconstructed

        auto endOne = std::chrono::high_resolution_clock::now();
        double decTime = std::chrono::duration<double>(endOne - startOne).count();

#pragma omp critical
        {
            pi.component_times.push_back(decTime);
        }
    }

    auto endAll = std::chrono::high_resolution_clock::now();
    pi.total_time_decompressed = std::chrono::duration<double>(endAll - startAll).count();
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

#endif // ZSTD_PARALLEL_H
