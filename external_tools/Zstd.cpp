#include <benchmark/benchmark.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cstdint>
#include <zstd.h>
#include "common.h"

// Function to convert a float array to a uint8_t vector
std::vector<uint8_t> convertFloatToBytes(const std::vector<float>& floatArray) {
    std::vector<uint8_t> byteArray(floatArray.size() * 4);

    // Iterate through the float array and cast each float to its byte representation
    for (size_t i = 0; i < floatArray.size(); i++) {
        uint8_t* floatBytes = reinterpret_cast<uint8_t*>(const_cast<float*>(&floatArray[i]));
        for (size_t j = 0; j < 4; j++) {
            byteArray[i * 4 + j] = floatBytes[j];
        }
    }

    return byteArray;
}

// split bytes into three components: leading, content, and trailing
void splitBytesIntoComponents(const std::vector<uint8_t>& byteArray,
                              std::vector<uint8_t>& leading,
                              std::vector<uint8_t>& content,
                              std::vector<uint8_t>& trailing) {
    size_t floatCount = byteArray.size() / 4;

    for (size_t i = 0; i < floatCount; i++) {
        // Leading component: first byte
        leading.push_back(byteArray[i * 4]);

        // Content component: second and third bytes
        content.push_back(byteArray[i * 4 + 1]);
        content.push_back(byteArray[i * 4 + 2]);

        // Trailing component: fourth byte
        trailing.push_back(byteArray[i * 4 + 3]);
    }
}


size_t compressWithZstd(const std::vector<uint8_t>& data, std::vector<uint8_t>& compressedData, int compressionLevel) {
    size_t const cBuffSize = ZSTD_compressBound(data.size());
    compressedData.resize(cBuffSize);

    // Compress with Zstd
    size_t const cSize = ZSTD_compress(compressedData.data(), cBuffSize, data.data(), data.size(), compressionLevel);
    CHECK_ZSTD(cSize);

    // Resize to actual compressed size
    compressedData.resize(cSize);
    return cSize;
}

//  decompress a byte vector with Zstd and return decompressed size
size_t decompressWithZstd(const std::vector<uint8_t>& compressedData, std::vector<uint8_t>& decompressedData, size_t originalSize) {
    decompressedData.resize(originalSize);

    // Decompress with Zstd
    size_t const dSize = ZSTD_decompress(decompressedData.data(), originalSize, compressedData.data(), compressedData.size());
    CHECK_ZSTD(dSize);

    // Verify decompressed size matches the original size
    CHECK(dSize == originalSize, "Decompressed size does not match original size!");

    return dSize;
}

//  verify decompressed data matches original data
bool verifyData(const std::vector<uint8_t>& original, const std::vector<uint8_t>& decompressed) {
    if (original.size() != decompressed.size()) {
        return false;
    }

    // Compare each element
    for (size_t i = 0; i < original.size(); i++) {
        if (original[i] != decompressed[i]) {
            return false;
        }
    }

    return true;
}

// comp ratio
double calculateCompressionRatio(size_t originalSize, size_t compressedSize) {
    return static_cast<double>(originalSize) / static_cast<double>(compressedSize);
}


static void ZSTD_BENCH(benchmark::State& state) {
    int fSize = state.range(0);
    int zstdLevel = state.range(1);

    // Generate a float array
    std::vector<float> floatArray(fSize);
    for (size_t i = 0; i < fSize; i++) {
        floatArray[i] = static_cast<float>(i);
    }

    // Convert float array to byte array
    std::vector<uint8_t> byteArray = convertFloatToBytes(floatArray);

    // Split into components
    std::vector<uint8_t> leading, content, trailing;
    splitBytesIntoComponents(byteArray, leading, content, trailing);

    std::vector<uint8_t> compressedLeading, compressedContent, compressedTrailing;
    std::vector<uint8_t> decompressedLeading, decompressedContent, decompressedTrailing;

    //  compression, decompression, and comp ratio for each component
    for (auto _: state) {
        // Compress each component
        size_t leadingCompressedSize = compressWithZstd(leading, compressedLeading, zstdLevel);
        size_t contentCompressedSize = compressWithZstd(content, compressedContent, zstdLevel);
        size_t trailingCompressedSize = compressWithZstd(trailing, compressedTrailing, zstdLevel);

        // Decompress each component
        decompressWithZstd(compressedLeading, decompressedLeading, leading.size());
        decompressWithZstd(compressedContent, decompressedContent, content.size());
        decompressWithZstd(compressedTrailing, decompressedTrailing, trailing.size());

        // Verify decompressed data matches original data
        bool leadingMatch = verifyData(leading, decompressedLeading);
        bool contentMatch = verifyData(content, decompressedContent);
        bool trailingMatch = verifyData(trailing, decompressedTrailing);

        //  comp ratios
        double leadingCompRatio = calculateCompressionRatio(leading.size(), leadingCompressedSize);
        double contentCompRatio = calculateCompressionRatio(content.size(), contentCompressedSize);
        double trailingCompRatio = calculateCompressionRatio(trailing.size(), trailingCompressedSize);

        // total comp ratio
        size_t totalOriginalSize = leading.size() + content.size() + trailing.size();
        size_t totalCompressedSize = leadingCompressedSize + contentCompressedSize + trailingCompressedSize;
        double totalCompRatio = calculateCompressionRatio(totalOriginalSize, totalCompressedSize);

        // Print compression results
        printf("Compressed sizes - Leading: %zu, Content: %zu, Trailing: %zu\n",
               leadingCompressedSize, contentCompressedSize, trailingCompressedSize);
        printf("Compression Ratios - Leading: %.2f, Content: %.2f, Trailing: %.2f\n",
               leadingCompRatio, contentCompRatio, trailingCompRatio);
        printf("Total Compression Ratio: %.2f\n", totalCompRatio);


        if (leadingMatch && contentMatch && trailingMatch) {
            printf("Decompression successful: All components match the original data.\n");
        } else {
            if (!leadingMatch) {
                printf("Decompression failed for leading component.\n");
            }
            if (!contentMatch) {
                printf("Decompression failed for content component.\n");
            }
            if (!trailingMatch) {
                printf("Decompression failed for trailing component.\n");
            }
        }
    }
}


BENCHMARK(ZSTD_BENCH)
    ->Args({1000000, 3})
    ->Args({1000000, 22})
    ->Args({2000000, 3})
    ->Args({2000000, 22})
    ->Args({1000000, 1})
    ->Args({2000000, 1});

BENCHMARK_MAIN();
