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

// Split bytes into three components: leading, content, and trailing
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

// Function to compress a byte vector with Zstd and return compressed size
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

// Decompress a byte vector with Zstd and return decompressed size
size_t decompressWithZstd(const std::vector<uint8_t>& compressedData, std::vector<uint8_t>& decompressedData, size_t originalSize) {
    decompressedData.resize(originalSize);

    // Decompress with Zstd
    size_t const dSize = ZSTD_decompress(decompressedData.data(), originalSize, compressedData.data(), compressedData.size());
    CHECK_ZSTD(dSize);

    return dSize;
}

// Verify if two byte arrays are equal
bool verifyBytes(const std::vector<uint8_t>& original, const std::vector<uint8_t>& reconstructed) {
    if (original.size() != reconstructed.size()) {
        return false;
    }

    // Compare each byte
    for (size_t i = 0; i < original.size(); i++) {
        if (original[i] != reconstructed[i]) {
            printf("Byte mismatch at index %zu: original = %d, reconstructed = %d\n",
                   i, original[i], reconstructed[i]);
            return false;
        }
    }
    return true;
}

// Compression ratio calculation
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


    for (auto _: state) {
        // Compress each component
        size_t leadingCompressedSize = compressWithZstd(leading, compressedLeading, zstdLevel);
        size_t contentCompressedSize = compressWithZstd(content, compressedContent, zstdLevel);
        size_t trailingCompressedSize = compressWithZstd(trailing, compressedTrailing, zstdLevel);

        // Decompress each component
        decompressWithZstd(compressedLeading, decompressedLeading, leading.size());
        decompressWithZstd(compressedContent, decompressedContent, content.size());
        decompressWithZstd(compressedTrailing, decompressedTrailing, trailing.size());

        // Reconstruct full byte array from decompressed
        std::vector<uint8_t> reconstructedBytes(byteArray.size());
        size_t floatCount = byteArray.size() / 4;


        for (size_t i = 0; i < floatCount; i++) {
            reconstructedBytes[i * 4] = decompressedLeading[i];
            reconstructedBytes[i * 4 + 1] = decompressedContent[i * 2];
            reconstructedBytes[i * 4 + 2] = decompressedContent[i * 2 + 1];
            reconstructedBytes[i * 4 + 3] = decompressedTrailing[i];
        }

        // Verify reconstructed byte array matches the original byte array
        bool byteMatch = verifyBytes(byteArray, reconstructedBytes);

        // Calculate total compression ratio
        size_t totalOriginalSize = leading.size() + content.size() + trailing.size();
        size_t totalCompressedSize = leadingCompressedSize + contentCompressedSize + trailingCompressedSize;
        double totalCompRatio = calculateCompressionRatio(totalOriginalSize, totalCompressedSize);


        printf("Compressed sizes - Leading: %zu, Content: %zu, Trailing: %zu\n",
               leadingCompressedSize, contentCompressedSize, trailingCompressedSize);
        printf("Compression Ratios - Leading: %.2f, Content: %.2f, Trailing: %.2f\n",
               calculateCompressionRatio(leading.size(), leadingCompressedSize),
               calculateCompressionRatio(content.size(), contentCompressedSize),
               calculateCompressionRatio(trailing.size(), trailingCompressedSize));
        printf("Total Compression Ratio: %.2f\n", totalCompRatio);


        if (byteMatch) {
            printf("Reconstructed byte array matches the original data.\n");
        } else {
            printf("Reconstructed byte array does not match the original.\n");
        }
    }
}

BENCHMARK(ZSTD_BENCH)
    ->Args({1000000, 3});


BENCHMARK_MAIN();
