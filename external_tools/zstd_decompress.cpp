#include <benchmark/benchmark.h>
#include <iostream>
#include <vector>
#include <bitset>
#include <zstd.h>
#include <cstring>
#include <cstdlib>
#include "common.h"  // Helper functions, CHECK(), and CHECK_ZSTD()
#include "zstd.h"
#include "utils.h"   // Inline compareBuffers function and binaryToFloat declaration

// Convert float to IEEE 754 binary representation
std::bitset<32> floatToIEEE754(float value) {
    uint32_t intBits = *reinterpret_cast<uint32_t*>(&value);
    return std::bitset<32>(intBits);
}

// Decompose the 32-bit binary into three parts
std::tuple<std::bitset<32>, std::bitset<32>, std::bitset<32>> decomposeArrayThree(
    const std::bitset<32>& binaryData, int maxLead, int minTail) {

    std::bitset<32> leadingZeroArray;
    std::bitset<32> contentArray;
    std::bitset<32> trailingZeroArray;

    for (int i = 0; i < maxLead; ++i) {
        leadingZeroArray[i] = binaryData[i];
    }
    for (int i = maxLead; i < minTail; ++i) {
        contentArray[i - maxLead] = binaryData[i];
    }
    for (int i = minTail; i < 32; ++i) {
        trailingZeroArray[i - minTail] = binaryData[i];
    }

    return std::make_tuple(leadingZeroArray, contentArray, trailingZeroArray);
}

// Function to convert data to binary, decompose, and convert each part back to float
void prepareDataAndSizes(const std::vector<float>& fBuff,
                         std::vector<float>& leadingFloats,
                         std::vector<float>& contentFloats,
                         std::vector<float>& trailingFloats,
                         size_t& leadingSize, size_t& contentSize, size_t& trailingSize) {

    // Decompose each float into leading, content, and trailing parts
    for (const float& f : fBuff) {
        // Convert to binary
        std::bitset<32> binary = floatToIEEE754(f);

        // Decompose the binary representation
        auto [lead, cont, trail] = decomposeArrayThree(binary, 8, 24);

        // Convert each part back to float
        leadingFloats.push_back(binaryToFloat(lead));
        contentFloats.push_back(binaryToFloat(cont));
        trailingFloats.push_back(binaryToFloat(trail));
    }

    // Calculate sizes of each component
    leadingSize = leadingFloats.size() * sizeof(float);
    contentSize = contentFloats.size() * sizeof(float);
    trailingSize = trailingFloats.size() * sizeof(float);
}

// Benchmark function for the full process, including size check
static void ZSTD_BENCH_FULL(benchmark::State& state) {
    int fSize = state.range(0);
    int zstdLevel = state.range(1);

    // Allocate memory for the original buffer
    std::vector<float> fBuff(fSize);
    for (size_t i = 0; i < fSize; ++i) {
        fBuff[i] = static_cast<float>(i) + 0.123f;
    }

    // Prepare data and calculate sizes for leading, content, and trailing parts
    std::vector<float> leadingFloats, contentFloats, trailingFloats;
    size_t leadingSize = 0, contentSize = 0, trailingSize = 0;
    prepareDataAndSizes(fBuff, leadingFloats, contentFloats, trailingFloats,
                        leadingSize, contentSize, trailingSize);

    // Calculate original data size
    size_t originalSize = fBuff.size() * sizeof(float);
    size_t totalDecomposedSize = leadingSize + contentSize + trailingSize;

    // Check if the total size of decomposed parts matches the original size
    if (totalDecomposedSize != originalSize) {
        std::cerr << "Error: Total size of decomposed parts (" << totalDecomposedSize
                  << " bytes) does not match original size (" << originalSize << " bytes)!\n";
        return;
    }

    // Display sizes of each component
    std::cout << "\nOriginal Size: " << originalSize << " bytes\n";
    std::cout << "Total Size of Decomposed Parts: " << totalDecomposedSize << " bytes\n";
    std::cout << "Leading Part Size: " << leadingSize << " bytes\n";
    std::cout << "Content Part Size: " << contentSize << " bytes\n";
    std::cout << "Trailing Part Size: " << trailingSize << " bytes\n";

    // Calculate the maximum possible compressed size
    size_t const cBuffSize = ZSTD_compressBound(fSize * sizeof(float));
    void* const cBuff = malloc_orDie(cBuffSize);

    // Create a compression context
    ZSTD_CCtx* cctx = ZSTD_createCCtx();
    ZSTD_CCtx_setParameter(cctx, ZSTD_c_compressionLevel, zstdLevel);

    for (auto _ : state) {
        // Compress the leading part
        size_t leadingCSize = ZSTD_compressCCtx(cctx, cBuff, cBuffSize, leadingFloats.data(), leadingSize, zstdLevel);
        CHECK_ZSTD(leadingCSize);

        // Compress the content part
        size_t contentCSize = ZSTD_compressCCtx(cctx, cBuff, cBuffSize, contentFloats.data(), contentSize, zstdLevel);
        CHECK_ZSTD(contentCSize);

        // Compress the trailing part
        size_t trailingCSize = ZSTD_compressCCtx(cctx, cBuff, cBuffSize, trailingFloats.data(), trailingSize, zstdLevel);
        CHECK_ZSTD(trailingCSize);

        // Display compressed sizes and compression ratios during the first iteration
        if (state.iterations() == 1) {
            std::cout << "\nCompressed Leading Size: " << leadingCSize << " bytes\n";
            std::cout << "Compressed Content Size: " << contentCSize << " bytes\n";
            std::cout << "Compressed Trailing Size: " << trailingCSize << " bytes\n";
            std::cout << "Compression Ratio (Leading): " << static_cast<double>(leadingSize) / leadingCSize << "\n";
            std::cout << "Compression Ratio (Content): " << static_cast<double>(contentSize) / contentCSize << "\n";
            std::cout << "Compression Ratio (Trailing): " << static_cast<double>(trailingSize) / trailingCSize << "\n";
        }
    }

    // Free resources
    free(cBuff);
    ZSTD_freeCCtx(cctx);
}

// Register the benchmark
int main(int argc, char** argv) {
    ::benchmark::Initialize(&argc, argv);

    // Register the full process benchmark
    benchmark::RegisterBenchmark("ZSTD_BENCH_FULL", ZSTD_BENCH_FULL)
        ->Args({1000, 22})  // Original size: 1000 * 4 bytes = 4000 bytes
        ->Args({2000000, 22});  // Larger size

    ::benchmark::RunSpecifiedBenchmarks();
    return 0;
}
