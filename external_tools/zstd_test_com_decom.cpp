#include <benchmark/benchmark.h>
#include <cstdio>     // printf
#include <cstdlib>    // free
#include <cstring>    // strlen, strcat, memset
#include <zstd.h>     // presumes zstd library is installed
#include "common.h"   // Helper functions, CHECK(), and CHECK_ZSTD()
#include "zstd.h"

// Compare original and decompressed data
bool compareBuffers(const float* original, const float* decompressed, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        if (original[i] != decompressed[i]) {
            return false;
        }
    }
    return true;
}

// Benchmarking function with decompression and verification
static void ZSTD_BENCH(benchmark::State& state) {
    int fSize = state.range(0);
    int zstdLevel = state.range(1);
    int numThreads = state.range(2);

    // Allocate memory for the original buffer
    float* const fBuff = (float*)malloc(fSize * sizeof(float));
    for (size_t i = 0; i < fSize; i++) fBuff[i] = (float)i;

    // Calculate the maximum possible compressed size
    size_t const cBuffSize = ZSTD_compressBound(fSize * sizeof(float));
    void* const cBuff = malloc_orDie(cBuffSize);

    // Create a compression context
    ZSTD_CCtx* cctx = ZSTD_createCCtx();
    ZSTD_CCtx_setParameter(cctx, ZSTD_c_compressionLevel, zstdLevel);
    ZSTD_CCtx_setParameter(cctx, ZSTD_c_nbWorkers, numThreads);

    size_t cSize = 0;

    for (auto _ : state) {
        // Compress
        cSize = ZSTD_compressCCtx(cctx, cBuff, cBuffSize, fBuff, fSize * sizeof(float), zstdLevel);
        CHECK_ZSTD(cSize);
    }

    // Save the compressed data to a file
    saveFile_orDie("test_file.zst", cBuff, cSize);

    // Load compressed data from file for decompression
    size_t compressedSize;
    void* const loadedCBuff = mallocAndLoadFile_orDie("test_file.zst", &compressedSize);

    // Allocate memory for the decompressed buffer
    float* const decompressedBuff = (float*)malloc_orDie(fSize * sizeof(float));

    // Decompress
    size_t const decompressedSize = ZSTD_decompress(decompressedBuff, fSize * sizeof(float), loadedCBuff, compressedSize);
    CHECK_ZSTD(decompressedSize);

    // Verify if decompressed data matches the original data
    bool isEqual = compareBuffers(fBuff, decompressedBuff, fSize);

    // Calculate and display compression ratio
    double compressionRatio = (double)fSize * sizeof(float) / (double)cSize;

    // Display the results
    printf("\nOriginal Size: %zu bytes\n", fSize * sizeof(float));
    printf("Compressed Size: %zu bytes\n", cSize);
    printf("Compression Ratio: %.2f\n", compressionRatio);
    printf("Decompression Match: %s\n", isEqual ? "Yes" : "No");

    // Free resources
    free(fBuff);
    free(cBuff);
    free(loadedCBuff);
    free(decompressedBuff);
    ZSTD_freeCCtx(cctx);
}

// Register the function as a benchmark
int main(int argc, char** argv) {
    ::benchmark::Initialize(&argc, argv);

    benchmark::RegisterBenchmark("ZSTD_BENCH", ZSTD_BENCH)
        ->Args({1000, 22, 1})->Args({2000000, 22, 1})
        ->Args({1000000, 3, 1})->Args({2000000, 3, 1})
        ->Args({1000000, 22, 1})->Args({2000000, 1, 1})
        ->Args({1000000, 22, 4})->Args({2000000, 22, 4})
        ->Args({1000000, 3, 4})->Args({2000000, 3, 4})
        ->Args({1000000, 22, 4})->Args({2000000, 1, 4});

    ::benchmark::RunSpecifiedBenchmarks();
    return 0;
}
