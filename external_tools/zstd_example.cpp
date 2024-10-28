#include <benchmark/benchmark.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <zstd.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include "common.h"

// Function to load a TSV dataset into a float vector
std::vector<float> loadTSVDataset(const std::string& filePath, size_t maxRows = 4000000) {
    std::vector<float> floatArray;
    std::ifstream file(filePath);
    std::string line;

    // Skip the header if present
    if (file.is_open()) {
        size_t rowCount = 0;
        while (std::getline(file, line) && rowCount < maxRows) {
            std::stringstream ss(line);
            std::string value;
            // Skip the first column
            std::getline(ss, value, '\t');

            // Read the rest of the row
            while (std::getline(ss, value, '\t')) {
                floatArray.push_back(std::stof(value));
            }
            rowCount++;
        }
        file.close();
    } else {
        std::cerr << "Unable to open file: " << filePath << std::endl;
    }

    return floatArray;
}

// Function to convert a float array to a uint8_t vector
std::vector<uint8_t> convertFloatToBytes(const std::vector<float>& floatArray) {
    std::vector<uint8_t> byteArray(floatArray.size() * 4);

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
        leading.push_back(byteArray[i * 4]);
        content.push_back(byteArray[i * 4 + 1]);
        content.push_back(byteArray[i * 4 + 2]);
        trailing.push_back(byteArray[i * 4 + 3]);
    }
}

// Compress a byte vector with Zstd and return compressed size
size_t compressWithZstd(const std::vector<uint8_t>& data, std::vector<uint8_t>& compressedData, int compressionLevel) {
    size_t const cBuffSize = ZSTD_compressBound(data.size());
    compressedData.resize(cBuffSize);

    size_t const cSize = ZSTD_compress(compressedData.data(), cBuffSize, data.data(), data.size(), compressionLevel);
    CHECK_ZSTD(cSize);

    compressedData.resize(cSize);
    return cSize;
}

// Decompress a byte vector with Zstd and return decompressed size
size_t decompressWithZstd(const std::vector<uint8_t>& compressedData, std::vector<uint8_t>& decompressedData, size_t originalSize) {
    decompressedData.resize(originalSize);

    size_t const dSize = ZSTD_decompress(decompressedData.data(), originalSize, compressedData.data(), compressedData.size());
    CHECK_ZSTD(dSize);

    return dSize;
}

// Verify if arrays are equal
bool verifyBytes(const std::vector<uint8_t>& original, const std::vector<uint8_t>& reconstructed) {
    if (original.size() != reconstructed.size()) {
        return false;
    }

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

// Benchmark for loading the dataset
static void BM_LoadTSVDataset(benchmark::State& state) {
    std::string datasetPath = "/home/jamalids/Documents/2D/data1/Fcbench/HPC/H/num_brain_f64.tsv";
    for (auto _ : state) {
        std::vector<float> floatArray = loadTSVDataset(datasetPath, state.range(0));
        benchmark::DoNotOptimize(floatArray);
    }
}
BENCHMARK(BM_LoadTSVDataset)->Arg(4000000);

// Benchmark for converting floats to bytes
static void BM_ConvertFloatToBytes(benchmark::State& state) {
    std::string datasetPath = "/home/jamalids/Documents/2D/data1/Fcbench/HPC/H/num_brain_f64.tsv";
    std::vector<float> floatArray = loadTSVDataset(datasetPath);

    for (auto _ : state) {
        std::vector<uint8_t> byteArray = convertFloatToBytes(floatArray);
        benchmark::DoNotOptimize(byteArray);
    }
}
BENCHMARK(BM_ConvertFloatToBytes);

// Benchmark for compression with Zstd
static void BM_CompressWithZstd(benchmark::State& state) {
    std::string datasetPath = "/home/jamalids/Documents/2D/data1/Fcbench/HPC/H/num_brain_f64.tsv";
    std::vector<float> floatArray = loadTSVDataset(datasetPath);
    std::vector<uint8_t> byteArray = convertFloatToBytes(floatArray);
    std::vector<uint8_t> compressedData;

    for (auto _ : state) {
        size_t compressedSize = compressWithZstd(byteArray, compressedData, 3);
        benchmark::DoNotOptimize(compressedSize);
    }
}
BENCHMARK(BM_CompressWithZstd);

// Benchmark for decompression with Zstd
static void BM_DecompressWithZstd(benchmark::State& state) {
    std::string datasetPath = "/home/jamalids/Documents/2D/data1/Fcbench/HPC/H/num_brain_f64.tsv";
    std::vector<float> floatArray = loadTSVDataset(datasetPath);
    std::vector<uint8_t> byteArray = convertFloatToBytes(floatArray);
    std::vector<uint8_t> compressedData;
    size_t compressedSize = compressWithZstd(byteArray, compressedData, 3);
    std::vector<uint8_t> decompressedData;

    for (auto _ : state) {
        size_t decompressedSize = decompressWithZstd(compressedData, decompressedData, byteArray.size());
        benchmark::DoNotOptimize(decompressedSize);
    }
}
BENCHMARK(BM_DecompressWithZstd);

// Benchmark for splitting bytes into components
static void BM_SplitBytesIntoComponents(benchmark::State& state) {
    std::string datasetPath = "/home/jamalids/Documents/2D/data1/Fcbench/HPC/H/num_brain_f64.tsv";
    std::vector<float> floatArray = loadTSVDataset(datasetPath);
    std::vector<uint8_t> byteArray = convertFloatToBytes(floatArray);
    std::vector<uint8_t> leading, content, trailing;

    for (auto _ : state) {
        splitBytesIntoComponents(byteArray, leading, content, trailing);
        benchmark::DoNotOptimize(leading);
        benchmark::DoNotOptimize(content);
        benchmark::DoNotOptimize(trailing);
    }
}
BENCHMARK(BM_SplitBytesIntoComponents);

// Benchmark for verifying bytes
static void BM_VerifyBytes(benchmark::State& state) {
    std::string datasetPath = "/home/jamalids/Documents/2D/data1/Fcbench/HPC/H/num_brain_f64.tsv";
    std::vector<float> floatArray = loadTSVDataset(datasetPath);
    std::vector<uint8_t> byteArray = convertFloatToBytes(floatArray);
    std::vector<uint8_t> decompressedData = byteArray; // Use the same for verification

    for (auto _ : state) {
        bool result = verifyBytes(byteArray, decompressedData);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_VerifyBytes);

// Main function for Google Benchmark
BENCHMARK_MAIN();
