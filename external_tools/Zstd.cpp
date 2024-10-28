#include <benchmark/benchmark.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <zstd.h>
#include <chrono>
#include <cstdint>
#include "common.h"

// Global variable to hold the dataset
std::vector<uint8_t> globalByteArray;

// Function declarations
std::vector<float> loadTSVDataset(const std::string& filePath, size_t maxRows = 4000000);
std::vector<uint8_t> convertFloatToBytes(const std::vector<float>& floatArray);
void splitBytesIntoComponents(const std::vector<uint8_t>& byteArray, std::vector<uint8_t>& leading,
                              std::vector<uint8_t>& content, std::vector<uint8_t>& trailing);
size_t compressWithZstd(const std::vector<uint8_t>& data, std::vector<uint8_t>& compressedData, int compressionLevel);
size_t decompressWithZstd(const std::vector<uint8_t>& compressedData, std::vector<uint8_t>& decompressedData, size_t originalSize);
double calculateCompressionRatio(size_t originalSize, size_t compressedSize);

// Function to load a TSV dataset into a float vector
std::vector<float> loadTSVDataset(const std::string& filePath, size_t maxRows) {
    std::vector<float> floatArray;
    std::ifstream file(filePath);
    std::string line;

    if (file.is_open()) {
        size_t rowCount = 0;
        while (std::getline(file, line) && rowCount < maxRows) {
            std::stringstream ss(line);
            std::string value;
            std::getline(ss, value, '\t'); // Skip the first column

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
    if (ZSTD_isError(cSize)) {
        std::cerr << "Zstd compression error: " << ZSTD_getErrorName(cSize) << std::endl;
        return 0;
    }

    compressedData.resize(cSize);
    return cSize;
}

// Decompress a byte vector with Zstd and return decompressed size
size_t decompressWithZstd(const std::vector<uint8_t>& compressedData, std::vector<uint8_t>& decompressedData, size_t originalSize) {
    decompressedData.resize(originalSize);

    size_t const dSize = ZSTD_decompress(decompressedData.data(), originalSize, compressedData.data(), compressedData.size());
    if (ZSTD_isError(dSize)) {
        std::cerr << "Zstd decompression error: " << ZSTD_getErrorName(dSize) << std::endl;
        return 0;
    }

    return dSize;
}

// Compression ratio calculation
double calculateCompressionRatio(size_t originalSize, size_t compressedSize) {
    return static_cast<double>(originalSize) / static_cast<double>(compressedSize);
}

// Reconstruct data from components
void reconstructFromComponents(const std::vector<uint8_t>& leading, const std::vector<uint8_t>& content,
                               const std::vector<uint8_t>& trailing, std::vector<uint8_t>& reconstructedData) {
    size_t floatCount = leading.size();
    reconstructedData.resize(floatCount * 4);

    for (size_t i = 0; i < floatCount; i++) {
        reconstructedData[i * 4] = leading[i];
        reconstructedData[i * 4 + 1] = content[i * 2];
        reconstructedData[i * 4 + 2] = content[i * 2 + 1];
        reconstructedData[i * 4 + 3] = trailing[i];
    }
}

// Benchmark function for compression, decompression, and reconstruction
static void ZSTD_BENCH(benchmark::State& state) {
    int zstdLevel = state.range(1);

    // Split global byte array into components
    std::vector<uint8_t> leading, content, trailing;
    splitBytesIntoComponents(globalByteArray, leading, content, trailing);

    std::vector<uint8_t> compressedLeading, compressedContent, compressedTrailing;
    std::vector<uint8_t> decompressedLeading(leading.size()), decompressedContent(content.size()), decompressedTrailing(trailing.size());
    std::vector<uint8_t> reconstructedData;

    // Variables for time measurements
    std::chrono::duration<double> leadingCompTime, contentCompTime, trailingCompTime;
    std::chrono::duration<double> leadingDecompTime, contentDecompTime, trailingDecompTime;
    std::chrono::duration<double> reconstructionTime;

    // Measure compression ratio and time before decomposition
    std::vector<uint8_t> compressedData;
    auto start = std::chrono::high_resolution_clock::now();
    size_t compressedSize = compressWithZstd(globalByteArray, compressedData, zstdLevel);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> compressionTimeBeforeDecomp = end - start;

    double compRatioBeforeDecomp = calculateCompressionRatio(globalByteArray.size(), compressedSize);

    // Decompression before decomposition
    std::vector<uint8_t> decompressedData(globalByteArray.size());
    start = std::chrono::high_resolution_clock::now();
    decompressWithZstd(compressedData, decompressedData, globalByteArray.size());
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> decompressionTimeBeforeDecomp = end - start;
    double totalCompRatioAfterDecomp = 0.0; // Declare outside the loop
    // Measure compression and decompression for components
    for (auto _ : state) {
        // Compression for each component
        start = std::chrono::high_resolution_clock::now();
        size_t leadingCompressedSize = compressWithZstd(leading, compressedLeading, zstdLevel);
        end = std::chrono::high_resolution_clock::now();
        leadingCompTime = end - start;

        start = std::chrono::high_resolution_clock::now();
        size_t contentCompressedSize = compressWithZstd(content, compressedContent, zstdLevel);
        end = std::chrono::high_resolution_clock::now();
        contentCompTime = end - start;

        start = std::chrono::high_resolution_clock::now();
        size_t trailingCompressedSize = compressWithZstd(trailing, compressedTrailing, zstdLevel);
        end = std::chrono::high_resolution_clock::now();
        trailingCompTime = end - start;

        // Decompression for each component
        start = std::chrono::high_resolution_clock::now();
        decompressWithZstd(compressedLeading, decompressedLeading, leading.size());
        end = std::chrono::high_resolution_clock::now();
        leadingDecompTime = end - start;

        start = std::chrono::high_resolution_clock::now();
        decompressWithZstd(compressedContent, decompressedContent, content.size());
        end = std::chrono::high_resolution_clock::now();
        contentDecompTime = end - start;

        start = std::chrono::high_resolution_clock::now();
        decompressWithZstd(compressedTrailing, decompressedTrailing, trailing.size());
        end = std::chrono::high_resolution_clock::now();
        trailingDecompTime = end - start;

        // Reconstruction
        start = std::chrono::high_resolution_clock::now();
        reconstructFromComponents(decompressedLeading, decompressedContent, decompressedTrailing, reconstructedData);
        end = std::chrono::high_resolution_clock::now();
        reconstructionTime = end - start;

        // Verify if the reconstructed data matches the original data
        if (reconstructedData != globalByteArray) {
            state.SkipWithError("Reconstructed data doesn't match the original.");
        }

        // Calculate total compression ratio after decomposition
        size_t totalOriginalSize = leading.size() + content.size() + trailing.size();
        size_t totalCompressedSize = leadingCompressedSize + contentCompressedSize + trailingCompressedSize;
        totalCompRatioAfterDecomp = calculateCompressionRatio(totalOriginalSize, totalCompressedSize);




    }
    // Set benchmark counters
    state.counters["CompRatio_BeforeDecomposion"] = compRatioBeforeDecomp;
    state.counters["CompressedTime_BeforeDecomposion"] = compressionTimeBeforeDecomp.count();
    state.counters["DecompressedTime_BeforeDecomposion"] = decompressionTimeBeforeDecomp.count();
    state.counters["Total_CompRatio_AfterDecomposion"] = totalCompRatioAfterDecomp;
    state.counters["Trailing_CompressedTime"] = trailingCompTime.count();
    state.counters["Trailing_DecompressedTime"] = trailingDecompTime.count();
    state.counters["leading_CompressedTime"] = leadingCompTime.count();
    state.counters["leading_DecompressedTime"] = leadingDecompTime.count();
    state.counters["content_CompressedTime"] = contentCompTime.count();
    state.counters["content_DecompressedTime"] = contentDecompTime.count();
    // Add reconstruction time counter
    state.counters["ReconstructionTime"] = reconstructionTime.count();

}

int main(int argc, char** argv) {
    // Load the dataset before running benchmarks
    std::string datasetPath = "/home/jamalids/Documents/2D/data1/Fcbench/HPC/H/num_brain_f64.tsv";
    std::vector<float> floatArray = loadTSVDataset(datasetPath);

    if (floatArray.empty()) {
        std::cerr << "Failed to load dataset from " << datasetPath << std::endl;
        return 1;
    }

    // Convert float array to byte array
    globalByteArray = convertFloatToBytes(floatArray);

    // Register the benchmark
    benchmark::RegisterBenchmark("ZSTD_BENCH", ZSTD_BENCH)
        ->Args({1000000, 3})
        ->UseRealTime()
        ->Iterations(1000);

    // Initialize Google Benchmark
    benchmark::Initialize(&argc, argv);

    // Run the benchmarks
    benchmark::RunSpecifiedBenchmarks();

    return 0;
}