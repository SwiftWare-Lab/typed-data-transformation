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
void writeResultsToCSV(const std::string& filename, const benchmark::State& state);

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

// Compression ratio calculation
double calculateCompressionRatio(size_t originalSize, size_t compressedSize) {
    return static_cast<double>(originalSize) / static_cast<double>(compressedSize);
}

// Function to write benchmark results to a CSV file
// Write aggregated benchmark results to a CSV file
void writeResultsToCSV(const std::string& filePath, const benchmark::State& state) {
    std::ofstream csvFile(filePath, std::ios::out);

    if (!csvFile.is_open()) {
        std::cerr << "Failed to open CSV file for writing: " << filePath << std::endl;
        return;
    }

    // Write headers
    csvFile << "Metric,Value\n";

    // Helper lambda to write a counter if it exists
    auto writeCounterIfExists = [&csvFile, &state](const std::string& metric) {
        if (state.counters.find(metric) != state.counters.end()) {
            csvFile << metric << "," << state.counters.at(metric) << "\n";
        }
    };

    // Write metrics safely
    writeCounterIfExists("CompRatio_BeforeDecomp");
    writeCounterIfExists("CompTime_BeforeDecomp");
    writeCounterIfExists("DecompTime_BeforeDecomp");
    writeCounterIfExists("Leading_CompRatio");
    writeCounterIfExists("Content_CompRatio");
    writeCounterIfExists("Trailing_CompRatio");
    writeCounterIfExists("Total_CompRatio_AfterDecomp");
    writeCounterIfExists("Leading_CompTime");
    writeCounterIfExists("Content_CompTime");
    writeCounterIfExists("Trailing_CompTime");
    writeCounterIfExists("Leading_DecompTime");
    writeCounterIfExists("Content_DecompTime");
    writeCounterIfExists("Trailing_DecompTime");

    csvFile.close();
    std::cout << "Results saved to: " << filePath << std::endl;
}
#include <fstream>
#include <iomanip>

void writeResultsToCSV3(const std::string& filePath, const benchmark::State& state) {
    std::ofstream csvFile(filePath, std::ios::app);

    if (!csvFile.is_open()) {
        std::cerr << "Failed to open CSV file for writing: " << filePath << std::endl;
        return;
    }

    // Write header if the file is empty
    if (csvFile.tellp() == 0) {
        csvFile << "Benchmark,Iterations,CompRatio_BeforeDecomp,CompTime_BeforeDecomp,Content_CompTime,"
                   "Content_DecompTime,DecompTime_BeforeDecomp,Leading_CompTime,Leading_DecompTime,"
                   "Total_CompRatio_AfterDecomp,Trailing_CompTime,Trailing_DecompTime\n";
    }

    // Write the benchmark results to a single row
    csvFile << "ZSTD_BENCH/1000000/3," << state.iterations();

    // Helper lambda to add a counter value if it exists
    auto writeCounterIfExists = [&csvFile, &state](const std::string& metric) {
        if (state.counters.find(metric) != state.counters.end()) {
            csvFile << "," << state.counters.at(metric);
        } else {
            csvFile << ",0";  // Add 0 if the metric doesn't exist
        }
    };

    // Append metrics to the row
    writeCounterIfExists("CompRatio_BeforeDecomp");
    writeCounterIfExists("CompTime_BeforeDecomp");
    writeCounterIfExists("Content_CompTime");
    writeCounterIfExists("Content_DecompTime");
    writeCounterIfExists("DecompTime_BeforeDecomp");
    writeCounterIfExists("Leading_CompTime");
    writeCounterIfExists("Leading_DecompTime");
    writeCounterIfExists("Total_CompRatio_AfterDecomp");
    writeCounterIfExists("Trailing_CompTime");
    writeCounterIfExists("Trailing_DecompTime");

    csvFile << "\n";  // Newline for the next entry
    csvFile.close();

    std::cout << "Complete results saved to: " << filePath << std::endl;
}


void writeResultsToCSV2(const std::string& filename, const benchmark::State& state) {
    std::ofstream csvFile(filename, std::ios::app);

    if (!csvFile.is_open()) {
        std::cerr << "Failed to open the CSV file for writing." << std::endl;
        return;
    }

    csvFile << "Metric,Value\n";
    for (const auto& counter : state.counters) {
        csvFile << counter.first << "," << counter.second << "\n";
    }

    csvFile.close();
}

// Benchmark function for compression and decompression
static void ZSTD_BENCH(benchmark::State& state) {
    int zstdLevel = state.range(1);


    // Measure compression ratio and time before decomposition
    std::vector<uint8_t> compressedData;
    auto start = std::chrono::high_resolution_clock::now();
    size_t compressedSize = compressWithZstd(globalByteArray, compressedData, zstdLevel);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> compressionTimeBeforeDecomp = end - start;

    // Decompress the entire byte array
    std::vector<uint8_t> decompressedData;
    start = std::chrono::high_resolution_clock::now();
    decompressWithZstd(compressedData, decompressedData, globalByteArray.size());
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> decompressionTimeBeforeDecomp = end - start;

    // Calculate compression ratio before decomposition
    double compRatioBeforeDecomp = calculateCompressionRatio(globalByteArray.size(), compressedSize);

    // Split global byte array into components
    std::vector<uint8_t> leading, content, trailing;
    splitBytesIntoComponents(globalByteArray, leading, content, trailing);

    std::vector<uint8_t> compressedLeading, compressedContent, compressedTrailing;
    std::vector<uint8_t> decompressedLeading, decompressedContent, decompressedTrailing;

    size_t leadingCompressedSize = 0, contentCompressedSize = 0, trailingCompressedSize = 0;

    for (auto _ : state) {
        // Measure compression and decompression time for each component
        start = std::chrono::high_resolution_clock::now();
        leadingCompressedSize = compressWithZstd(leading, compressedLeading, zstdLevel);
        end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> leadingCompTime = end - start;

        start = std::chrono::high_resolution_clock::now();
        contentCompressedSize = compressWithZstd(content, compressedContent, zstdLevel);
        end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> contentCompTime = end - start;

        start = std::chrono::high_resolution_clock::now();
        trailingCompressedSize = compressWithZstd(trailing, compressedTrailing, zstdLevel);
        end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> trailingCompTime = end - start;

        // Decompress each component
        start = std::chrono::high_resolution_clock::now();
        decompressWithZstd(compressedLeading, decompressedLeading, leading.size());
        end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> leadingDecompTime = end - start;

        start = std::chrono::high_resolution_clock::now();
        decompressWithZstd(compressedContent, decompressedContent, content.size());
        end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> contentDecompTime = end - start;

        start = std::chrono::high_resolution_clock::now();
        decompressWithZstd(compressedTrailing, decompressedTrailing, trailing.size());
        end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> trailingDecompTime = end - start;

        // Calculate total compression ratio after decomposition
        size_t totalOriginalSize = leading.size() + content.size() + trailing.size();
        size_t totalCompressedSize = leadingCompressedSize + contentCompressedSize + trailingCompressedSize;
        double totalCompRatioAfterDecomp = calculateCompressionRatio(totalOriginalSize, totalCompressedSize);

        // Compression ratios for each component
        double leadingCompRatio = calculateCompressionRatio(leading.size(), leadingCompressedSize);
        double contentCompRatio = calculateCompressionRatio(content.size(), contentCompressedSize);
        double trailingCompRatio = calculateCompressionRatio(trailing.size(), trailingCompressedSize);


        // Set benchmark counters
        state.counters["CompRatio_BeforeDecomp"] = compRatioBeforeDecomp;
        state.counters["CompTime_BeforeDecomp"] = compressionTimeBeforeDecomp.count();
        state.counters["DecompTime_BeforeDecomp"] = decompressionTimeBeforeDecomp.count();
        //state.counters["Leading_CompRatio"] = leadingCompRatio;
       // state.counters["Content_CompRatio"] = contentCompRatio;
       // state.counters["Trailing_CompRatio"] = trailingCompRatio;
        state.counters["Total_CompRatio_AfterDecomp"] = totalCompRatioAfterDecomp;
        state.counters["Leading_CompTime"] = leadingCompTime.count();
        state.counters["Content_CompTime"] = contentCompTime.count();
        state.counters["Trailing_CompTime"] = trailingCompTime.count();
        state.counters["Leading_DecompTime"] = leadingDecompTime.count();
        state.counters["Content_DecompTime"] = contentDecompTime.count();
        state.counters["Trailing_DecompTime"] = trailingDecompTime.count();

        // Write results to CSV after each iteration
        writeResultsToCSV("/home/jamalids/Documents/compression-part4/new/big-data-compression/benchmark_results1.csv", state);
        // Write aggregated results to CSV after benchmarking
        writeResultsToCSV3("/home/jamalids/Documents/compression-part4/new/big-data-compression/benchmark_results3.csv", state);
    }
}

#include <fstream>
#include <string>
#include <benchmark/benchmark.h>

// Function to write benchmark results to a CSV file
void writeResultsToCSV1(const std::string& filePath, const benchmark::State& state) {
    std::ofstream csvFile(filePath, std::ios::app);  // Open the file in append mode

    if (!csvFile.is_open()) {
        std::cerr << "Failed to open CSV file at " << filePath << std::endl;
        return;
    }

    // Write the results to the CSV file
    csvFile << "CompRatio_BeforeDecomp," << state.counters.at("CompRatio_BeforeDecomp") << "\n";
    csvFile << "CompTime_BeforeDecomp," << state.counters.at("CompTime_BeforeDecomp") << "\n";
    csvFile << "DecompTime_BeforeDecomp," << state.counters.at("DecompTime_BeforeDecomp") << "\n";
    csvFile << "Leading_CompRatio," << state.counters.at("Leading_CompRatio") << "\n";
    csvFile << "Content_CompRatio," << state.counters.at("Content_CompRatio") << "\n";
    csvFile << "Trailing_CompRatio," << state.counters.at("Trailing_CompRatio") << "\n";
    csvFile << "Total_CompRatio_AfterDecomp," << state.counters.at("Total_CompRatio_AfterDecomp") << "\n";
    csvFile << "Leading_CompTime," << state.counters.at("Leading_CompTime") << "\n";
    csvFile << "Content_CompTime," << state.counters.at("Content_CompTime") << "\n";
    csvFile << "Trailing_CompTime," << state.counters.at("Trailing_CompTime") << "\n";
    csvFile << "Leading_DecompTime," << state.counters.at("Leading_DecompTime") << "\n";
    csvFile << "Content_DecompTime," << state.counters.at("Content_DecompTime") << "\n";
    csvFile << "Trailing_DecompTime," << state.counters.at("Trailing_DecompTime") << "\n";

    csvFile.close();
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
        ->Args({1000000, 3});   // Example arguments


    // Run the benchmark
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    //writeResultsToCSV("/home/jamalids/Documents/compression-part4/new/big-data-compression/benchmark_results.csv", state);


    return 0;
}
