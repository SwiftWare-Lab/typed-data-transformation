//
// Created by jamalids on 04/11/24.
//

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

#include "zstd_parallel.h"
std::vector<uint8_t> globalByteArray;


std::pair<std::vector<float>, size_t> loadTSVDataset(const std::string& filePath) {
  std::vector<float> floatArray;
  std::ifstream file(filePath);
  std::string line;
  size_t rowCount = 0;

  if (file.is_open()) {
    while (std::getline(file, line)) {
      std::stringstream ss(line);
      std::string value;

      // Skip the first column value
      std::getline(ss, value, '\t');

      // Read the rest of the line and convert to floats
      while (std::getline(ss, value, '\t')) {
        floatArray.push_back(std::stof(value));
      }
      rowCount++;
    }
    file.close();
  } else {
    std::cerr << "Unable to open file: " << filePath << std::endl;
  }

  return {floatArray, rowCount};
}

//double64
std::pair<std::vector<double>, size_t> loadTSVDatasetdouble(const std::string& filePath) {
  std::vector<double> doubleArray;  // Use double for 64-bit floating-point
  std::ifstream file(filePath);
  std::string line;
  size_t rowCount = 0;

  if (file.is_open()) {
    while (std::getline(file, line)) {
      std::stringstream ss(line);
      std::string value;

      // Skip the first column value
      std::getline(ss, value, '\t');

      // Read the rest of the line and convert to doubles
      while (std::getline(ss, value, '\t')) {
        doubleArray.push_back(std::stod(value));  // Convert to double
      }
      rowCount++;
    }
    file.close();
  } else {
    std::cerr << "Unable to open file: " << filePath << std::endl;
  }

  return {doubleArray, rowCount};
}

// Function to load a TSV dataset and convert to bfloat16 (uint16_t)
std::pair<std::vector<uint16_t>, size_t> loadTSVDatasetAsBFloat16(const std::string& filePath) {
  std::vector<uint16_t> bfloat16Array;
  std::ifstream file(filePath);
  std::string line;
  size_t rowCount = 0;

  if (file.is_open()) {
    while (std::getline(file, line)) {
      std::stringstream ss(line);
      std::string value;

      // Skip the first column value
      std::getline(ss, value, '\t');

      // Read the rest of the line and convert to bfloat16
      while (std::getline(ss, value, '\t')) {
        float floatValue = std::stof(value);

        // Convert 32-bit float to bfloat16 (store the most significant 16 bits)
        uint16_t bfloat16Value = *reinterpret_cast<uint32_t*>(&floatValue) >> 16;

        bfloat16Array.push_back(bfloat16Value);
      }
      rowCount++;
    }
    file.close();
  } else {
    std::cerr << "Unable to open file: " << filePath << std::endl;
  }

  return {bfloat16Array, rowCount};
}
std::pair<std::vector<int16_t>, size_t> loadTSVDatasetAsInt16(const std::string& filePath) {
  std::vector<int16_t> int16Array;
  std::ifstream file(filePath);
  std::string line;
  size_t rowCount = 0;

  if (file.is_open()) {
    while (std::getline(file, line)) {
      std::stringstream ss(line);
      std::string value;

      // Skip the first column value
      //std::getline(ss, value, '\t');

      // Read the rest of the line and convert to int16_t
      while (std::getline(ss, value, '\t')) {
        try {
          // Convert string to float, then cast to int16_t
          float floatValue = std::stof(value);
          int16_t int16Value = static_cast<int16_t>(floatValue);

          // Store the converted value
          int16Array.push_back(int16Value);
        } catch (const std::exception& e) {
          std::cerr << "Error converting value: " << value << " - " << e.what() << std::endl;
        }
      }
      rowCount++;
    }
    file.close();
  } else {
    std::cerr << "Unable to open file: " << filePath << std::endl;
  }

  return {int16Array, rowCount};
}
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
std::vector<uint8_t> convertDoubleToBytes(const std::vector<double>& doubleArray) {
  std::vector<uint8_t> byteArray(doubleArray.size() * 8);  // Each double is 8 bytes
  for (size_t i = 0; i < doubleArray.size(); i++) {
    // Get a pointer to the bytes of the double
    uint8_t* doubleBytes = reinterpret_cast<uint8_t*>(const_cast<double*>(&doubleArray[i]));
    for (size_t j = 0; j < 8; j++) {
      // Copy each byte of the double into the correct position in the byteArray
      byteArray[i * 8 + j] = doubleBytes[j];
    }
  }
  return byteArray;
}

std::vector<uint8_t> convertInt16ToBytes(const std::vector<int16_t>& int16Array) {
  std::vector<uint8_t> byteArray(int16Array.size() * 2); // Each int16_t needs 2 bytes
  for (size_t i = 0; i < int16Array.size(); ++i) {
    int16_t value = int16Array[i];
    // Extract low and high bytes
    byteArray[i * 2] = static_cast<uint8_t>(value & 0xFF);        // Low byte
    byteArray[i * 2 + 1] = static_cast<uint8_t>((value >> 8));    // High byte
  }
  return byteArray;
}

std::pair<double, double> calculateCompDecomThroughput(size_t originalSize, double compressedTime, double decompressedTime) {
  // Convert originalSize from bytes to gigabytes
  double originalSizeGB = static_cast<double>(originalSize) / 1e9;

  // Calculate throughput in GB/s
  double compressionThroughput = originalSizeGB / static_cast<double>(compressedTime);
  double decompressionThroughput = originalSizeGB / static_cast<double>(decompressedTime);

  return {compressionThroughput, decompressionThroughput};
}
int main(int argc, char* argv[]) {
    cxxopts::Options options("DataCompressor", "Compress datasets and profile the compression");
    options.add_options()
        ("d,dataset", "Path to the dataset file", cxxopts::value<std::string>())
        ("o,outcsv", "Output CSV file path", cxxopts::value<std::string>())
        ("t,threads", "Number of threads to use", cxxopts::value<int>()->default_value("10"))
        ("h,help", "Print help");

    auto result = options.parse(argc, argv);

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    std::string datasetPath = result["dataset"].as<std::string>();
    std::string outputCSV = result["outcsv"].as<std::string>();
    int numThreads = result["threads"].as<int>();

    auto [floatArray, rowCount] = loadTSVDatasetdouble(datasetPath);
    std::cout << "Loaded " << rowCount << " rows with " << floatArray.size() << " total values." << std::endl;
    if (floatArray.empty()) {
        std::cerr << "Failed to load dataset from " << datasetPath << std::endl;
        return 1;
    }

    globalByteArray = convertDoubleToBytes(floatArray);

    // Define multiple configurations for component sizes
  std::vector<std::vector<size_t>> componentSizesList = {
    {7, 1, 0, 0, 0, 0, 0, 0},  // 7 bytes + 1 byte
    {6, 2, 0, 0, 0, 0, 0, 0},  // 6 bytes + 2 bytes
    {6, 1, 1, 0, 0, 0, 0, 0},  // 6 bytes + 1 byte + 1 byte
    {5, 3, 0, 0, 0, 0, 0, 0},  // 5 bytes + 3 bytes
    {5, 2, 1, 0, 0, 0, 0, 0},  // 5 bytes + 2 bytes + 1 byte
    {5, 1, 1, 1, 0, 0, 0, 0},  // 5 bytes + 1 byte + 1 byte + 1 byte
    {4, 4, 0, 0, 0, 0, 0, 0},  // 4 bytes + 4 bytes
    {4, 3, 1, 0, 0, 0, 0, 0},  // 4 bytes + 3 bytes + 1 byte
    {4, 2, 2, 0, 0, 0, 0, 0},  // 4 bytes + 2 bytes + 2 bytes
    {4, 2, 1, 1, 0, 0, 0, 0},  // 4 bytes + 2 bytes + 1 byte + 1 byte
    {4, 1, 1, 1, 1, 0, 0, 0},  // 4 bytes + 1 byte + 1 byte + 1 byte + 1 byte
    {3, 3, 2, 0, 0, 0, 0, 0},  // 3 bytes + 3 bytes + 2 bytes
    {3, 3, 1, 1, 0, 0, 0, 0},  // 3 bytes + 3 bytes + 1 byte + 1 byte
    {3, 2, 2, 1, 0, 0, 0, 0},  // 3 bytes + 2 bytes + 2 bytes + 1 byte
    {3, 2, 1, 1, 1, 0, 0, 0},  // 3 bytes + 2 bytes + 1 byte + 1 byte + 1 byte
    {3, 1, 1, 1, 1, 1, 0, 0},  // 3 bytes + 1 byte + 1 byte + 1 byte + 1 byte + 1 byte
    {2, 2, 2, 2, 0, 0, 0, 0},  // 2 bytes + 2 bytes + 2 bytes + 2 bytes
    {2, 2, 2, 1, 1, 0, 0, 0},  // 2 bytes + 2 bytes + 2 bytes + 1 byte + 1 byte
    {2, 2, 1, 1, 1, 1, 0, 0},  // 2 bytes + 2 bytes + 1 byte + 1 byte + 1 byte + 1 byte
    {2, 1, 1, 1, 1, 1, 1, 0},  // 2 bytes + 1 byte + 1 byte + 1 byte + 1 byte + 1 byte + 1 byte
    {1, 1, 1, 1, 1, 1, 1, 1},  // 8 groups of 1 byte
    {1, 5, 1, 1, 0, 0, 0, 0},  // 1 byte + 5 bytes + 1 byte + 1 byte
    {1, 4, 2, 1, 0, 0, 0, 0},  // 1 byte + 4 bytes + 2 bytes + 1 byte
    {2, 3, 2, 1, 0, 0, 0, 0}   // 2 bytes + 3 bytes + 2 bytes + 1 byte
  };

    std::vector<ProfilingInfo> pi_array;
for (const auto& componentSizes : componentSizesList) {
    std::cout << "Testing with component sizes: ";
    for (auto size : componentSizes) std::cout << size << " ";
    std::cout << std::endl;

    // Temporary storage for compressed components
    std::vector<std::vector<uint8_t>> compressedComponents(componentSizes.size());

    // Loop over each run type (Full, Sequential, Parallel)
    for (int i = 0; i < 1; ++i) { // Set `num_iter` for multiple runs if needed

        // --- Full Compression ---
        ProfilingInfo pi_full(componentSizes.size());  // Initialize profiling info
        std::vector<uint8_t> compressedData, decompressedData; // Reset temporary storage

        auto start = std::chrono::high_resolution_clock::now();
        double compressedSize = zstdCompression(globalByteArray, pi_full, compressedData);
        auto end = std::chrono::high_resolution_clock::now();
        pi_full.total_time_compressed = std::chrono::duration<double>(end - start).count();

        start = std::chrono::high_resolution_clock::now();
        zstdDecompression(compressedData, decompressedData, pi_full);
        end = std::chrono::high_resolution_clock::now();
        pi_full.total_time_decompressed = std::chrono::duration<double>(end - start).count();
        pi_full.com_ratio = calculateCompressionRatio(globalByteArray.size(), compressedSize);
        pi_full.total_values = rowCount;

        pi_full.type = "Full"; // Set type for CSV output
        pi_array.push_back(pi_full);

        // --- Sequential Compression ---
        ProfilingInfo pi_seq(componentSizes.size());  // Initialize profiling info
        compressedComponents.clear();                // Reset components for this run
        compressedComponents.resize(componentSizes.size()); // Reinitialize

        start = std::chrono::high_resolution_clock::now();
        compressedSize = zstdDecomposedSequential(globalByteArray, pi_seq, compressedComponents, componentSizes);
        end = std::chrono::high_resolution_clock::now();
        pi_seq.total_time_compressed = std::chrono::duration<double>(end - start).count();

        start = std::chrono::high_resolution_clock::now();
        zstdDecomposedSequentialDecompression(compressedComponents, pi_seq, componentSizes);
        end = std::chrono::high_resolution_clock::now();
        pi_seq.total_time_decompressed = std::chrono::duration<double>(end - start).count();
        pi_seq.com_ratio = calculateCompressionRatio(globalByteArray.size(), compressedSize);
        pi_seq.total_values = rowCount;

        pi_seq.type = "Sequential"; // Set type for CSV output
        pi_array.push_back(pi_seq);

        // --- Parallel Compression ---
        ProfilingInfo pi_parallel(componentSizes.size()); // Initialize profiling info
        compressedComponents.clear();                    // Reset components for this run
        compressedComponents.resize(componentSizes.size()); // Reinitialize

        start = std::chrono::high_resolution_clock::now();
        compressedSize = zstdDecomposedParallel(globalByteArray, pi_parallel, compressedComponents, componentSizes, numThreads);
        end = std::chrono::high_resolution_clock::now();
        pi_parallel.total_time_compressed = std::chrono::duration<double>(end - start).count();

        start = std::chrono::high_resolution_clock::now();
        zstdDecomposedParallelDecompression(compressedComponents, pi_parallel, componentSizes, numThreads);
        end = std::chrono::high_resolution_clock::now();
        pi_parallel.total_time_decompressed = std::chrono::duration<double>(end - start).count();
        pi_parallel.com_ratio = calculateCompressionRatio(globalByteArray.size(), compressedSize);
        pi_parallel.total_values = rowCount;

        pi_parallel.type = "Parallel"; // Set type for CSV output
        pi_array.push_back(pi_parallel);
    }
}

  // Write results to CSV
  std::ofstream file(outputCSV);
  if (!file) {
    std::cerr << "Failed to open the file for writing: " << outputCSV << std::endl;
    return 1;
  }

  // Write the CSV header
  file << "Iteration,ComponentSizes,Type,CompressionRatio,TotalTimeCompressed,TotalTimeDecompressed,"
       << "Component1Time,Component2Time,Component3Time,Component4Time,Component5Time,"
       << "Component6Time,Component7Time,Component8Time,CompressionThroughput,DecompressionThroughput,TotalValues\n";

  size_t configIndex = 0; // Track current configuration index
  size_t iteration = 1;

  for (size_t i = 0; i < pi_array.size(); ++i) {
    if (i % 3 == 0 && i > 0) {
      // Move to the next configuration after every 3 entries (Full, Sequential, Parallel)
      configIndex++;
    }

    // Write iteration and component sizes
    file << iteration++ << ",";
    for (size_t size : componentSizesList[configIndex]) {
      file << size << " ";
    }
    file << ",";

    // Write profiling info
    pi_array[i].printCSV(file, iteration);
  }


    return 0;
}
