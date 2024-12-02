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
#include <cmath>
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




std::vector<float> convertBytesToFloat(const std::vector<uint8_t>& byteArray) {
  if (byteArray.size() % 4 != 0) {
    throw std::runtime_error("Byte array size is not a multiple of 4.");
  }

  std::vector<float> floatArray(byteArray.size() / 4);

  for (size_t i = 0; i < floatArray.size(); i++) {
    const uint8_t* bytePtr = &byteArray[i * 4];
    float* floatPtr = reinterpret_cast<float*>(const_cast<uint8_t*>(bytePtr));
    floatArray[i] = *floatPtr;
  }

  return floatArray;
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

std::vector<double> convertBytesToDouble(const std::vector<uint8_t>& byteArray) {
  // Each double is 8 bytes, so the size of the byteArray must be a multiple of 8
  if (byteArray.size() % 8 != 0) {
    throw std::runtime_error("Byte array size is not a multiple of 8.");
  }

  // Prepare a vector of doubles to hold the result
  std::vector<double> doubleArray(byteArray.size() / 8);

  // Iterate through the byteArray in chunks of 8 bytes
  for (size_t i = 0; i < doubleArray.size(); i++) {
    // Get a pointer to the bytes for this double
    const uint8_t* bytePtr = &byteArray[i * 8];

    // Reinterpret the bytes as a double and assign to the doubleArray
    const double* doublePtr = reinterpret_cast<const double*>(bytePtr);
    doubleArray[i] = *doublePtr;
  }

  return doubleArray;
}

std::pair<double, double> calculateCompDecomThroughput(size_t originalSize, double compressedTime, double decompressedTime) {
  // Convert originalSize from bytes to gigabytes
  double originalSizeGB = static_cast<double>(originalSize) / 1e9;

  // Calculate throughput in GB/s
  double compressionThroughput = originalSizeGB / static_cast<double>(compressedTime);
  double decompressionThroughput = originalSizeGB / static_cast<double>(decompressedTime);

  return {compressionThroughput, decompressionThroughput};
}

bool areVectorsEqual(const std::vector<float>& a, const std::vector<float>& b, float epsilon = 1e-5) {
  if (a.size() != b.size()) {
    return false;
  }
  for (size_t i = 0; i < a.size(); i++) {
    if (std::fabs(a[i] - b[i]) > 0) { // Compare with tolerance for floating-point values
      return false;
    }
  }
  return true;
}
bool areVectorsEqualdouble(const std::vector<double>& a, const std::vector<double>& b, float epsilon = 1e-5) {
  if (a.size() != b.size()) {
    return false;
  }
  for (size_t i = 0; i < a.size(); i++) {
    if (std::fabs(a[i] - b[i]) > 0) { // Compare with tolerance for floating-point values
      return false;
    }
  }
  return true;
}
int main(int argc, char* argv[]) {
    cxxopts::Options options("DataCompressor", "Compress datasets and profile the compression");
    options.add_options()
        ("d,dataset", "Path to the dataset file", cxxopts::value<std::string>())
        ("o,outcsv", "Output CSV file path", cxxopts::value<std::string>())
        ("t,threads", "Number of threads to use", cxxopts::value<int>()->default_value("10"))
        ("b,bits", "Floating-point precision (32 or 64 bits)", cxxopts::value<int>()->default_value("64"))
        ("h,help", "Print help");

    auto result = options.parse(argc, argv);

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    std::string datasetPath = result["dataset"].as<std::string>();
    std::string outputCSV = result["outcsv"].as<std::string>();
    int numThreads = result["threads"].as<int>();
    int precisionBits = result["bits"].as<int>();


    size_t rowCount;
   std::vector<std::vector<size_t>> componentSizesList;

    if (precisionBits == 64) {
        auto [floatArray, rows] = loadTSVDatasetdouble(datasetPath);
        std::cout << "Loaded " << rows << " rows with " << floatArray.size() << " total values (64-bit)." << std::endl;
        if (floatArray.empty()) {
            std::cerr << "Failed to load dataset from " << datasetPath << std::endl;
            return 1;
        }
        globalByteArray = convertDoubleToBytes(floatArray);

        rowCount = rows;
       componentSizesList = {
        {7, 1, 0, 0, 0, 0, 0, 0},
        {6, 2, 0, 0, 0, 0, 0, 0},
        {6, 1, 1, 0, 0, 0, 0, 0},
        {5, 3, 0, 0, 0, 0, 0, 0},
        {5, 2, 1, 0, 0, 0, 0, 0},
        {5, 1, 1, 1, 0, 0, 0, 0},
        {4, 4, 0, 0, 0, 0, 0, 0},
        {4, 3, 1, 0, 0, 0, 0, 0},
        {4, 2, 2, 0, 0, 0, 0, 0},
        {4, 2, 1, 1, 0, 0, 0, 0},
        {4, 1, 1, 1, 1, 0, 0, 0},
        {3, 3, 2, 0, 0, 0, 0, 0},
        {3, 3, 1, 1, 0, 0, 0, 0},
        {3, 2, 2, 1, 0, 0, 0, 0},
        {3, 2, 1, 1, 1, 0, 0, 0},
        {3, 1, 1, 1, 1, 1, 0, 0},
        {2, 2, 2, 2, 0, 0, 0, 0},
        {2, 2, 2, 1, 1, 0, 0, 0},
        {2, 2, 1, 1, 1, 1, 0, 0},
        {2, 1, 1, 1, 1, 1, 1, 0},
        {1, 1, 1, 1, 1, 1, 1, 1},
        {1, 5, 1, 1, 0, 0, 0, 0},
        {1, 4, 2, 1, 0, 0, 0, 0},
        {2, 3, 2, 1, 0, 0, 0, 0}
      };

    } else if (precisionBits == 32) {
        auto [floatArray, rows] = loadTSVDataset(datasetPath);
        std::cout << "Loaded " << rows << " rows with " << floatArray.size() << " total values (32-bit)." << std::endl;
        if (floatArray.empty()) {
            std::cerr << "Failed to load dataset from " << datasetPath << std::endl;
            return 1;
        }
        globalByteArray = convertFloatToBytes(floatArray);
      ////////////////////////
      std::vector<uint8_t> byteArray = convertFloatToBytes(floatArray);
      std::vector<float> reconstructedArray = convertBytesToFloat(byteArray);

      // Check if the reconstructed array matches the original
      if (areVectorsEqual(floatArray, reconstructedArray)) {
        std::cout << "Reconstruction successful! Arrays are equal." << std::endl;
      } else {
        std::cout << "Reconstruction failed! Arrays are not equal." << std::endl;
      }
      ///
        rowCount = rows;
         componentSizesList = {

          {1, 1, 1, 1},
          {1, 1, 2,0},
         {1, 2, 1,0},
          {1, 3, 0,0},
          {2, 1, 1,0},
         {2, 2,0,0},
          {3, 1,0,0},

      };

    } else {
        std::cerr << "Unsupported floating-point precision: " << precisionBits << ". Use 32 or 64." << std::endl;
        return 1;
    }

std::vector<ProfilingInfo> pi_array;

    for (const auto& componentSizes : componentSizesList) {
        std::cout << "Testing with component sizes: ";
        for (auto size : componentSizes) std::cout << size << " ";
        std::cout << std::endl;

        std::vector<std::vector<uint8_t>> compressedComponents(componentSizes.size());
        double compressionThroughput = 0.0, decompressionThroughput = 0.0;

        for (int i = 0; i < 20; ++i) {
          // Outer loop for 3 runs
          // --- Full Compression ---
          ProfilingInfo pi_full(componentSizes.size());
          std::vector<uint8_t> compressedData, decompressedData;

          auto start = std::chrono::high_resolution_clock::now();
          double compressedSize = zstdCompression(globalByteArray, pi_full, compressedData);
          auto end = std::chrono::high_resolution_clock::now();
          pi_full.total_time_compressed = std::chrono::duration<double>(end - start).count();

          start = std::chrono::high_resolution_clock::now();
          zstdDecompression(compressedData, decompressedData, pi_full);
          end = std::chrono::high_resolution_clock::now();
          pi_full.total_time_decompressed = std::chrono::duration<double>(end - start).count();

          pi_full.com_ratio = calculateCompressionRatio(globalByteArray.size(), compressedSize);
          std::tie(compressionThroughput, decompressionThroughput) =
              calculateCompDecomThroughput(globalByteArray.size(), pi_full.total_time_compressed, pi_full.total_time_decompressed);

          pi_full.compression_throughput = compressionThroughput;
          pi_full.decompression_throughput = decompressionThroughput;
          pi_full.total_values = rowCount;
          pi_full.type = "Full";
          pi_array.push_back(pi_full);

          // --- Sequential Compression ---
          ProfilingInfo pi_seq(componentSizes.size());
          compressedComponents.clear();
          compressedComponents.resize(componentSizes.size());

          start = std::chrono::high_resolution_clock::now();
          compressedSize = zstdDecomposedSequential(globalByteArray, pi_seq, compressedComponents, componentSizes);
          end = std::chrono::high_resolution_clock::now();
          pi_seq.total_time_compressed = std::chrono::duration<double>(end - start).count();

          start = std::chrono::high_resolution_clock::now();
          zstdDecomposedSequentialDecompression(compressedComponents, pi_seq, componentSizes);
          end = std::chrono::high_resolution_clock::now();
          pi_seq.total_time_decompressed = std::chrono::duration<double>(end - start).count();

          pi_seq.com_ratio = calculateCompressionRatio(globalByteArray.size(), compressedSize);
          std::tie(compressionThroughput, decompressionThroughput) =
              calculateCompDecomThroughput(globalByteArray.size(), pi_seq.total_time_compressed, pi_seq.total_time_decompressed);

          pi_seq.compression_throughput = compressionThroughput;
          pi_seq.decompression_throughput = decompressionThroughput;
          pi_seq.total_values = rowCount;
          pi_seq.type = "Sequential";
          pi_array.push_back(pi_seq);

          // --- Parallel Compression ---
          ProfilingInfo pi_parallel(componentSizes.size());
          compressedComponents.clear();
          compressedComponents.resize(componentSizes.size());

          start = std::chrono::high_resolution_clock::now();
          compressedSize = zstdDecomposedParallel(globalByteArray, pi_parallel, compressedComponents, componentSizes, numThreads);
          end = std::chrono::high_resolution_clock::now();
          pi_parallel.total_time_compressed = std::chrono::duration<double>(end - start).count();

          start = std::chrono::high_resolution_clock::now();
          std::vector<uint8_t> decompressedData2 = zstdDecomposedParallelDecompression(compressedComponents, pi_parallel, componentSizes, numThreads);
          end = std::chrono::high_resolution_clock::now();
          pi_parallel.total_time_decompressed = std::chrono::duration<double>(end - start).count();

          pi_parallel.com_ratio = calculateCompressionRatio(globalByteArray.size(), compressedSize);
          std::tie(compressionThroughput, decompressionThroughput) =
              calculateCompDecomThroughput(globalByteArray.size(), pi_parallel.total_time_compressed, pi_parallel.total_time_decompressed);

          pi_parallel.compression_throughput = compressionThroughput;
          pi_parallel.decompression_throughput = decompressionThroughput;
          pi_parallel.total_values = rowCount;
          pi_parallel.type = "Parallel";
          if (precisionBits == 64) {
            // Load original dataset
            auto [floatArray1, rows] = loadTSVDatasetdouble(datasetPath);

            // Convert decompressed byte data back to double array
            std::vector<double> reconstructedArray2 = convertBytesToDouble(decompressedData2);

            // Compare original and reconstructed arrays
            if (areVectorsEqualdouble(floatArray1, reconstructedArray2)) {
              std::cout << "Reconstruction successful! Arrays are equal." << std::endl;
            } else {
              std::cerr << "Reconstruction failed! Arrays are not equal." << std::endl;
            }
          } else if (precisionBits == 32) {
            // Convert decompressed byte data back to float array
            std::vector<float> reconstructedArray2 = convertBytesToFloat(decompressedData2);

            // Load original dataset
            auto [floatArray, rows] = loadTSVDataset(datasetPath);

            // Compare original and reconstructed arrays
            if (areVectorsEqual(floatArray, reconstructedArray2)) {
              std::cout << "Reconstruction successful! Arrays are equal." << std::endl;
            } else {
              std::cerr << "Reconstruction failed! Arrays are not equal." << std::endl;
            }
          } else {
            std::cerr << "Unsupported precision: " << precisionBits << ". Use 32 or 64." << std::endl;
          }

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
    file << "Iteration,OuterLoop,ComponentSizes,id,RunType,CompressionRatio,TotalTimeCompressed,TotalTimeDecompressed,"
         << "Component1Time,Component2Time,Component3Time,Component4Time,Component5Time,"
         << "Component6Time,Component7Time,Component8Time,CompressionThroughput,DecompressionThroughput,TotalValues\n";

    size_t iteration = 1; // Global iteration counter

    for (size_t configIndex = 0; configIndex < componentSizesList.size(); ++configIndex) {
        const auto& componentSizes = componentSizesList[configIndex];

        for (int i = 0; i < 3; ++i) { // Outer loop for 3 runs
            for (int runTypeIndex = 0; runTypeIndex < 3; ++runTypeIndex) { // Full, Sequential, Parallel
                const ProfilingInfo& pi = pi_array[(configIndex * 3 * 3) + (i * 3) + runTypeIndex];

                // Write iteration, outer loop index, component sizes, and profiling data
                file << iteration++ << ",";    // Global iteration count
                file << i + 1 << ",";          // Outer loop iteration (1-based index)
                for (size_t size : componentSizes) {
                    file << size << " ";
                }
                file << ",";

                // Append profiling information in CSV format
                pi.printCSV(file, iteration);
            }
        }
    }

    return 0;
}
