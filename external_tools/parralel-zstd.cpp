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
std::vector<uint8_t> globalByteArrayNon;


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
//////RLE//////////////////////
// Helper function to convert RLE metadata to a flattened array
template <typename T>
std::vector<float> convertRLE(const std::vector<std::tuple<int, T, int>>& metadata) {
  std::vector<float> consecutiveValues;
  for (const auto& item : metadata) {
    int startIndex;
    T value;
    int count;
    std::tie(startIndex, value, count) = item;

    consecutiveValues.push_back(static_cast<float>(startIndex));
    consecutiveValues.push_back(static_cast<float>(value));
    consecutiveValues.push_back(static_cast<float>(count));
  }
  return consecutiveValues;
}

// Function to split array based on consecutive values
template <typename T>
std::tuple<std::vector<T>, std::vector<std::tuple<int, T, int>>>
splitArrayOnMultipleConsecutiveValues(const std::vector<T>& data, int threshold) {
  int totalLength = data.size();
  int consecutiveCount = 1;
  int startIdx = 0;

  std::vector<T> nonConsecutiveArray;
  std::vector<std::tuple<int, T, int>> metadata;

  for (int i = 1; i < totalLength; ++i) {
    if (data[i] == data[i - 1]) {
      consecutiveCount++;
    } else {
      if (consecutiveCount > threshold) {
        metadata.emplace_back(i - consecutiveCount, data[i - 1], consecutiveCount);
      } else {
        nonConsecutiveArray.insert(nonConsecutiveArray.end(), data.begin() + startIdx, data.begin() + i);
      }
      startIdx = i;
      consecutiveCount = 1;
    }
  }

  // Handle the last segment
  if (consecutiveCount > threshold) {
    metadata.emplace_back(totalLength - consecutiveCount, data.back(), consecutiveCount);
  } else {
    nonConsecutiveArray.insert(nonConsecutiveArray.end(), data.begin() + startIdx, data.end());
  }

  return std::make_tuple(nonConsecutiveArray, metadata);
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
    int threshold =5;
  size_t  metadataSize=0;


    size_t rowCount;
   std::vector<std::vector<size_t>> componentSizesList;

    if (precisionBits == 64) {
        auto [floatArray, rows] = loadTSVDatasetdouble(datasetPath);
      // Convert float to int32_t (bitwise reinterpretation)
     // auto intData = reinterpretToInteger(floatArray);

      // Call the function for int32_t
      auto [nonConsecutiveArray, metadata] = splitArrayOnMultipleConsecutiveValues(floatArray, threshold);
      std::vector<float> flattenedRLE = convertRLE(metadata);
        std::cout << "Loaded " << rows << " rows with " << floatArray.size() << " total values (64-bit)." << std::endl;
        std::cout << "Loaded " << rows << " rows with " << nonConsecutiveArray.size() << " nonConsecutiveArray values (64-bit)." << std::endl;
      // Measure original metadata size
       metadataSize = metadata.size() * sizeof(std::tuple<int, double, int>);
      std::cout << "Size of original metadata: " << metadataSize << " bytes" << std::endl;

        if (floatArray.empty()) {
            std::cerr << "Failed to load dataset from " << datasetPath << std::endl;
            return 1;
        }
        globalByteArrayNon = convertDoubleToBytes(nonConsecutiveArray);
      globalByteArray = convertDoubleToBytes(floatArray);


        rowCount = rows;
       componentSizesList = {
       {7, 1},
        {6, 2},
        {6, 1, 1},
        {5, 3},
        {5, 2, 1},
        {5, 1, 1, 1},
        {4, 4},
        {4, 3, 1},
        {4, 2, 2},
        {4, 2, 1, 1},
        {4, 1, 1, 1, 1},
        {3, 3, 2},
        {3, 3, 1, 1},
        {3, 2, 2, 1},
        {3, 2, 1, 1, 1},
        {3, 1, 1, 1, 1, 1},
        {2, 2, 2, 2},
        {2, 2, 2, 1, 1},
        {2, 2, 1, 1, 1, 1},
        {2, 1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1, 1, 1, 1},
        {1, 5, 1, 1},
        {1, 4, 2, 1},
        {2, 3, 2, 1}
      };

    } else if (precisionBits == 32) {
        auto [floatArray, rows] = loadTSVDataset(datasetPath);
      auto [nonConsecutiveArray, metadata] = splitArrayOnMultipleConsecutiveValues(floatArray, threshold);
      std::vector<float> flattenedRLE = convertRLE(metadata);
        std::cout << "Loaded " << rows << " rows with " << floatArray.size() << " total values (32-bit)." << std::endl;
        std::cout << "Loaded " << rows << " rows with " << nonConsecutiveArray.size() << " nonConsecutiveArray values (32-bit)." << std::endl;
      // Measure original metadata size
       metadataSize = metadata.size() * sizeof(std::tuple<int, float, int>);
      std::cout << "Size of original metadata: " << metadataSize << " bytes" << std::endl;

        if (floatArray.empty()) {
            std::cerr << "Failed to load dataset from " << datasetPath << std::endl;
            return 1;
        }
        globalByteArrayNon = convertFloatToBytes(nonConsecutiveArray);
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
         {1, 1, 2},
         {1, 2, 1},
         {1, 3},
         {2, 1, 1},
         {2, 2},
          {3, 1},

      };

    } else {
        std::cerr << "Unsupported floating-point precision: " << precisionBits << ". Use 32 or 64." << std::endl;
        return 1;
    }
std::vector<int> threadSizesList = {28};
std::vector<ProfilingInfo> pi_array;
  int iter=1;

    for (const auto& componentSizes : componentSizesList) {

      std::cout << "Testing with component sizes: ";
      for (auto size : componentSizes) std::cout << size << " ";
      std::cout << std::endl;

      std::vector<std::vector<uint8_t>> compressedComponents(componentSizes.size());
      double compressionThroughput = 0.0, decompressionThroughput = 0.0;

     for (int numThreads : threadSizesList) {
       std::cout << "Testing with threads: " << numThreads << std::endl;
        for (int i = 0; i < iter; i++) {
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

        // // --- Sequential Compression ---
        // ProfilingInfo pi_seq(componentSizes.size());
        // compressedComponents.clear();
        // compressedComponents.resize(componentSizes.size());
        //
        // start = std::chrono::high_resolution_clock::now();
        // compressedSize = zstdDecomposedSequential(globalByteArray, pi_seq, compressedComponents, componentSizes);
        // end = std::chrono::high_resolution_clock::now();
        // pi_seq.total_time_compressed = std::chrono::duration<double>(end - start).count();
        //
        // start = std::chrono::high_resolution_clock::now();
        // zstdDecomposedSequentialDecompression(compressedComponents, pi_seq, componentSizes);
        // end = std::chrono::high_resolution_clock::now();
        // pi_seq.total_time_decompressed = std::chrono::duration<double>(end - start).count();
        //
        // pi_seq.com_ratio = calculateCompressionRatio(globalByteArray.size(), compressedSize);
        // std::tie(compressionThroughput, decompressionThroughput) =
        //     calculateCompDecomThroughput(globalByteArray.size(), pi_seq.total_time_compressed, pi_seq.total_time_decompressed);
        //
        // pi_seq.compression_throughput = compressionThroughput;
        // pi_seq.decompression_throughput = decompressionThroughput;
        // pi_seq.total_values = rowCount;
        // pi_seq.type = "Sequential";
        // pi_array.push_back(pi_seq);

        // --- Parallel Compression ---
        ProfilingInfo pi_parallel(componentSizes.size());
        compressedComponents.clear();
        compressedComponents.resize(componentSizes.size());

        start = std::chrono::high_resolution_clock::now();
        compressedSize = zstdDecomposedParallel(globalByteArrayNon, pi_parallel, compressedComponents, componentSizes, numThreads);
        end = std::chrono::high_resolution_clock::now();
        pi_parallel.total_time_compressed = std::chrono::duration<double>(end - start).count();

        start = std::chrono::high_resolution_clock::now();
        std::vector<uint8_t> decompressedData2 = zstdDecomposedParallelDecompression(compressedComponents, pi_parallel, componentSizes, numThreads);
        end = std::chrono::high_resolution_clock::now();
        pi_parallel.total_time_decompressed = std::chrono::duration<double>(end - start).count();

        pi_parallel.com_ratio = calculateCompressionRatio(globalByteArray.size(), compressedSize+metadataSize);
        std::tie(compressionThroughput, decompressionThroughput) =
            calculateCompDecomThroughput(globalByteArray.size(), pi_parallel.total_time_compressed, pi_parallel.total_time_decompressed);

        pi_parallel.compression_throughput = compressionThroughput;
        pi_parallel.decompression_throughput = decompressionThroughput;
        pi_parallel.total_values = rowCount;
        pi_parallel.type = "Parallel";
        pi_parallel.thread_count=numThreads;
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
}
  // Initialize the CSV file and write the header
  std::ofstream file(outputCSV);
  if (!file) {
    std::cerr << "Failed to open the file for writing: " << outputCSV << std::endl;
    return 1;
  }

  file << "Iteration,OuterLoop,ComponentSizes,ThreadCount,RunType,CompressionRatio,TotalTimeCompressed,TotalTimeDecompressed,"
       << "CompressionThroughput,DecompressionThroughput,entropy_full_byte,entropy_decompose_byte,TotalValues\n";

  // Iterate over each configuration of component sizes
  int globalIteration = 1;
  size_t pi_index = 0;

  for (const auto& componentSizes : componentSizesList) {
    std::string componentSizesStr;
    for (auto size : componentSizes) {
      componentSizesStr += std::to_string(size) + " ";
    }

    for (int i : threadSizesList) {
      for(size_t i = 0; i < iter; i++) {
      for (int runTypeIndex = 0; runTypeIndex < 2; ++runTypeIndex) { // Full, Sequential, Parallel
        const ProfilingInfo& pi = pi_array[pi_index++];

        file << globalIteration << ","
             << i  << ","
             << componentSizesStr << ","
             << pi.thread_count << "," // Add thread count here
             << pi.type << ","
             << pi.com_ratio << ","
             << pi.total_time_compressed << ","
             << pi.total_time_decompressed << ","
             << pi.compression_throughput << ","
             << pi.decompression_throughput << ","
             <<pi.entropy_full_byte<<","
             <<pi.entropy_decompose_byte<<","
             << pi.total_values << "\n";
      }
      globalIteration++;
    }
  }
}
  file.close();


  return 0;
}