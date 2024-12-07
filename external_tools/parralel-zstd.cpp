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
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;


std::vector<uint8_t> globalByteArray;
#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <iomanip>

// Function to calculate Shannon entropy
double calculateEntropy(const std::vector<uint8_t>& data) {
    std::vector<size_t> frequency(256, 0);
    for (uint8_t byte : data) {
        frequency[byte]++;
    }
    double entropy = 0.0;
    size_t dataSize = data.size();
    for (size_t freq : frequency) {
        if (freq > 0) {
            double probability = static_cast<double>(freq) / dataSize;
            entropy -= probability * log2(probability);
        }
    }
    return entropy;
}
double calculateFloatEntropy(const std::vector<float>& data) {
  std::unordered_map<float, size_t> frequency;
  for (float value : data) {
    frequency[value]++;
  }

  double entropy = 0.0;
  size_t dataSize = data.size();
  for (const auto& [value, count] : frequency) {
    double probability = static_cast<double>(count) / dataSize;
    entropy -= probability * log2(probability);
  }
  return entropy;
}

void analyzeCompressionImpact(const std::vector<uint8_t>& globalByteArray,
                              const std::vector<std::vector<uint8_t>>& decomposedComponents,
                              const ProfilingInfo& fullProfile,
                              const ProfilingInfo& decomposedProfile,
                              const std::vector<float>& fullFloats) {
    // Float-Level Entropy Analysis
    std::cout << "===== Float-Level Entropy Analysis =====\n";
    double floatEntropy = calculateFloatEntropy(fullFloats);
    std::cout << "Full Data Float Entropy: " << floatEntropy << "\n";

    // Byte-Level Entropy Analysis
    std::cout << "\n===== Byte-Level Entropy Analysis =====\n";
    double fullEntropy = calculateEntropy(globalByteArray);
    std::cout << "Full Data Byte-Level Entropy: " << fullEntropy << "\n";

    double combinedEntropy = 0.0;
    for (size_t i = 0; i < decomposedComponents.size(); ++i) {
        double componentEntropy = calculateEntropy(decomposedComponents[i]);
        combinedEntropy += componentEntropy;
        std::cout << "Component " << i + 1 << " Byte-Level Entropy: " << componentEntropy << "\n";
    }
    std::cout << "Combined Decomposed Byte-Level Entropy: " << combinedEntropy << "\n\n";

    // Compression Ratio Analysis
    std::cout << "===== Compression Ratio Analysis =====\n";
    std::cout << "Full Data Compression Ratio: " << fullProfile.com_ratio << "\n";
    std::cout << "Decomposed Data Compression Ratio: " << decomposedProfile.com_ratio << "\n";
    double improvement = decomposedProfile.com_ratio - fullProfile.com_ratio;
    std::cout << "Compression Ratio Improvement: " << improvement << "\n\n";

    // Hypothetical Block Analysis
    std::cout << "===== Block Analysis =====\n";
    size_t fullBlockCount = 1;  // Example: Replace with real block count
    size_t decomposedBlockCount = decomposedComponents.size();
    std::cout << "Full Data Blocks: " << fullBlockCount << "\n";
    std::cout << "Decomposed Data Blocks: " << decomposedBlockCount << "\n";
    std::cout << "Decomposed Compression likely produced smaller, more efficient blocks.\n\n";

    // Summary
    std::cout << "===== Summary =====\n";
    if (improvement > 0) {
        std::cout << "Decomposed compression outperforms full compression due to:\n";
        std::cout << " - Patterns revealed in individual components.\n";
        std::cout << " - Smaller blocks with better match lengths.\n";
        std::cout << " - Increased efficiency in RLE and Huffman coding.\n";
    } else {
        std::cout << "No significant improvement from decomposition.\n";
    }
}



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

// Function to convert floats to bytes (IEEE 754 format)
std::vector<uint8_t> convertFloatToBytes1(const std::vector<float>& floatArray) {
  // Prepare a byte array large enough to store all float bytes
  std::vector<uint8_t> byteArray(floatArray.size() * sizeof(float));

  // Copy each float's bytes into the byte array
  for (size_t i = 0; i < floatArray.size(); ++i) {
    // Use memcpy to safely copy the 4 bytes of the float into the byte array
    memcpy(byteArray.data() + i * sizeof(float), &floatArray[i], sizeof(float));
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
// Function to get the compressed size of a file
size_t getCompressedFileSize(const std::string& filename) {
  std::ifstream inFile(filename, std::ios::binary | std::ios::ate); // Open file in binary mode, positioned at the end
  if (!inFile) {
    std::cerr << "Error: Could not open file: " << filename << std::endl;
    return 0;
  }

  // Get the file size
  size_t fileSize = inFile.tellg();
  inFile.close();

  return fileSize;
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

      // {1, 1, 1, 1},
      //  {1, 1, 2},
      // {1, 2, 1},
      //   {1, 3},
      {2, 1, 1},
     // {2, 2},
     //  {3, 1},

  };

  } else {
    std::cerr << "Unsupported floating-point precision: " << precisionBits << ". Use 32 or 64." << std::endl;
    return 1;
  }
  std::vector<int> threadSizesList = {32};
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
        ProfilingInfo fullProfile = pi_full; // After running zstdCompression
        ProfilingInfo decomposedProfile = pi_parallel; // After running zstdDecomposedParallel
        auto [fullFloats, rows] = loadTSVDataset(datasetPath);
        analyzeCompressionImpact(globalByteArray, compressedComponents, fullProfile, decomposedProfile, fullFloats);




      }

    }


    // Initialize the CSV file and write the header
    std::ofstream file(outputCSV);
    if (!file) {
      std::cerr << "Failed to open the file for writing: " << outputCSV << std::endl;
      return 1;
    }

    file << "Iteration,OuterLoop,ComponentSizes,ThreadCount,RunType,CompressionRatio,TotalTimeCompressed,TotalTimeDecompressed,"
         << "CompressionThroughput,DecompressionThroughput,TotalValues\n";

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
          for (int runTypeIndex = 0; runTypeIndex < 3; ++runTypeIndex) { // Full, Sequential, Parallel
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
                 << pi.total_values << "\n";
          }
          globalIteration++;
        }
      }
    }
    file.close();
    // Specify the folder to search for .zst files
    std::string folderPath = "/home/jamalids/Documents/file";

    // Check if the folder exists
    if (!fs::exists(folderPath) || !fs::is_directory(folderPath)) {
      std::cerr << "Error: Folder does not exist or is not a directory: " << folderPath << std::endl;
      return 1;
    }

    // Iterate over all files in the folder
    std::cout << "Reading .zst files in folder: " << folderPath << "\n" << std::endl;
    for (const auto& entry : fs::directory_iterator(folderPath)) {
      // Check if the file has a .zst extension
      if (entry.is_regular_file() && entry.path().extension() == ".zst") {
        std::string filepath = entry.path().string();
        size_t compressedSize = getCompressedFileSize(filepath);
        if (compressedSize > 0) {
          std::cout << "File: " << filepath << "\nCompressed size: " << compressedSize << " bytes\n" << std::endl;

        }

      }
    }

    return 0;
  }
}
