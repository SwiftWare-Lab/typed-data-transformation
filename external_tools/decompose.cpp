//
// Created by jamalids on 06/02/25.
//

#include "decompose.h"
// Created by jamalids on 04/11/24.
//
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <chrono>
#include <cstdint>
#include <omp.h>
#include <cstring>
#include <map>
#include <cmath>
#include <cxxopts.hpp>

// NOTE: Include your updated FastLZ-based parallel header.
// Ensure that this header includes the new functions for the reverse-order (decompose then chunk) approach.
#include "decompose.h"

std::vector<uint8_t> globalByteArray;

// Map to store dataset names and their multiple possible configurations
std::map<std::string, std::vector<std::vector<std::vector<size_t>>>> datasetComponentMap = {
  {"acs_wht_f32", {
          {{1,2}, {3}, {4}} ,
          // {{1, 2,3}, {4}},
  }},
  {"g24_78_usb2_f32", {
          {{1}, {2,3}, {4}},
          {{1,2,3}, {4}},
  }},
  {"jw_mirimage_f32", {
          {{1,2}, {3}, {4}},
          {{1,2,3}, {4}},
  }},
  {"spitzer_irac_f32", {
          {{1,2}, {3}, {4}},
          {{1,2,3}, {4}},
  }},
  {"turbulence_f32", {
          {{1,2}, {3}, {4}},
          {{1,2,3}, {4}},
  }},
  {"wave_f32", {
          {{1,2}, {3}, {4}},
          {{1,2,3}, {4}},
  }},
  {"hdr_night_f32", {
          {{1,4}, {2}, {3}},
          {{1}, {2}, {3}, {4}},
          {{1,4}, {2,3}},
  }},
  {"ts_gas_f32", {
          {{1,2}, {3}, {4}},
  }},
  {"solar_wind_f32", {
          {{1}, {4}, {2}, {3}},
          {{1}, {2,3}, {4}},
  }},
  {"tpch_lineitem_f32", {
          {{1,2,3}, {4}},
          {{1,2}, {3}, {4}},
  }},
  {"tpcds_web_f32", {
          {{1,2,3}, {4}},
          {{1}, {2,3}, {4}},
  }},
  {"tpcds_store_f32", {
          {{1,2,3}, {4}},
          {{1}, {2,3}, {4}},
  }},
  {"tpcds_catalog_f32", {
          {{1,2,3}, {4}},
          {{1}, {2,3}, {4}},
  }},
  {"citytemp_f32", {
          {{1,4}, {2,3}},
          {{1}, {2}, {3}, {4}},
          {{1,2}, {3}, {4}}
  }},
  {"hst_wfc3_ir_f32", {
          {{1}, {2}, {3}, {4}},
          {{1,2}, {3}, {4}}
  }},
  {"hst_wfc3_uvis_f32", {
          {{1}, {2}, {3}, {4}},
          {{1,2}, {3}, {4}}
  }},
  {"rsim_f32", {
          {{1,2,3}, {4}},
          {{1,2}, {3}, {4}}
  }},
  {"astro_mhd_f64", {
          {{1,2,3,4,5,6}, {7}, {8}},
          {{1,2,3,4,5}, {6}, {7}, {8}}
  }},
  {"astro_pt_f64", {
          {{1,2,3,4,5,6}, {7}, {8}},
          {{1,2,3,4,5}, {6}, {7}, {8}}
  }},
  {"jane_street_f64", {
          {{1,2,3,4,5,6}, {7}, {8}},
          {{3,2,5,6,4,1}, {7}, {8}}
  }},
  {"msg_bt_f64", {
          {{1,2,3,4,5}, {6}, {7}, {8}},
          {{3,2,1,4,5,6}, {7}, {8}},
          {{3,2,1,4,5}, {6}, {7}, {8}}
  }},
  {"num_brain_f64", {
          {{1,2,3,4,5,6}, {7}, {8}},
          {{3,2,4,5,1,6}, {7}, {8}},
  }},
  {"num_control_f64", {
          {{1,2,3,4,5,6}, {7}, {8}},
          {{4,5}, {6,3}, {1,2}, {7}, {8}},
          {{4,5,6,3,1,2}, {7}, {8}},
  }},
  {"nyc_taxi2015_f64", {
          {{7,4,6}, {5}, {3,2,1,8}},
          {{7,4,6,5}, {3,2,1,8}},
          {{7,4,6}, {5}, {3,2,1}, {8}},
          {{7,4}, {6}, {5}, {3,2}, {1}, {8}},
  }},
  {"phone_gyro_f64", {
          {{4,6}, {5}, {3,2,1,7}, {8}},
          {{4,6}, {1}, {3,2}, {5}, {7}, {8}},
          {{6,4,3,2,1,7}, {5}, {8}},
  }},
  {"tpch_order_f64", {
          {{3,2,4,1}, {7}, {6,5}, {8}},
          {{3,2,4,1,7}, {6,5}, {8}},
  }},
  {"tpcxbb_store_f64", {
          {{4,2,3}, {1}, {5}, {7}, {6}, {8}},
          {{4,2,3,1}, {5}, {7,6}, {8}},
          {{4,2,3,1,5}, {7,6}, {8}},
  }},
  {"tpcxbb_web_f64", {
          {{4,2,3}, {1}, {5}, {7}, {6}, {8}},
          {{4,2,3,1}, {5}, {7,6}, {8}},
          {{4,2,3,1,5}, {7,6}, {8}},
  }},
  {"wesad_chest_f64", {
          {{7,5}, {6}, {8,4,1}, {3,2}},
          {{7,5}, {6}, {8,4}, {1}, {3,2}},
          {{7,5}, {6}, {8,4,1,3,2}},
  }},
  {"default", {
          {{1}, {2}, {3}, {4}}
  }}
};

// --- Helper: Split a vector into N nearly equal chunks ---
std::vector<std::vector<uint8_t>> splitIntoChunks(const std::vector<uint8_t>& data, int numChunks) {
  std::vector<std::vector<uint8_t>> chunks;
  size_t totalSize = data.size();
  size_t chunkSize = totalSize / numChunks;
  size_t remainder = totalSize % numChunks;
  size_t offset = 0;
  for (int i = 0; i < numChunks; i++) {
    size_t currentSize = chunkSize + (i < remainder ? 1 : 0);
    std::vector<uint8_t> chunk(data.begin() + offset, data.begin() + offset + currentSize);
    chunks.push_back(chunk);
    offset += currentSize;
  }
  return chunks;
}

// Function to get all configurations for a dataset
std::vector<std::vector<std::vector<size_t>>> getComponentConfigurationsForDataset(const std::string& datasetName) {
  auto it = datasetComponentMap.find(datasetName);
  if (it != datasetComponentMap.end()) {
    return it->second;
  }
  return datasetComponentMap["default"];
}

std::string extractDatasetName(const std::string& filePath) {
  size_t lastSlashPos = filePath.find_last_of("/\\");
  std::string fileName = (lastSlashPos == std::string::npos) ? filePath : filePath.substr(lastSlashPos + 1);
  size_t lastDotPos = fileName.find_last_of('.');
  return (lastDotPos == std::string::npos) ? fileName : fileName.substr(0, lastDotPos);
}

std::string configToString1(const std::vector<std::vector<size_t>>& config) {
  std::stringstream ss;
  ss << "{ ";
  for (size_t i = 0; i < config.size(); ++i) {
    ss << "[";
    for (size_t j = 0; j < config[i].size(); ++j) {
      ss << config[i][j];
      if (j + 1 < config[i].size())
        ss << " ";
    }
    ss << "]";
    if (i + 1 < config.size())
      ss << "- ";
  }
  ss << " }";
  return ss.str();
}

// ----------------------------------------------------------------------------
// Dataset loading functions (for TSV files)
// ----------------------------------------------------------------------------
std::pair<std::vector<float>, size_t> loadTSVDataset(const std::string& filePath) {
  std::vector<float> floatArray;
  std::ifstream file(filePath);
  std::string line;
  size_t rowCount = 0;
  if (file.is_open()) {
    while (std::getline(file, line)) {
      std::stringstream ss(line);
      std::string value;
      std::getline(ss, value, '\t'); // skip first column
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

std::pair<std::vector<double>, size_t> loadTSVDatasetdouble(const std::string& filePath) {
  std::vector<double> doubleArray;
  std::ifstream file(filePath);
  std::string line;
  size_t rowCount = 0;
  if (file.is_open()) {
    while (std::getline(file, line)) {
      std::stringstream ss(line);
      std::string value;
      std::getline(ss, value, '\t'); // skip first column
      while (std::getline(ss, value, '\t')) {
        doubleArray.push_back(std::stod(value));
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
  if (byteArray.size() % 4 != 0)
    throw std::runtime_error("Byte array size is not a multiple of 4.");
  std::vector<float> floatArray(byteArray.size() / 4);
  for (size_t i = 0; i < floatArray.size(); i++) {
    const uint8_t* bytePtr = &byteArray[i * 4];
    float* floatPtr = reinterpret_cast<float*>(const_cast<uint8_t*>(bytePtr));
    floatArray[i] = *floatPtr;
  }
  return floatArray;
}

std::vector<uint8_t> convertDoubleToBytes(const std::vector<double>& doubleArray) {
  std::vector<uint8_t> byteArray(doubleArray.size() * 8);
  for (size_t i = 0; i < doubleArray.size(); i++) {
    uint8_t* doubleBytes = reinterpret_cast<uint8_t*>(const_cast<double*>(&doubleArray[i]));
    for (size_t j = 0; j < 8; j++) {
      byteArray[i * 8 + j] = doubleBytes[j];
    }
  }
  return byteArray;
}

std::vector<double> convertBytesToDouble(const std::vector<uint8_t>& byteArray) {
  if (byteArray.size() % 8 != 0)
    throw std::runtime_error("Byte array size is not a multiple of 8.");
  std::vector<double> doubleArray(byteArray.size() / 8);
  for (size_t i = 0; i < doubleArray.size(); i++) {
    const uint8_t* bytePtr = &byteArray[i * 8];
    const double* doublePtr = reinterpret_cast<const double*>(bytePtr);
    doubleArray[i] = *doublePtr;
  }
  return doubleArray;
}

std::pair<double, double> calculateCompDecomThroughput(size_t originalSize, double compressedTime, double decompressedTime) {
  double originalSizeGB = static_cast<double>(originalSize) / 1e9;
  double compressionThroughput = originalSizeGB / static_cast<double>(compressedTime);
  double decompressionThroughput = originalSizeGB / static_cast<double>(decompressedTime);
  return {compressionThroughput, decompressionThroughput};
}

bool areVectorsEqual(const std::vector<float>& a, const std::vector<float>& b, float epsilon = 1e-5) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); i++) {
    if (std::fabs(a[i] - b[i]) > epsilon) return false;
  }
  return true;
}

bool areVectorsEqualdouble(const std::vector<double>& a, const std::vector<double>& b, float epsilon = 1e-5) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); i++) {
    if (std::fabs(a[i] - b[i]) > epsilon) return false;
  }
  return true;
}

// ----------------------------------------------------------------------------
// Helper: Block a vector into blocks of a given blockSize (in bytes)
// ----------------------------------------------------------------------------
std::vector<std::vector<uint8_t>> blockData(const std::vector<uint8_t>& data, size_t blockSize) {
  std::vector<std::vector<uint8_t>> blocks;
  size_t totalSize = data.size();
  size_t numBlocks = (totalSize + blockSize - 1) / blockSize;
  blocks.reserve(numBlocks);
  for (size_t i = 0; i < totalSize; i += blockSize) {
    size_t end = std::min(i + blockSize, totalSize);
    blocks.push_back(std::vector<uint8_t>(data.begin() + i, data.begin() + end));
  }
  return blocks;
}

/////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[]) {
  cxxopts::Options options("DataCompressorFastLZ",
                           "Compress datasets and profile the compression (FastLZ version)");
  options.add_options()
      ("d,dataset",   "Path to the dataset file", cxxopts::value<std::string>())
      ("o,outcsv",    "Output CSV file path",     cxxopts::value<std::string>())
      ("t,threads",   "Number of threads to use", cxxopts::value<int>()->default_value("10"))
      ("b,bits",      "Floating-point precision (32 or 64 bits)", cxxopts::value<int>()->default_value("64"))
      ("h,help",      "Print help");
  auto result = options.parse(argc, argv);
  if (result.count("help")) {
    std::cout << options.help() << std::endl;
    return 0;
  }

  std::string datasetPath = result["dataset"].as<std::string>();
  std::string outputCSV   = result["outcsv"].as<std::string>();
  int userThreads         = result["threads"].as<int>();
  int precisionBits       = result["bits"].as<int>();

  // List of threads to test.
  std::vector<int> threadList = { userThreads};

  // Number of runs.
  int runCount = 1;

  // 2. Load dataset
  size_t rowCount;
  std::string datasetName = extractDatasetName(datasetPath);
  std::cout << "Dataset Name: " << datasetName << std::endl;
  if (precisionBits == 64) {
    auto [doubleArray, rows] = loadTSVDatasetdouble(datasetPath);
    if (doubleArray.empty()) {
      std::cerr << "Failed to load dataset from " << datasetPath << std::endl;
      return 1;
    }
    globalByteArray = convertDoubleToBytes(doubleArray);
    rowCount = rows;
    std::cout << "Loaded " << rows << " rows (64-bit) with " << doubleArray.size() << " total values.\n";
  } else if (precisionBits == 32) {
    auto [floatArray, rows] = loadTSVDataset(datasetPath);
    if (floatArray.empty()) {
      std::cerr << "Failed to load dataset from " << datasetPath << std::endl;
      return 1;
    }
    globalByteArray = convertFloatToBytes(floatArray);
    rowCount = rows;
    std::cout << "Loaded " << rows << " rows (32-bit) with " << floatArray.size() << " total values.\n";
  } else {
    std::cerr << "Unsupported floating-point precision: " << precisionBits << ". Use 32 or 64." << std::endl;
    return 1;
  }

  // Determine total data size.
  size_t totalBytes = globalByteArray.size();

  // List of block sizes (in bytes) to test.
  std::vector<size_t> blockSizes = {
    100 * 1024,
    1000 * 1024,
    10000 * 1024,
    100000 * 1024,
    600 * 1024,
    640 * 1024,
  };

  // Open CSV output file.
  std::ofstream file(outputCSV);
  if (!file) {
    std::cerr << "Failed to open the file for writing: " << outputCSV << std::endl;
    return 1;
  }

  file << "Index;DatasetName;Threads;BlockSize;ConfigString;RunType;CompressionRatio;"
       << "TotalTimeCompressed;TotalTimeDecompressed;CompressionThroughput;DecompressionThroughput;TotalValues\n";

  int recordIndex = 1;

  // ----------------------------------------------------------------------
  // Loop over thread counts and run iterations.
  // ----------------------------------------------------------------------
  for (int currentThreads : threadList) {
    for (int run = 1; run <= runCount; run++) {

      std::cout << "\n[INFO] Starting run " << run << "/" << runCount
                << " with " << currentThreads << " threads.\n";

      int numThreads = currentThreads;
      /////////////////////////////////////////////

      // A. FULL COMPRESSION WITH BLOCKING - PARALLEL
      // ------------------------------
      for (size_t bs : blockSizes) {
        std::cout << "Testing with block size = " << bs << " bytes." << std::endl;

        size_t totalSize = globalByteArray.size();
        size_t numBlocks = (totalSize + bs - 1) / bs;

        size_t totalCompressedSize = 0;
        double totalCompTime = 0.0, totalDecompTime = 0.0;
        std::vector<std::vector<uint8_t>> compressedBlocks(numBlocks);
        std::vector<double> blockCompTimes(numBlocks, 0.0);
        std::vector<double> blockDecompTimes(numBlocks, 0.0);
        std::vector<size_t> blockCompressedSizes(numBlocks, 0);
        ProfilingInfo pi_parallel;
        pi_parallel.config_string = "N/A";

        omp_set_num_threads(numThreads);

        // Parallel Compression
        auto compStartOverall = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
        for (int i = 0; i < static_cast<int>(numBlocks); i++) {
          size_t start = i * bs;
          size_t end = std::min(start + bs, totalSize);
          size_t blockLength = end - start;

          const uint8_t* blockStart = globalByteArray.data() + start;
          auto startTime = std::chrono::high_resolution_clock::now();
          size_t cSize = compressWithFastLZ1(blockStart, blockLength, compressedBlocks[i]);
          auto endTime = std::chrono::high_resolution_clock::now();

          blockCompTimes[i] = std::chrono::duration<double>(endTime - startTime).count();
          blockCompressedSizes[i] = cSize;
#pragma omp atomic
          totalCompressedSize += cSize;
        }
        auto compEndOverall = std::chrono::high_resolution_clock::now();
        totalCompTime = std::chrono::duration<double>(compEndOverall - compStartOverall).count();

        // Parallel Decompression
        std::vector<uint8_t> finalReconstructed(totalSize, 0);
        double decompStartOverall = omp_get_wtime();
#pragma omp parallel for
        for (int i = 0; i < static_cast<int>(numBlocks); i++) {
          size_t start = i * bs;
          size_t end = std::min(start + bs, totalSize);
          size_t blockLength = end - start;

          uint8_t* dest = finalReconstructed.data() + start;
          double startTime = omp_get_wtime();
          decompressWithFastLZ1(compressedBlocks[i], dest, blockLength);
          double endTime = omp_get_wtime();
          blockDecompTimes[i] = endTime - startTime;
        }
        double decompEndOverall = omp_get_wtime();
        totalDecompTime = decompEndOverall - decompStartOverall;

        if (finalReconstructed == globalByteArray) {
          std::cout << "[INFO] Reconstructed full data matches the original data (PARALLEL)." << std::endl;
        } else {
          std::cerr << "[ERROR] Reconstructed data does NOT match the original data (PARALLEL)." << std::endl;
        }

        double compRatio = calculateCompressionRatio(totalBytes, totalCompressedSize);
        auto [compThroughput, decompThroughput] =
          calculateCompDecomThroughput(totalBytes, totalCompTime, totalDecompTime);

        // Write CSV
        file << recordIndex++ << ";" << datasetName << ";" << numThreads << ";" << bs << ";" << "N/A" << ";"
             << "Chunked_parallel" << ";" << compRatio << ";" << totalCompTime << ";" << totalDecompTime << ";"
             << compThroughput << ";" << decompThroughput << ";" << rowCount << "\n";
      }

      // ------------------------------
      // B. FULL COMPRESSION WITHOUT BLOCKING (non-blocking)
      // ------------------------------
      {
        ProfilingInfo pi_full;
        pi_full.type = "Full Compression (Non-blocking)";
        std::vector<uint8_t> compressedData, decompressedData;

        auto start = std::chrono::high_resolution_clock::now();
        size_t compressedSize = compressWithFastLZ1(globalByteArray.data(), globalByteArray.size(), compressedData);
        auto end = std::chrono::high_resolution_clock::now();
        pi_full.total_time_compressed = std::chrono::duration<double>(end - start).count();

        decompressedData.resize(globalByteArray.size());

        start = std::chrono::high_resolution_clock::now();
        decompressWithFastLZ1(compressedData, decompressedData.data(), globalByteArray.size());
        end = std::chrono::high_resolution_clock::now();
        pi_full.total_time_decompressed = std::chrono::duration<double>(end - start).count();

        double compRatio = calculateCompressionRatio(totalBytes, compressedSize);
        auto [compThroughput, decompThroughput] = calculateCompDecomThroughput(
            totalBytes, pi_full.total_time_compressed, pi_full.total_time_decompressed);

        file << recordIndex++ << ";" << datasetName << ";" << numThreads << ";" << "N/A" << ";" << "N/A" << ";"
             << "Full" << ";" << compRatio << ";" << pi_full.total_time_compressed << ";" << pi_full.total_time_decompressed << ";"
             << compThroughput << ";" << decompThroughput << ";" << rowCount << "\n";
      }

      // ------------------------------
      // A.Decompose COMPRESSION WITH BLOCKING - PARALLEL (for various block sizes)
      // ------------------------------
      auto componentConfigurationsList = getComponentConfigurationsForDataset(datasetName);
      for (const auto& componentConfig : componentConfigurationsList) {
        std::string configStr = configToString1(componentConfig);
        std::cout << "\nConfig with " << componentConfig.size()
                  << " sub-config(s): " << configStr << "\n";
        for (size_t subIdx = 0; subIdx < componentConfig.size(); ++subIdx) {
          std::cout << "  Sub-configuration " << subIdx << ": ";
          for (const auto& s : componentConfig[subIdx]) {
            std::cout << s << " ";
          }
          std::cout << std::endl;
        }

        for (size_t bs : blockSizes) {
          std::cout << "Testing with block size = " << bs << " bytes." << std::endl;

          size_t totalSize = globalByteArray.size();
          size_t numBlocks = (totalSize + bs - 1) / bs;

          struct BlockView {
            const uint8_t* data;
            size_t size;
          };
          std::vector<BlockView> fullBlocks;
          fullBlocks.reserve(numBlocks);
          for (size_t i = 0; i < numBlocks; i++) {
            size_t start = i * bs;
            size_t end = std::min(start + bs, totalSize);
            fullBlocks.push_back({ globalByteArray.data() + start, end - start });
          }

          size_t totalCompressedSize = 0;
          double totalCompTime = 0.0, totalDecompTime = 0.0;
          std::vector<std::vector<std::vector<uint8_t>>> compressedBlocks(numBlocks);
          std::vector<double> blockCompTimes(numBlocks, 0.0);
          std::vector<double> blockDecompTimes(numBlocks, 0.0);
          std::vector<size_t> blockCompressedSizes(numBlocks, 0);
          ProfilingInfo pi_parallel;
          pi_parallel.config_string = configStr;

          omp_set_num_threads(numThreads);

          auto compStartOverall = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
          for (int i = 0; i < static_cast<int>(numBlocks); i++) {
            auto start = std::chrono::high_resolution_clock::now();
            double cSize = fastlzFusedDecomposedParallel(
                fullBlocks[i].data, fullBlocks[i].size,
                pi_parallel, compressedBlocks[i],
                componentConfig, 4);
            auto end = std::chrono::high_resolution_clock::now();
            blockCompTimes[i] = std::chrono::duration<double>(end - start).count();
            blockCompressedSizes[i] = cSize;
            totalCompressedSize += cSize;
          }
          auto compEndOverall = std::chrono::high_resolution_clock::now();
          totalCompTime = std::chrono::duration<double>(compEndOverall - compStartOverall).count();

          std::vector<uint8_t> fullReconstructed(totalSize);
          double decompStartOverall = omp_get_wtime();
#pragma omp parallel for
          for (int i = 0; i < static_cast<int>(numBlocks); i++) {
            double start = omp_get_wtime();
            uint8_t* dest = fullReconstructed.data() + i * bs;
            fastlzDecomposedParallelDecompression(
                compressedBlocks[i], pi_parallel, componentConfig,
                4, fullBlocks[i].size, dest);
            double end = omp_get_wtime();
            blockDecompTimes[i] = end - start;
          }
          double decompEndOverall = omp_get_wtime();
          totalDecompTime = decompEndOverall - decompStartOverall;

          if (fullReconstructed == globalByteArray) {
            std::cout << "[INFO] Full reconstructed data matches the original data." << std::endl;
          } else {
            std::cerr << "[ERROR] Full reconstructed data does NOT match the original data." << std::endl;
          }

          double compRatio = calculateCompressionRatio(totalBytes, totalCompressedSize);
          auto [compThroughput, decompThroughput] = calculateCompDecomThroughput(
              totalBytes, totalCompTime, totalDecompTime);

          file << recordIndex++ << ";" << datasetName << ";" << numThreads << ";" << bs << ";" << configStr << ";"
               << "chunked_decomposed_parallel" << ";" << compRatio << ";" << totalCompTime << ";" << totalDecompTime << ";"
               << compThroughput << ";" << decompThroughput << ";" << rowCount << "\n";
        }
      }


        // ------------------------------
        // C. DECOMPOSE THEN CHUNK COMPRESSION - PARALLEL
        // (Reversed order: first decompose full data, then chunk each component using the block size as chunk size)
        // ------------------------------
        //auto componentConfigurationsList = getComponentConfigurationsForDataset(datasetName);
        for (const auto& componentConfig : componentConfigurationsList) {
          std::string configStr = configToString1(componentConfig);
          std::cout << "\nConfig with " << componentConfig.size()
                    << " sub-config(s): " << configStr << "\n";
          for (size_t bs : blockSizes) {
            std::cout << "Testing with chunk block size = " << bs << " bytes." << std::endl;

            size_t totalSize = globalByteArray.size();
            ProfilingInfo pi_chunk;
            pi_chunk.config_string = configStr;

            // Compressed output will be a 3D vector: per component, per chunk, per compressed data.
            std::vector<std::vector<std::vector<uint8_t>>> compressedBlocks;

            omp_set_num_threads(numThreads);
            double compStartOverall = omp_get_wtime();
            // Call the new function that first decomposes and then chunks/compresses.
            size_t totalCompressedSize = fastlzDecomposedThenChunkedParallelCompression(
                globalByteArray.data(), globalByteArray.size(),
                pi_chunk,
                compressedBlocks,
                componentConfig,
                numThreads,
                bs  // use the block size as the chunk block size
            );
            double compEndOverall = omp_get_wtime();
            pi_chunk.total_time_compressed = compEndOverall - compStartOverall;

            // Now decompress using the corresponding new function.
            std::vector<uint8_t> finalReconstructed(totalSize);
            double decompStartOverall = omp_get_wtime();
            fastlzDecomposedThenChunkedParallelDecompression(
                compressedBlocks,
                pi_chunk,
                componentConfig,
                numThreads,
                globalByteArray.size(),  // original full data size
                bs,                      // the same chunk block size used during compression
                finalReconstructed.data()
            );
            double decompEndOverall = omp_get_wtime();
            pi_chunk.total_time_decompressed = decompEndOverall - decompStartOverall;

            if (finalReconstructed == globalByteArray)
              std::cout << "[INFO] Final reconstructed data matches the original data." << std::endl;
            else
              std::cerr << "[ERROR] Final reconstructed data does NOT match the original data." << std::endl;

            double compRatio = calculateCompressionRatio(totalSize, totalCompressedSize);
            auto [compThroughput, decompThroughput] = calculateCompDecomThroughput(
                totalSize, pi_chunk.total_time_compressed, pi_chunk.total_time_decompressed);

            file << recordIndex++ << ";" << datasetName << ";" << numThreads << ";" << bs << ";" << configStr << ";"
                 << "Decompose_Chunk_Parallel" << ";" << compRatio << ";" << pi_chunk.total_time_compressed << ";" << pi_chunk.total_time_decompressed << ";"
                 << compThroughput << ";" << decompThroughput << ";" << rowCount << "\n";
          }
        }
      }
    }

    file.close();
    std::cout << "Profiling results saved to " << outputCSV << "\n";
    return 0;
  }


