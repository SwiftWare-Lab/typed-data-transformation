//
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
#include <vector>
#include <cxxopts.hpp>
#include <cmath>

// NOTE: Include FastLZ-based parallel header
#include "FASTLZ_PARALLEL.h"

std::vector<uint8_t> globalByteArray;

// Map to store dataset names and their multiple possible configurations
std::map<std::string, std::vector<std::vector<std::vector<size_t>>>> datasetComponentMap = {
  {"acs_wht_f32", {
          {{1,2}, {3}, {4}} ,
          {{1, 2,3}, {4}} ,

  }},
  {"g24_78_usb2_f32", {
          {{1}, {2,3}, {4}},
          {{1, 2,3}, {4}},
  }},
  {"jw_mirimage_f32", {
          {{1,2}, {3}, {4}},
          {{1, 2,3}, {4}},
  }},
  {"spitzer_irac_f32", {
          {{1,2}, {3}, {4}},
          {{1, 2,3}, {4}},
  }},
  {"turbulence_f32", {
          {{1,2}, {3}, {4}},
          {{1, 2,3}, {4}},
  }},
  {"wave_f32", {
          {{1,2}, {3}, {4}},
          {{1, 2,3}, {4}},
  }},
  {"hdr_night_f32", {
          {{1,4}, {2}, {3}},
          {{1}, {2},{3}, {4}},
          {{1,4},{2,3}},
  }},
  {"ts_gas_f32", {
          {{1,2},{3}, {4}},
  }},
  {"solar_wind_f32", {
          {{1},{4}, {2}, {3}},
          {{1}, {2,3}, {4}},
  }},
  {"tpch_lineitem_f32", {
          {{1,2,3}, {4},},
          {{1,2},{3}, {4}},
  }},
  {"tpcds_web_f32", {
          {{1,2,3}, {4},},
          {{1},{2,3}, {4}},
  }},
  {"tpcds_store_f32", {
          {{1,2,3}, {4},},
          {{1},{2,3}, {4}},
  }},
  {"tpcds_catalog_f32", {
          {{1,2,3}, {4},},
          {{1},{2,3}, {4}},
  }},
  {"citytemp_f32", {
          {{1,4}, {2,3}},
          {{1}, {2},{3}, {4}},
          {{1,2},{3},{4}}
  }},
  {"hst_wfc3_ir_f32", {
          {{1}, {2},{3}, {4}},
          {{1,2},{3},{4}}
  }},
  {"hst_wfc3_uvis_f32", {
          {{1}, {2},{3}, {4}},
          {{1,2},{3},{4}}
  }},
  {"rsim_f32", {
          {{1,2,3}, {4}},
          {{1,2},{3},{4}}
  }},
  {"astro_mhd_f64", {
          {{1,2,3,4,5,6},{7},{8}},
          {{1,2,3,4,5},{6},{7},{8}}
  }},
  {"astro_pt_f64", {
          {{1,2,3,4,5,6},{7},{8}},
          {{1,2,3,4,5},{6},{7},{8}}
  }},
  {"astro_pt_f64", {
          {{1,2,3,4,5,6},{7},{8}},
          {{1,2,3,4,5},{6},{7},{8}}
  }},
  {"jane_street_f64", {
          {{1,2,3,4,5,6},{7},{8}},
          {{3,2,5,6,4,1},{7},{8}}
  }},
  {"msg_bt_f64", {
          {{1,2,3,4,5},{6},{7},{8}},
          {{3,2,1,4,5,6},{7},{8}},
          {{3,2,1,4,5},{6},{7},{8}}
  }},
  {"num_brain_f64", {
          {{1,2,3,4,5,6},{7},{8}},
          {{3,2,4,5,1,6},{7},{8}},
  }},
  {"num_control_f64", {
          {{1,2,3,4,5,6},{7},{8}},
          {{4,5},{6,3},{1,2},{7},{8}},
          {{4,5,6,3,1,2},{7},{8}},
  }},
  {"nyc_taxi2015_f64", {
          {{7,4,6},{5},{3,2,1,8}},
          {{7,4,6,5},{3,2,1,8}},
          {{7,4,6},{5},{3,2,1},{8}},
          {{7,4},{6},{5},{3,2},{1},{8}},
  }},
  {"phone_gyro_f64", {
          {{4,6},{5},{3,2,1,7},{8}},
          {{4,6},{1},{3,2},{5},{7},{8}},
          {{6,4,3,2,1,7},{5},{8}},
  }},
  {"tpch_order_f64", {
          {{3,2,4,1},{7},{6,5},{8}},
          {{3,2,4,1,7},{6,5},{8}},
  }},
  {"tpcxbb_store_f64", {
          {{4,2,3},{1},{5},{7},{6},{8}},
          {{4,2,3,1},{5},{7,6},{8}},
          {{4,2,3,1,5},{7,6},{8}},
  }},
  {"tpcxbb_web_f64", {
          {{4,2,3},{1},{5},{7},{6},{8}},
          {{4,2,3,1},{5},{7,6},{8}},
          {{4,2,3,1,5},{7,6},{8}},
  }},
  {"wesad_chest_f64", {
          {{7,5},{6},{8,4,1},{3,2}},
          {{7,5},{6},{8,4},{1},{3,2}},
          {{7,5},{6},{8,4,1,3,2}},
  }},
  {"default", {
          {{1}, {2}, {3}, {4}}
  }}
};
/////////////////////////chunking////////////////////
// --- Helper: Split a vector into N nearly equal chunks ---
std::vector<std::vector<uint8_t>> splitIntoChunks(const std::vector<uint8_t>& data, int numChunks) {
  std::vector<std::vector<uint8_t>> chunks;
  size_t totalSize = data.size();
  size_t chunkSize = totalSize / numChunks;
  size_t remainder = totalSize % numChunks;
  size_t offset = 0;

  for (int i = 0; i < numChunks; i++) {
    // Distribute the remainder among the first few chunks.
    size_t currentSize = chunkSize + (i < remainder ? 1 : 0);
    std::vector<uint8_t> chunk(data.begin() + offset, data.begin() + offset + currentSize);
    chunks.push_back(std::move(chunk));
    offset += currentSize;
  }
  return chunks;
}

////////////////////////////////////////////////////
// Function to get all configurations for a dataset
std::vector<std::vector<std::vector<size_t>>> getComponentConfigurationsForDataset(const std::string& datasetName) {
  auto it = datasetComponentMap.find(datasetName);
  if (it != datasetComponentMap.end()) {
    return it->second; // Return all configurations for the dataset
  }
  return datasetComponentMap["default"]; // Return default configurations if not found
}

std::string extractDatasetName(const std::string& filePath) {
  // Find the last occurrence of '/' or '\\' for cross-platform support
  size_t lastSlashPos = filePath.find_last_of("/\\");
  // Get the substring after the last slash
  std::string fileName = (lastSlashPos == std::string::npos) ? filePath : filePath.substr(lastSlashPos + 1);

  // Find the last occurrence of '.' to remove the extension
  size_t lastDotPos = fileName.find_last_of('.');
  // Get the substring before the last dot
  return (lastDotPos == std::string::npos) ? fileName : fileName.substr(0, lastDotPos);
}

std::string configToString1(const std::vector<std::vector<size_t>>& config) {
  std::stringstream ss;
  ss << "{ ";
  for (size_t i = 0; i < config.size(); ++i) {
    ss << "[";
    for (size_t j = 0; j < config[i].size(); ++j) {
      ss << config[i][j];
      if (j + 1 < config[i].size()) {
        ss << " "; // space within a single sub-config
      }
    }
    ss << "]";
    // Use "- " between sub-configs, not a comma
    if (i + 1 < config.size()) {
      ss << "- ";
    }
  }
  ss << " }";
  return ss.str();
}

//---------------------------------------
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

// double64
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
  std::vector<double> doubleArray(byteArray.size() / 8);
  for (size_t i = 0; i < doubleArray.size(); i++) {
    const uint8_t* bytePtr = &byteArray[i * 8];
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

// ----------------------------------------------------------------------------
// Helper: Block a vector into blocks of a given blockSize (in bytes)
// ----------------------------------------------------------------------------
std::vector<std::vector<uint8_t>> blockData(const std::vector<uint8_t>& data, size_t blockSize) {
  std::vector<std::vector<uint8_t>> blocks;
  size_t totalSize = data.size();
  size_t numBlocks = (totalSize + blockSize - 1) / blockSize; // ceiling division
  blocks.reserve(numBlocks);
  for (size_t i = 0; i < totalSize; i += blockSize) {
    size_t end = std::min(i + blockSize, totalSize);
    blocks.push_back(std::vector<uint8_t>(data.begin() + i, data.begin() + end));
  }
  return blocks;
}
// ----------------------------------------------------------------------------
// Helper: Block each decomposed component (same idea as blockData)
// ----------------------------------------------------------------------------
std::vector<std::vector<std::vector<uint8_t>>> blockComponents(
    const std::vector<std::vector<uint8_t>>& components, size_t blockSize)
{
  // For each component, split into blocks.
  std::vector<std::vector<std::vector<uint8_t>>> blockedComponents;
  blockedComponents.resize(components.size());
  for (size_t comp = 0; comp < components.size(); comp++) {
    blockedComponents[comp] = blockData(components[comp], blockSize);
  }
  return blockedComponents;
}


/////////////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[]) {
    // 1. Parse command-line arguments using cxxopts.
    cxxopts::Options options("DataCompressorFastLZ",
                             "Compress datasets and profile the compression (FastLZ version)");
    options.add_options()
        ("d,dataset",   "Path to the dataset file", cxxopts::value<std::string>())
        ("o,outcsv",    "Output CSV file path",     cxxopts::value<std::string>())
        ("t,threads",   "Number of threads to use", cxxopts::value<int>()->default_value("10"))
        ("b,bits",      "Floating-point precision (32 or 64 bits)", cxxopts::value<int>()->default_value("64"))
        // The blocksize argument is no longer used because we run a set of block sizes.
        ("h,help",      "Print help");
    auto result = options.parse(argc, argv);
    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }
    std::string datasetPath = result["dataset"].as<std::string>();
    std::string outputCSV   = result["outcsv"].as<std::string>();
    int numThreads          = result["threads"].as<int>();
    int precisionBits       = result["bits"].as<int>();

    // 2. Load dataset into globalByteArray.
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

    // Determine the total data size.
    size_t totalBytes = globalByteArray.size();

    // For uniform CSV output we will create per-block columns up to the maximum number of blocks.
    // Here we use the smallest block size (20K) to determine the maximum number of blocks.
    const size_t minBlockSize = 20 * 1024; // 20K (20 * 1024 bytes)
    size_t maxBlocks = (totalBytes + minBlockSize - 1) / minBlockSize;

    // List of block sizes to test (expressed as multiples of 1024 bytes)
    std::vector<size_t> blockSizes = {
        20 * 1024,      // 20K
        40 * 1024,      // 40K
        60 * 1024,      // 60K
        65 *1024,
        80 * 1024,      // 80K
        100 * 1024,     // 100K
        1000 * 1024,    // 1000K
        10000 * 1024,   // 10000K
        100000 * 1024   // 100000K
    };

    // Open CSV output file.
    std::ofstream file(outputCSV);
    if (!file) {
        std::cerr << "Failed to open the file for writing: " << outputCSV << std::endl;
        return 1;
    }

    // CSV header.
    // Note: "BlockSize" (the block size used in the experiment) and overall "CompressionRatio" are included.
    // Additionally, for each potential block (up to maxBlocks), we add columns for:
    //   BlockCompTime, BlockDecompTime, BlockCompressedSize, and BlockCompRatio.
    file << "Index;DatasetName;Threads;BlockSize;ConfigString;RunType;CompressionRatio;"
         << "TotalTimeCompressed;TotalTimeDecompressed;CompressionThroughput;DecompressionThroughput;TotalValues";
    for (size_t i = 0; i < maxBlocks; i++) {
        file  << ";BlockCompRatio_" << (i + 1)
             << ";BlockCompTime_" << (i + 1)
             << ";BlockDecompTime_" << (i + 1)
             << ";BlockCompressedSize_" << (i + 1)
           ;
    }
    file << "\n";

    int recordIndex = 1;

    // ------------------------------
    // A. FULL COMPRESSION WITH BLOCKING - PARALLEL (for various block sizes)
    // ------------------------------
    for (size_t bs : blockSizes) {
        std::cout << "Testing with block size = " << bs << " bytes." << std::endl;
        // Partition data into blocks using the current block size.
        std::vector<std::vector<uint8_t>> fullBlocks = blockData(globalByteArray, bs);
        size_t totalCompressedSize = 0;
        double totalCompTime = 0.0, totalDecompTime = 0.0;
        std::vector<std::vector<uint8_t>> compressedBlocks(fullBlocks.size());
        std::vector<std::vector<uint8_t>> decompressedBlocks(fullBlocks.size());
        std::vector<double> blockCompTimes(fullBlocks.size(), 0.0);
        std::vector<double> blockDecompTimes(fullBlocks.size(), 0.0);
        std::vector<size_t> blockCompressedSizes(fullBlocks.size(), 0);

        omp_set_num_threads(numThreads);
        auto compStartOverall = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for reduction(+:totalCompressedSize)
        for (int i = 0; i < (int)fullBlocks.size(); i++) {
            auto start = std::chrono::high_resolution_clock::now();
            size_t cSize = compressWithFastLZ(fullBlocks[i], compressedBlocks[i]);
            auto end = std::chrono::high_resolution_clock::now();
            blockCompTimes[i] = std::chrono::duration<double>(end - start).count();
            blockCompressedSizes[i] = cSize; // Save the compressed size for this block.
            totalCompressedSize += cSize;
        }
        auto compEndOverall = std::chrono::high_resolution_clock::now();
        totalCompTime = std::chrono::duration<double>(compEndOverall - compStartOverall).count();

        double decompStartOverall = omp_get_wtime();
        #pragma omp parallel for
        for (int i = 0; i < (int)compressedBlocks.size(); i++) {
            double start = omp_get_wtime();
            decompressWithFastLZ(compressedBlocks[i], decompressedBlocks[i], fullBlocks[i].size());
            double end = omp_get_wtime();
            blockDecompTimes[i] = end - start;
        }
        double decompEndOverall = omp_get_wtime();
        totalDecompTime = decompEndOverall - decompStartOverall;

        double compRatio = calculateCompressionRatio(totalBytes, totalCompressedSize);
        auto [compThroughput, decompThroughput] = calculateCompDecomThroughput(totalBytes, totalCompTime, totalDecompTime);

        // Write CSV row for the parallel, blocking experiment.
        file << recordIndex++ << ";" << datasetName << ";" << numThreads << ";" << bs << ";" << "N/A" << ";"
             << "Full_Block_Parallel" << ";" << compRatio << ";" << totalCompTime << ";" << totalDecompTime << ";"
             << compThroughput << ";" << decompThroughput << ";" << rowCount;
        // Write per-block times, compressed sizes, and compression ratios.
        for (size_t i = 0; i < maxBlocks; i++) {
            if (i < fullBlocks.size()) {
                // Compute the per-block compression ratio.
                double blockRatio = (blockCompressedSizes[i] != 0) ? (double(fullBlocks[i].size()) / blockCompressedSizes[i]) : 0.0;
                file << ";" << blockCompTimes[i] << ";" << blockDecompTimes[i] << ";" << blockCompressedSizes[i] << ";" << blockRatio;
            } else {
                file << ";N/A;N/A;N/A;N/A";
            }
        }
        file << "\n";
    }

    // ------------------------------
    // C. FULL COMPRESSION WITH BLOCKING - SEQUENTIAL (for various block sizes)
    // ------------------------------
    for (size_t bs : blockSizes) {
        std::cout << "Testing sequential blocking with block size = " << bs << " bytes." << std::endl;
        std::vector<std::vector<uint8_t>> fullBlocks = blockData(globalByteArray, bs);
        size_t totalCompressedSize = 0;
        double totalCompTime = 0.0, totalDecompTime = 0.0;
        std::vector<std::vector<uint8_t>> compressedBlocks(fullBlocks.size());
        std::vector<std::vector<uint8_t>> decompressedBlocks(fullBlocks.size());
        std::vector<double> blockCompTimes(fullBlocks.size(), 0.0);
        std::vector<double> blockDecompTimes(fullBlocks.size(), 0.0);
        std::vector<size_t> blockCompressedSizes(fullBlocks.size(), 0);

        // Sequential compression
        for (size_t i = 0; i < fullBlocks.size(); i++) {
            auto start = std::chrono::high_resolution_clock::now();
            size_t cSize = compressWithFastLZ(fullBlocks[i], compressedBlocks[i]);
            auto end = std::chrono::high_resolution_clock::now();
            blockCompTimes[i] = std::chrono::duration<double>(end - start).count();
            blockCompressedSizes[i] = cSize;
            totalCompTime += blockCompTimes[i];
            totalCompressedSize += cSize;
        }

        // Sequential decompression
        for (size_t i = 0; i < compressedBlocks.size(); i++) {
            auto start = std::chrono::high_resolution_clock::now();
            decompressWithFastLZ(compressedBlocks[i], decompressedBlocks[i], fullBlocks[i].size());
            auto end = std::chrono::high_resolution_clock::now();
            blockDecompTimes[i] = std::chrono::duration<double>(end - start).count();
            totalDecompTime += blockDecompTimes[i];
        }

        double compRatio = calculateCompressionRatio(totalBytes, totalCompressedSize);
        auto [compThroughput, decompThroughput] = calculateCompDecomThroughput(totalBytes, totalCompTime, totalDecompTime);

        file << recordIndex++ << ";" << datasetName << ";" << numThreads << ";" << bs << ";" << "N/A" << ";"
             << "Full_Block_Sequential" << ";" << compRatio << ";" << totalCompTime << ";" << totalDecompTime << ";"
             << compThroughput << ";" << decompThroughput << ";" << rowCount;
        for (size_t i = 0; i < maxBlocks; i++) {
            if (i < fullBlocks.size()) {
                double blockRatio = (blockCompressedSizes[i] != 0) ? (double(fullBlocks[i].size()) / blockCompressedSizes[i]) : 0.0;
                file << ";" << blockCompTimes[i] << ";" << blockDecompTimes[i] << ";" << blockCompressedSizes[i] << ";" << blockRatio;
            } else {
                file << ";N/A;N/A;N/A;N/A";
            }
        }
        file << "\n";
    }

    // ------------------------------
    // B. FULL COMPRESSION WITHOUT BLOCKING (non-blocking)
    // ------------------------------
    {
        ProfilingInfo pi_full;
        pi_full.type = "Full";
        std::vector<uint8_t> compressedData, decompressedData;

        auto start = std::chrono::high_resolution_clock::now();
        double compressedSize = fastlzCompression(globalByteArray, pi_full, compressedData);
        auto end = std::chrono::high_resolution_clock::now();
        pi_full.total_time_compressed = std::chrono::duration<double>(end - start).count();

        start = std::chrono::high_resolution_clock::now();
        fastlzDecompression(compressedData, decompressedData, pi_full);
        end = std::chrono::high_resolution_clock::now();
        pi_full.total_time_decompressed = std::chrono::duration<double>(end - start).count();

        double compRatio = calculateCompressionRatio(totalBytes, compressedSize);
        auto [compThroughput, decompThroughput] = calculateCompDecomThroughput(totalBytes,
                                                   pi_full.total_time_compressed,
                                                   pi_full.total_time_decompressed);

        // For non-blocking experiments, mark BlockSize and all per-block fields as "N/A".
        file << recordIndex++ << ";" << datasetName << ";" << numThreads << ";" << "N/A" << ";" << "N/A" << ";"
             << "Full" << ";" << compRatio << ";" << pi_full.total_time_compressed << ";" << pi_full.total_time_decompressed << ";"
             << compThroughput << ";" << decompThroughput << ";" << rowCount;
        for (size_t i = 0; i < maxBlocks; i++) {
            file << ";N/A;N/A;N/A;N/A";
        }
        file << "\n";
    }

    file.close();
    std::cout << "Profiling results saved to " << outputCSV << "\n";
    return 0;
}