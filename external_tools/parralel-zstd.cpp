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
// Map to store dataset names and their multiple possible configurations
std::map<std::string, std::vector<std::vector<std::vector<size_t>>>> datasetComponentMap = {
  {"acs_wht_f32", {  // Multiple configurations for a single dataset
          {{1,2}, {3}, {4}} ,    // First configuration
          {{1, 2,3}, {4}} ,
            // Second configuration
  }},
{"g24_78_usb2_f32", {  // Multiple configurations for a single dataset
            {{1}, {2,3}, {4}} ,    // First configuration
            {{1, 2,3}, {4}} ,
              // Second configuration
    }},
{"jw_mirimage_f32", {  // Multiple configurations for a single dataset
                {{1,2}, {3}, {4}} ,    // First configuration
            {{1, 2,3}, {4}},
                // Second configuration
      }},
{"spitzer_irac_f32", {  // Multiple configurations for a single dataset
                  {{1,2}, {3}, {4}} ,    // First configuration
              {{1, 2,3}, {4}},
                  // Second configuration
        }},
{"turbulence_f32", {  // Multiple configurations for a single dataset
                    {{1,2}, {3}, {4}} ,    // First configuration
                {{1, 2,3}, {4}},
                    // Second configuration
          }},
{"wave_f32", {  // Multiple configurations for a single dataset
                      {{1,2}, {3}, {4}} ,    // First configuration
                  {{1, 2,3}, {4}},
                      // Second configuration
            }},
{"hdr_night_f32", {  // Multiple configurations for a single dataset
                        {{1,4}, {2}, {3}} ,    // First configuration
                    {{1}, {2},{3}, {4}},
                    {{1,4},{2,3}},

              }},
{"ts_gas_f32", { {{1,2},{3}, {4}},  }},
{"solar_wind_f32", {  {{1},{4}, {2}, {3}} ,    // First configuration
                      {{1}, {2,3}, {4}}, }},
{"tpch_lineitem_f32", {  {{1,2,3}, {4}, } ,    // First configuration
                    {{1,2},{3}, {4}}, }},
{"tpcds_web_f32", {  {{1,2,3}, {4}, } ,    // First configuration
                  {{1},{2,3}, {4}}, }},
{"tpcds_store_f32", {  {{1,2,3}, {4}, } ,    // First configuration
                  {{1},{2,3}, {4}}, }},
{"tpcds_catalog_f32", {  {{1,2,3}, {4}, } ,    // First configuration
                {{1},{2,3}, {4}}, }},
{"citytemp_f32", {  {{1,4}, {2,3}} ,    // First configuration
                      {{1}, {2},{3}, {4}},
                          {{1,2},{3},{4}}}},
{"hst_wfc3_ir_f32", {     // First configuration
                    {{1}, {2},{3}, {4}},
                        {{1,2},{3},{4}}}},
{"hst_wfc3_uvis_f32", {     // First configuration
                      {{1}, {2},{3}, {4}},
                          {{1,2},{3},{4}}}},
{"rsim_f32", {     // First configuration
                        {{1,2,3}, {4}},
                            {{1,2},{3},{4}}}},

  {"default", {               // Default fallback with one configuration
          {{1}, {2}, {3}, {4}}
  }}
};

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



//////////////////////////////////////
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
    // 1. Parse command-line arguments
    cxxopts::Options options("DataCompressor", "Compress datasets and profile the compression");
    options.add_options()
        ("d,dataset", "Path to the dataset file", cxxopts::value<std::string>())
        ("o,outcsv",  "Output CSV file path",     cxxopts::value<std::string>())
        ("t,threads", "Number of threads to use", cxxopts::value<int>()->default_value("10"))
        ("b,bits",    "Floating-point precision (32 or 64 bits)", cxxopts::value<int>()->default_value("64"))
        ("h,help",    "Print help");

    auto result = options.parse(argc, argv);
    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    std::string datasetPath = result["dataset"].as<std::string>();
    std::string outputCSV   = result["outcsv"].as<std::string>();
    int numThreads          = result["threads"].as<int>();
    int precisionBits       = result["bits"].as<int>();

    // 2. Load Dataset (Float32 or Float64) into globalByteArray
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
        std::cout << "Loaded " << rows << " rows (64-bit) with "
                  << doubleArray.size() << " total values.\n";
    }
    else if (precisionBits == 32) {
        auto [floatArray, rows] = loadTSVDataset(datasetPath);
        if (floatArray.empty()) {
            std::cerr << "Failed to load dataset from " << datasetPath << std::endl;
            return 1;
        }
        globalByteArray = convertFloatToBytes(floatArray);
        rowCount = rows;
        std::cout << "Loaded " << rows << " rows (32-bit) with "
                  << floatArray.size() << " total values.\n";

        if (areVectorsEqual(floatArray, convertBytesToFloat(globalByteArray))) {
            std::cout << "Reconstruction successful! Arrays are equal.\n";
        } else {
            std::cerr << "Reconstruction failed! Arrays differ.\n";
        }
    }
    else {
        std::cerr << "Unsupported floating-point precision: " << precisionBits
                  << ". Use 32 or 64." << std::endl;
        return 1;
    }

    // 3. Fetch All Configuration Sets
    auto componentConfigurationsList = getComponentConfigurationsForDataset(datasetName);

    // (NEW) Helper for string conversion
    auto configToString = [&](const std::vector<std::vector<size_t>>& config){
        std::stringstream ss;
        ss << "{ ";
        for (size_t i = 0; i < config.size(); i++) {
            ss << "[";
            for (size_t j = 0; j < config[i].size(); j++) {
                ss << config[i][j];
                if (j + 1 < config[i].size()) ss << " ";
            }
            ss << "]";
            if (i + 1 < config.size()) ss << ", ";
        }
        ss << " }";
        return ss.str();
    };

    std::vector<int> threadSizesList = { 8};
    int iterations = 1;

    std::vector<ProfilingInfo> pi_array;

    // 4. Prepare CSV Output
    std::ofstream file(outputCSV);
    if (!file) {
        std::cerr << "Failed to open the file for writing: " << outputCSV << std::endl;
        return 1;
    }

    // (NEW) We add an extra column for the config string
    file << "Index,DatasetName,Threads,ConfigString,RunType,CompressionRatio,"
         << "TotalTimeCompressed,TotalTimeDecompressed,"
         << "CompressionThroughput,DecompressionThroughput,TotalValues\n";

    int recordIndex = 1;

    // 5. Loop Over Each Configuration (Nested Vector)
    for (const auto& componentConfig : componentConfigurationsList) {
        // Convert config to string once
        std::string configStr = configToString1(componentConfig);

        std::cout << "\nConfig with " << componentConfig.size() << " sub-config(s): " << configStr << "\n";
        for (size_t subIdx = 0; subIdx < componentConfig.size(); ++subIdx) {
            std::cout << "  Sub-configuration " << subIdx << ": ";
            for (auto s : componentConfig[subIdx]) {
                std::cout << s << " ";
            }
            std::cout << std::endl;
        }

        for (int threads : threadSizesList) {
            std::cout << "\n  Using " << threads << " threads...\n";
            for (int iter = 0; iter < iterations; iter++) {
                std::cout << "    Iteration #" << (iter + 1) << "...\n";

                // 7.a) Full
                {
                    ProfilingInfo pi_full;
                    // (NEW) store config string
                    pi_full.config_string = configStr;

                    std::vector<uint8_t> compressedData, decompressedData;
                    auto start = std::chrono::high_resolution_clock::now();
                    double compressedSize = zstdCompression(globalByteArray, pi_full, compressedData);
                    auto end   = std::chrono::high_resolution_clock::now();
                    pi_full.total_time_compressed = std::chrono::duration<double>(end - start).count();

                    start = std::chrono::high_resolution_clock::now();
                    zstdDecompression(compressedData, decompressedData, pi_full);
                    end   = std::chrono::high_resolution_clock::now();
                    pi_full.total_time_decompressed = std::chrono::duration<double>(end - start).count();

                    pi_full.com_ratio = calculateCompressionRatio(globalByteArray.size(), compressedSize);
                    auto [compThroughput, decompThroughput] =
                        calculateCompDecomThroughput(globalByteArray.size(),
                                                     pi_full.total_time_compressed,
                                                     pi_full.total_time_decompressed);

                    pi_full.compression_throughput   = compThroughput;
                    pi_full.decompression_throughput = decompThroughput;
                    pi_full.total_values             = rowCount;
                    pi_full.type                     = "Full";
                    pi_full.thread_count             = threads;
                    pi_array.push_back(pi_full);

                    std::cout << "      Full ratio=" << pi_full.com_ratio
                              << " Tcompress=" << pi_full.total_time_compressed
                              << " Tdecompress=" << pi_full.total_time_decompressed << "\n";
                }

                // 7.b) Seq
                {
                    ProfilingInfo pi_seq;
                    pi_seq.config_string = configStr; // store config
                    std::vector<std::vector<uint8_t>> compressedComponents;

                    auto start = std::chrono::high_resolution_clock::now();
                    double compressedSize = zstdDecomposedSequential(
                                                globalByteArray,
                                                pi_seq,
                                                compressedComponents,
                                                componentConfig
                                            );
                    auto end   = std::chrono::high_resolution_clock::now();
                    pi_seq.total_time_compressed = std::chrono::duration<double>(end - start).count();

                    start = std::chrono::high_resolution_clock::now();
                    zstdDecomposedSequentialDecompression(
                        compressedComponents,
                        pi_seq,
                        componentConfig
                    );
                    end   = std::chrono::high_resolution_clock::now();
                    pi_seq.total_time_decompressed = std::chrono::duration<double>(end - start).count();

                    pi_seq.com_ratio = calculateCompressionRatio(globalByteArray.size(), compressedSize);
                    auto [compThroughput, decompThroughput] =
                        calculateCompDecomThroughput(globalByteArray.size(),
                                                     pi_seq.total_time_compressed,
                                                     pi_seq.total_time_decompressed);

                    pi_seq.compression_throughput   = compThroughput;
                    pi_seq.decompression_throughput = decompThroughput;
                    pi_seq.total_values             = rowCount;
                    pi_seq.type                     = "Sequential";
                    pi_seq.thread_count             = threads;
                    pi_array.push_back(pi_seq);

                    std::cout << "      Seq ratio=" << pi_seq.com_ratio
                              << " Tcompress=" << pi_seq.total_time_compressed
                              << " Tdecompress=" << pi_seq.total_time_decompressed << "\n";
                }

                // 7.c) Parallel
                {
                    ProfilingInfo pi_parallel;
                    pi_parallel.config_string = configStr;
                    std::vector<std::vector<uint8_t>> compressedComponents;

                    auto start = std::chrono::high_resolution_clock::now();
                    double compressedSize = zstdDecomposedParallel(
                                                globalByteArray,
                                                pi_parallel,
                                                compressedComponents,
                                                componentConfig,
                                                threads
                                            );
                    auto end   = std::chrono::high_resolution_clock::now();
                    pi_parallel.total_time_compressed = std::chrono::duration<double>(end - start).count();

                    start = std::chrono::high_resolution_clock::now();
                    zstdDecomposedParallelDecompression(
                        compressedComponents,
                        pi_parallel,
                        componentConfig,
                        threads
                    );
                    end   = std::chrono::high_resolution_clock::now();
                    pi_parallel.total_time_decompressed = std::chrono::duration<double>(end - start).count();

                    pi_parallel.com_ratio = calculateCompressionRatio(globalByteArray.size(), compressedSize);
                    auto [compThroughput, decompThroughput] =
                        calculateCompDecomThroughput(globalByteArray.size(),
                                                     pi_parallel.total_time_compressed,
                                                     pi_parallel.total_time_decompressed);

                    pi_parallel.compression_throughput   = compThroughput;
                    pi_parallel.decompression_throughput = decompThroughput;
                    pi_parallel.total_values             = rowCount;
                    pi_parallel.type                     = "Parallel";
                    pi_parallel.thread_count             = threads;
                    pi_array.push_back(pi_parallel);

                    std::cout << "      Par ratio=" << pi_parallel.com_ratio
                              << " Tcompress=" << pi_parallel.total_time_compressed
                              << " Tdecompress=" << pi_parallel.total_time_decompressed << "\n";
                }

            } // iteration
        } // threadSizesList
    } // componentConfigurationsList

    // 8. Write profiling data to CSV
    // (NEW) We added "ConfigString" as a column
    for (const auto& pi : pi_array) {
        file << recordIndex++ << ","
             << datasetName << ","
             << pi.thread_count << ","
             << pi.config_string << ","    // <-- the new column
             << pi.type << ","
             << pi.com_ratio << ","
             << pi.total_time_compressed << ","
             << pi.total_time_decompressed << ","
             << pi.compression_throughput << ","
             << pi.decompression_throughput << ","
             << pi.total_values << "\n";
    }

    file.close();
    std::cout << "Profiling results saved to " << outputCSV << "\n";
    return 0;
}
