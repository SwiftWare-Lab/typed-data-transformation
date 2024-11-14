//
// Created by jamalids on 13/11/24.
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
#include <cxxopts.hpp>

#include "zlib.h" // Use gzip for compression
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
std::pair<double, double> calculateCompDecomThroughput(size_t originalSize, double compressedTime, double decompressedTime) {
  // Convert originalSize from bytes to gigabytes
  double originalSizeGB = static_cast<double>(originalSize) / 1e9;

  // Calculate throughput in GB/s
  double compressionThroughput = originalSizeGB / static_cast<double>(compressedTime);
  double decompressionThroughput = originalSizeGB / static_cast<double>(decompressedTime);

  return {compressionThroughput, decompressionThroughput};
}

int main(int argc, char* argv[])
  {
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
    //std::string mode = result["mode"].as<std::string>();

    auto [floatArray, rowCount] = loadTSVDataset(datasetPath);
    std::cout << "Loaded " << rowCount << " rows with " << floatArray.size() << " total values." << std::endl;
    if (floatArray.empty()) {
      std::cerr << "Failed to load dataset from " << datasetPath << std::endl;
      return 1;
    }

  globalByteArray = convertFloatToBytes(floatArray);

    size_t leadingBytes = 1; // size in bytes for leading segment
    size_t contentBytes = 2; // size in bytes for content segment
    size_t trailingBytes = 1; // size in bytes for trailing segment

    int num_iter = 1;
    std::vector<ProfilingInfo> pi_array;
    double compressedSize;

    // Profiling setup



    for (int i = 0; i < num_iter; i++) {
      // Full compression and decompression without decomposition
      ProfilingInfo pi_full;
      std::vector<uint8_t> compressedData, decompressedData;
      auto start = std::chrono::high_resolution_clock::now();
      compressedSize = gzipCompression(globalByteArray, pi_full, compressedData);
      auto end = std::chrono::high_resolution_clock::now();
      pi_full.total_time_compressed = std::chrono::duration<double>(end - start).count();

      start = std::chrono::high_resolution_clock::now();
      gzipDecompression(compressedData, decompressedData, pi_full);
      end = std::chrono::high_resolution_clock::now();
      pi_full.total_time_decompressed = std::chrono::duration<double>(end - start).count();
      pi_full.com_ratio = calculateCompressionRatio(globalByteArray.size(), compressedSize);
      // Calculate compression and decompression throughput
      auto [CT, DT] = calculateCompDecomThroughput(
          globalByteArray.size(),
          pi_full.total_time_compressed,
          pi_full.total_time_decompressed
      );

      // Optionally store CT and DT in ProfilingInfo, if needed
      pi_full.compression_throughput = CT;
      pi_full.decompression_throughput = DT;
      pi_full.total_values=rowCount;

      pi_array.push_back(pi_full);

      // Sequential and Parallel
      ProfilingInfo pi_seq, pi_parallel;
      std::vector<uint8_t> compressedLeading, compressedContent, compressedTrailing;

      // Sequential operations

      start = std::chrono::high_resolution_clock::now();
      compressedSize =gzipDecomposedSequential(globalByteArray, pi_seq, compressedLeading, compressedContent, compressedTrailing, leadingBytes, contentBytes, trailingBytes);
      end = std::chrono::high_resolution_clock::now();
      pi_seq.total_time_compressed = std::chrono::duration<double>(end - start).count();
      start = std::chrono::high_resolution_clock::now();
      gzipDecomposedSequentialDecompression(compressedLeading, compressedContent, compressedTrailing, pi_seq, leadingBytes, contentBytes, trailingBytes);
      end = std::chrono::high_resolution_clock::now();
      pi_seq.total_time_decompressed = std::chrono::duration<double>(end - start).count();
      pi_seq.com_ratio = calculateCompressionRatio(globalByteArray.size(), compressedSize);

      pi_seq.com_ratio = calculateCompressionRatio(globalByteArray.size(), compressedSize);
      // Calculate compression and decompression throughput
      auto [CT_seq, DT_seq] = calculateCompDecomThroughput(
          globalByteArray.size(),
          pi_seq.total_time_compressed,
          pi_seq.total_time_decompressed
      );

      // Optionally store CT and DT in ProfilingInfo, if needed
      pi_seq.compression_throughput = CT_seq;
      pi_seq.decompression_throughput = DT_seq;
      pi_seq.total_values=rowCount;

      pi_array.push_back(pi_seq);

      // Parallel operations
      start = std::chrono::high_resolution_clock::now();
      compressedSize = gzipDecomposedParallel(globalByteArray, pi_parallel, compressedLeading, compressedContent, compressedTrailing, leadingBytes, contentBytes, trailingBytes,numThreads);
      end = std::chrono::high_resolution_clock::now();
      pi_parallel.total_time_compressed = std::chrono::duration<double>(end - start).count();
      start = std::chrono::high_resolution_clock::now();
      gzipDecomposedParallelDecompression(compressedLeading, compressedContent, compressedTrailing, pi_parallel, leadingBytes, contentBytes, trailingBytes, numThreads);
      end = std::chrono::high_resolution_clock::now();
      pi_parallel.total_time_decompressed = std::chrono::duration<double>(end - start).count();
      pi_parallel.com_ratio = calculateCompressionRatio(globalByteArray.size(), compressedSize);
      pi_parallel.com_ratio = calculateCompressionRatio(globalByteArray.size(), compressedSize);

      pi_parallel.com_ratio = calculateCompressionRatio(globalByteArray.size(), compressedSize);
      // Calculate compression and decompression throughput
      auto [CT_par, DT_par] = calculateCompDecomThroughput(
          globalByteArray.size(),
          pi_parallel.total_time_compressed,
          pi_parallel.total_time_decompressed
      );

      // Optionally store CT and DT in ProfilingInfo, if needed
      pi_parallel.compression_throughput = CT_par;
      pi_parallel.decompression_throughput = DT_par;
      pi_parallel.total_values=rowCount;
      pi_array.push_back(pi_parallel);
    }

  std::ofstream file(outputCSV);
  if (!file) {
    std::cerr << "Failed to open the file for writing: " << outputCSV << std::endl;
    return 1;
  }
  // Write the CSV header
  file << "Iteration,Type,CompressionRatio,TotalTimeCompressed,TotalTimeDecompressed,LeadingTime,ContentTime,TrailingTime,compression_throughput,decompression_throughput,rowCount\n";

  for (size_t i = 0; i < pi_array.size(); ++i) {
    pi_array[i].printCSV(file, i / 3 + 1); // Correct iteration numbering for three tests per iteration
  }



    //file.close();
    return 0;
  }