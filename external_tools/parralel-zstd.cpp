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
////////////////////////////////Entropy///////////////////////
  std::vector<uint8_t> leading, content, trailing;
  splitBytesIntoComponents(globalByteArray, leading, content, trailing, leadingBytes, contentBytes, trailingBytes);
  double leadingEntropy = calculateEntropy(leading,8);
  double contentEntropy = calculateEntropy(content,16);
  double trailingEntropy = calculateEntropy(trailing,8);
  double totalEntropy = calculateEntropy(globalByteArray,32);
  // double correlation_all = calculateCorrelation(globalByteArray,32);
  // std::cout << "Correlation_all: " << correlation_all << std::endl;
  // double correlation_leading = calculateCorrelation(leading,8);
  // std::cout << "Correlation_leading: " << correlation_leading << std::endl;
  // double correlation_content = calculateCorrelation(content,16);
  // std::cout << "Correlation_content: " << correlation_content << std::endl;
  // double correlation_trailing = calculateCorrelation(trailing,8);
  // std::cout << "Correlation_trailing: " << correlation_trailing << std::endl;


  // // Measure and print distribution for the entire dataset
  // auto dis_all = measurePatternDistribution(globalByteArray, 32);
  // std::cout << "Distribution (all):" << std::endl;
  // for (const auto& pair : dis_all) {
  //   std::cout << pair.first << ": " << pair.second << std::endl;
  // }
  //
  // // Measure and print distribution for the leading part
  // auto dist_leading = measurePatternDistribution(leading, 8);
  // std::cout << "Distribution (leading):" << std::endl;
  // for (const auto& pair : dist_leading) {
  //   std::cout << pair.first << ": " << pair.second << std::endl;
  // }
  //
  // // Measure and print distribution for the content part
  // auto dist_content = measurePatternDistribution(content, 16);
  // std::cout << "Distribution (content):" << std::endl;
  // for (const auto& pair : dist_content) {
  //   std::cout << pair.first << ": " << pair.second << std::endl;
  // }
  //
  // // Measure and print distribution for the trailing part
  // auto dist_trailing = measurePatternDistribution(trailing, 8);
  // std::cout << "Distribution (trailing):" << std::endl;
  // for (const auto& pair : dist_trailing) {
  //   std::cout << pair.first << ": " << pair.second << std::endl;
  // }
  //
  //
  // Calculate entropy for expanded data
  double expandedEntropy = calculateExpandedEntropy(globalByteArray, leadingBytes, contentBytes, trailingBytes);

  // Print the result
  std::cout << "Entropy for Full Data with Expanded Symbols: " << expandedEntropy << std::endl;


    int num_iter = 1;
    std::vector<ProfilingInfo> pi_array;
    double compressedSize;

    // Profiling setup



    for (int i = 0; i < num_iter; i++) {
      // Full compression and decompression without decomposition
      ProfilingInfo pi_full;
      std::vector<uint8_t> compressedData, decompressedData;
      auto start = std::chrono::high_resolution_clock::now();
      compressedSize = zstdCompression(globalByteArray, pi_full, compressedData);
      auto end = std::chrono::high_resolution_clock::now();
      pi_full.total_time_compressed = std::chrono::duration<double>(end - start).count();

      start = std::chrono::high_resolution_clock::now();
      zstdDecompression(compressedData, decompressedData, pi_full);
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
      pi_full.total_entropy=totalEntropy;
      pi_full.leading_entropy=leadingEntropy;
      pi_full.content_entropy=contentEntropy;
      pi_full.trailing_entropy=trailingEntropy;

      pi_array.push_back(pi_full);

      // Sequential and Parallel
      ProfilingInfo pi_seq, pi_parallel;
      std::vector<uint8_t> compressedLeading, compressedContent, compressedTrailing;

      // Sequential operations

      start = std::chrono::high_resolution_clock::now();
      compressedSize = zstdDecomposedSequential(globalByteArray, pi_seq, compressedLeading, compressedContent, compressedTrailing, leadingBytes, contentBytes, trailingBytes);
      end = std::chrono::high_resolution_clock::now();
      pi_seq.total_time_compressed = std::chrono::duration<double>(end - start).count();
      start = std::chrono::high_resolution_clock::now();
      zstdDecomposedSequentialDecompression(compressedLeading, compressedContent, compressedTrailing, pi_seq, leadingBytes, contentBytes, trailingBytes);
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
      //compressedSize,compressedLeading , compressedContent , compressedTrailing = zstdDecomposedParallel(globalByteArray, pi_parallel, compressedLeading, compressedContent, compressedTrailing, leadingBytes, contentBytes, trailingBytes,numThreads);
      auto [compressedSize, compressedLeadingSize, compressedContentSize, compressedTrailingSize] =
    zstdDecomposedParallel(globalByteArray, pi_parallel, compressedLeading, compressedContent, compressedTrailing, leadingBytes, contentBytes, trailingBytes, numThreads);


      end = std::chrono::high_resolution_clock::now();
      pi_parallel.total_time_compressed = std::chrono::duration<double>(end - start).count();
      start = std::chrono::high_resolution_clock::now();
      zstdDecomposedParallelDecompression(compressedLeading, compressedContent, compressedTrailing, pi_parallel, leadingBytes, contentBytes, trailingBytes, numThreads);
      end = std::chrono::high_resolution_clock::now();
      pi_parallel.total_time_decompressed = std::chrono::duration<double>(end - start).count();
      pi_parallel.com_ratio = calculateCompressionRatio(globalByteArray.size(), compressedSize);
      pi_parallel.com_ratio = calculateCompressionRatio(globalByteArray.size(), compressedSize);

      pi_parallel.com_ratio = calculateCompressionRatio(globalByteArray.size(), compressedSize);
      pi_parallel.com_ratio_leading = calculateCompressionRatio(globalByteArray.size()/4, compressedLeadingSize);
      pi_parallel.com_ratio_content = calculateCompressionRatio(globalByteArray.size()/2, compressedContentSize);
      pi_parallel.com_ratio_trailing = calculateCompressionRatio(globalByteArray.size()/4, compressedTrailingSize);
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
  file << "Iteration,Type,CompressionRatio,CompressionRatio_leading,CompressionRatio_content,CompressionRatio_trailing,TotalTimeCompressed,TotalTimeDecompressed,LeadingTime,ContentTime,"
  "TrailingTime,compression_throughput,decompression_throughput,rowCount ,total_entropy,leading_entropy,content_entropy,trailing_entropy\n";

  for (size_t i = 0; i < pi_array.size(); ++i) {
    pi_array[i].printCSV(file, i / 3 + 1); // Correct iteration numbering for three tests per iteration
  }



    //file.close();
    return 0;
  }

// Created by samira on 11/4/24.
//