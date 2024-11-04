//
// Created by jamalids on 04/11/24.
//
// Created by jamalids on 31/10/24.
#include "zstd_parallel.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cxxopts.hpp>



int main(int argc, char* argv[]) {
    cxxopts::Options options("DataCompressor", "Compress datasets and profile the compression");
    options.add_options()
        ("d,dataset", "Path to the UCR dataset tsv/csv", cxxopts::value<std::string>())
        ("o,outcsv", "Output CSV file path", cxxopts::value<std::string>()->default_value("./log_out.csv"));
        //("t,nthreads", "Number of threads to use", cxxopts::value<int>()->default_value("1"))
       // ("m,mode", "Run mode", cxxopts::value<std::string>()->default_value("signal"))
      //  ("h,help", "Print help");

    auto result = options.parse(argc, argv);

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    std::string datasetPath = result["dataset"].as<std::string>();
    std::string outputCSV = result["outcsv"].as<std::string>();
    //int numThreads = result["nthreads"].as<int>();
    //std::string mode = result["mode"].as<std::string>();

    std::vector<float> floatArray = loadTSVDataset(datasetPath);
    if (floatArray.empty()) {
        std::cerr << "Failed to load dataset from " << datasetPath << std::endl;
        return 1;
    }

    std::vector<uint8_t> globalByteArray = convertFloatToBytes(floatArray);

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
        compressedSize = zstdCompression(globalByteArray, pi_full, compressedData);
        auto end = std::chrono::high_resolution_clock::now();
        pi_full.total_time_compressed = std::chrono::duration<double>(end - start).count();

        start = std::chrono::high_resolution_clock::now();
        zstdDecompression(compressedData, decompressedData, pi_full);
        end = std::chrono::high_resolution_clock::now();
        pi_full.total_time_decompressed = std::chrono::duration<double>(end - start).count();
        pi_full.com_ratio = calculateCompressionRatio(globalByteArray.size(), compressedSize);
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
        pi_array.push_back(pi_seq);

        // Parallel operations
        start = std::chrono::high_resolution_clock::now();
        compressedSize = zstdDecomposedParallel(globalByteArray, pi_parallel, compressedLeading, compressedContent, compressedTrailing, leadingBytes, contentBytes, trailingBytes);
        end = std::chrono::high_resolution_clock::now();
        pi_parallel.total_time_compressed = std::chrono::duration<double>(end - start).count();
        start = std::chrono::high_resolution_clock::now();
        zstdDecomposedParallelDecompression(compressedLeading, compressedContent, compressedTrailing, pi_parallel, leadingBytes, contentBytes, trailingBytes);
        end = std::chrono::high_resolution_clock::now();
        pi_parallel.total_time_decompressed = std::chrono::duration<double>(end - start).count();
        pi_parallel.com_ratio = calculateCompressionRatio(globalByteArray.size(), compressedSize);
        pi_array.push_back(pi_parallel);
    }

    std::ofstream file(outputCSV);
    if (!file) {
        std::cerr << "Failed to open the file for writing: " << outputCSV << std::endl;
        return 1;
    }
  for (size_t i = 0; i < pi_array.size(); ++i) {
    pi_array[i].printCSV(file, i / 3 + 1); // Correct iteration numbering for three tests per iteration
  }



  file.close();
  return 0;
  }


