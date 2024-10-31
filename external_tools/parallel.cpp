#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <zstd.h>
#include <chrono>
#include <cstdint>
#include <omp.h>
#include "parallel.h"
#include "utils.h"



///////////////////////////////

double calculateCompressionRatio(size_t originalSize, size_t compressedSize) {
  return static_cast<double>(originalSize) / static_cast<double>(compressedSize);
}

int main(int argc, char *argv[]) {
//    auto a = argv[1];
//    auto b = argv[2];
//    std::cout << a << std::endl;
//    std::cout << b << std::endl;
// use argparse to get the file path
    // Load dataset
    argparse::ArgumentParser parser("Big Data");
    auto args = swiftware::bigdata::get_args(argc, argv, &parser);
    std::string datasetPath = args->input_file;
    //std::string datasetPath = "/home/jamalids/Documents/2D/data1/Fcbench/HPC/H/num_brain_f64.tsv";
    std::vector<float> floatArray = loadTSVDataset(datasetPath);
    if (floatArray.empty()) {
        std::cerr << "Failed to load dataset from " << datasetPath << std::endl;
        return 1;
    }

    // Convert float array to byte array
    globalByteArray = convertFloatToBytes(floatArray);

    // Profiling
    int num_iter = 5;
    std::vector<ProfilingInfo> pi_array;
    double com_ratio;
    double compressedSize;

    for (int i = 0; i < num_iter; i++) {
        // Full compression and decompression without decomposition
        ProfilingInfo pi_full;
        std::vector<uint8_t> compressedData, decompressedData;
        auto start = std::chrono::high_resolution_clock::now();
        compressedSize=zstdCompression(globalByteArray, pi_full, compressedData);
        auto end = std::chrono::high_resolution_clock::now();
        pi_full.total_time_compressed = std::chrono::duration<double>(end - start).count();
        start = std::chrono::high_resolution_clock::now();
        zstdDecompression(compressedData, decompressedData, pi_full);
        end = std::chrono::high_resolution_clock::now();
        pi_full.total_time_decompressed = std::chrono::duration<double>(end - start).count();

        pi_full.com_ratio=calculateCompressionRatio(globalByteArray.size(), compressedSize);
        pi_array.push_back(pi_full);

        // Sequential compression and decompression with decomposition
        ProfilingInfo pi_seq;
        std::vector<uint8_t> compressedLeading, compressedContent, compressedTrailing;
        start = std::chrono::high_resolution_clock::now();
        compressedSize=zstdDecomposedSequential(globalByteArray, pi_seq, compressedLeading, compressedContent, compressedTrailing);
        end = std::chrono::high_resolution_clock::now();
        pi_seq.total_time_compressed = std::chrono::duration<double>(end - start).count();
       start = std::chrono::high_resolution_clock::now();
        zstdDecomposedSequentialDecompression(compressedLeading, compressedContent, compressedTrailing, pi_seq);
       end = std::chrono::high_resolution_clock::now();
      pi_seq.total_time_decompressed = std::chrono::duration<double>(end - start).count();
      pi_seq.com_ratio=calculateCompressionRatio(globalByteArray.size(), compressedSize);
        pi_array.push_back(pi_seq);

        // Parallel compression and decompression with decomposition
        ProfilingInfo pi_parallel;
      start = std::chrono::high_resolution_clock::now();
        compressedSize=zstdDecomposedParallel(globalByteArray, pi_parallel, compressedLeading, compressedContent, compressedTrailing);
      end = std::chrono::high_resolution_clock::now();
      pi_parallel.total_time_compressed = std::chrono::duration<double>(end - start).count();
      start = std::chrono::high_resolution_clock::now();
        zstdDecomposedParallelDecompression(compressedLeading, compressedContent, compressedTrailing, pi_parallel);
       end = std::chrono::high_resolution_clock::now();
      pi_parallel.total_time_decompressed = std::chrono::duration<double>(end - start).count();
      pi_parallel.com_ratio=calculateCompressionRatio(globalByteArray.size(), compressedSize);
        pi_array.push_back(pi_parallel);
    }


    std::ofstream file("/home/jamalids/Documents/compression-part4/new1/num_brain_f64.csv");
  file << "Iteration,Type,CompressionRatio,TotalTimeCompressed,TotalTimeDecompressed,LeadingTime,ContentTime,TrailingTime\n";
  for (size_t i = 0; i < pi_array.size(); ++i) {
    pi_array[i].printCSV(file, (i / 2) + 1); // Adjust iteration numbering
  }
  file.close();

    std::cout << "Profiling completed and results saved." << std::endl;
    return 0;
}