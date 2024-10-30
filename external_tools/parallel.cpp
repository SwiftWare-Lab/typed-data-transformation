#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <zstd.h>
#include <chrono>
#include <cstdint>
#include <omp.h>

// Struct for profiling information
struct ProfilingInfo {
    double com_ratio=0;
    double total_time_compressed;
    double total_time_decompressed;
    std::string type;
    double leading_time=0.0;
    double content_time=0.0;
    double trailing_time=0.0;

    // Print results to CSV format
    void printCSV(std::ofstream &file, int iteration) {
        file << iteration << ","
             << type << ","
             << com_ratio << ","
             << total_time_compressed << ","
             << total_time_decompressed << ","
             << leading_time << ","
             << content_time << ","
             << trailing_time << "\n";
    }
};

// Global variable to hold the dataset
std::vector<uint8_t> globalByteArray;

// Function declarations
std::vector<float> loadTSVDataset(const std::string& filePath, size_t maxRows = 8000000);
std::vector<uint8_t> convertFloatToBytes(const std::vector<float>& floatArray);
void splitBytesIntoComponents(const std::vector<uint8_t>& byteArray, std::vector<uint8_t>& leading, std::vector<uint8_t>& content, std::vector<uint8_t>& trailing);
size_t compressWithZstd(const std::vector<uint8_t>& data, std::vector<uint8_t>& compressedData, int compressionLevel);
size_t decompressWithZstd(const std::vector<uint8_t>& compressedData, std::vector<uint8_t>& decompressedData, size_t originalSize);
bool verifyDataMatch(const std::vector<uint8_t>& original, const std::vector<uint8_t>& reconstructed);

// Function to load a TSV dataset into a float vector
std::vector<float> loadTSVDataset(const std::string& filePath, size_t maxRows) {
    std::vector<float> floatArray;
    std::ifstream file(filePath);
    std::string line;

    if (file.is_open()) {
        size_t rowCount = 0;
        while (std::getline(file, line) && rowCount < maxRows) {
            std::stringstream ss(line);
            std::string value;
            std::getline(ss, value, '\t'); // Skip the first column

            while (std::getline(ss, value, '\t')) {
                floatArray.push_back(std::stof(value));
            }
            rowCount++;
        }
        file.close();
    } else {
        std::cerr << "Unable to open file: " << filePath << std::endl;
    }

    return floatArray;
}

// Convert a float array to a uint8_t vector
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

// Split bytes into three components: leading, content, and trailing
void splitBytesIntoComponents(const std::vector<uint8_t>& byteArray, std::vector<uint8_t>& leading, std::vector<uint8_t>& content, std::vector<uint8_t>& trailing) {
    size_t floatCount = byteArray.size() / 4;
    for (size_t i = 0; i < floatCount; i++) {
        leading.push_back(byteArray[i * 4]);
        content.push_back(byteArray[i * 4 + 1]);
        content.push_back(byteArray[i * 4 + 2]);
        trailing.push_back(byteArray[i * 4 + 3]);
    }
}
// Reconstruct data from components
void reconstructFromComponents(const std::vector<uint8_t>& leading, const std::vector<uint8_t>& content,
                               const std::vector<uint8_t>& trailing, std::vector<uint8_t>& reconstructedData) {
  size_t floatCount = leading.size();
  reconstructedData.resize(floatCount * 4);

  for (size_t i = 0; i < floatCount; i++) {
    reconstructedData[i * 4] = leading[i];
    reconstructedData[i * 4 + 1] = content[i * 2];
    reconstructedData[i * 4 + 2] = content[i * 2 + 1];
    reconstructedData[i * 4 + 3] = trailing[i];
  }
}
// Compress with Zstd
size_t compressWithZstd(const std::vector<uint8_t>& data, std::vector<uint8_t>& compressedData, int compressionLevel) {
    size_t const cBuffSize = ZSTD_compressBound(data.size());
    compressedData.resize(cBuffSize);
    size_t const cSize = ZSTD_compress(compressedData.data(), cBuffSize, data.data(), data.size(), compressionLevel);
    if (ZSTD_isError(cSize)) {
        std::cerr << "Zstd compression error: " << ZSTD_getErrorName(cSize) << std::endl;
        return 0;
    }
    compressedData.resize(cSize);
    return cSize;
}

// Decompress with Zstd
size_t decompressWithZstd(const std::vector<uint8_t>& compressedData, std::vector<uint8_t>& decompressedData, size_t originalSize) {
    decompressedData.resize(originalSize);
    size_t const dSize = ZSTD_decompress(decompressedData.data(), originalSize, compressedData.data(), compressedData.size());
    if (ZSTD_isError(dSize)) {
        std::cerr << "Zstd decompression error: " << ZSTD_getErrorName(dSize) << std::endl;
        return 0;
    }
    return dSize;
}

// Verify if original and reconstructed data match
bool verifyDataMatch(const std::vector<uint8_t>& original, const std::vector<uint8_t>& reconstructed) {
    if (original.size() != reconstructed.size()) {
        std::cerr << "Size mismatch: Original size = " << original.size() << ", Reconstructed size = " << reconstructed.size() << std::endl;
        return false;
    }

    for (size_t i = 0; i < original.size(); i++) {
        if (original[i] != reconstructed[i]) {
            std::cerr << "Data mismatch at index " << i << ": Original = " << static_cast<int>(original[i]) << ", Reconstructed = " << static_cast<int>(reconstructed[i]) << std::endl;
            return false;
        }
    }
    return true;
}

// Full compression without decomposition
size_t zstdCompression(const std::vector<uint8_t>& data, ProfilingInfo &pi, std::vector<uint8_t>& compressedData) {

    size_t compressedSize =compressWithZstd(data, compressedData, 3);

    pi.type = "Full Compression";
  return compressedSize;
}

// Full decompression without decomposition
void zstdDecompression(const std::vector<uint8_t>& compressedData, std::vector<uint8_t>& decompressedData, ProfilingInfo &pi) {

    decompressWithZstd(compressedData, decompressedData, globalByteArray.size());



    // Verify decompressed data
    if (!verifyDataMatch(globalByteArray, decompressedData)) {
        std::cerr << "Error: Decompressed data doesn't match the original." << std::endl;
   }
}

// Sequential compression with decomposition
size_t zstdDecomposedSequential(const std::vector<uint8_t>& data, ProfilingInfo &pi, std::vector<uint8_t>& compressedLeading, std::vector<uint8_t>& compressedContent, std::vector<uint8_t>& compressedTrailing) {
    std::vector<uint8_t> leading, content, trailing;
    splitBytesIntoComponents(data, leading, content, trailing);

    auto start = std::chrono::high_resolution_clock::now();
    size_t compressedSize_l=compressWithZstd(leading, compressedLeading, 3);
    auto end = std::chrono::high_resolution_clock::now();
    pi.leading_time = std::chrono::duration<double>(end - start).count();

    start = std::chrono::high_resolution_clock::now();
    size_t compressedSize_c=compressWithZstd(content, compressedContent, 3);
    end = std::chrono::high_resolution_clock::now();
    pi.content_time = std::chrono::duration<double>(end - start).count();

    start = std::chrono::high_resolution_clock::now();
    size_t compressedSize_t=compressWithZstd(trailing, compressedTrailing, 3);
    end = std::chrono::high_resolution_clock::now();
    pi.trailing_time = std::chrono::duration<double>(end - start).count();

   // pi.total_time_compressed = pi.leading_time + pi.content_time + pi.trailing_time;
    pi.type = "Sequential Decomposition";
  return(compressedSize_l+compressedSize_c+compressedSize_t);
}

// Sequential decompression with decomposition
void zstdDecomposedSequentialDecompression(const std::vector<uint8_t>& compressedLeading, const std::vector<uint8_t>& compressedContent, const std::vector<uint8_t>& compressedTrailing, ProfilingInfo &pi) {
    std::vector<uint8_t> decompressedLeading(globalByteArray.size() / 4);
    std::vector<uint8_t> decompressedContent(globalByteArray.size() / 2);
    std::vector<uint8_t> decompressedTrailing(globalByteArray.size() / 4);
    std::vector<uint8_t> reconstructedData;

    auto start = std::chrono::high_resolution_clock::now();
    decompressWithZstd(compressedLeading, decompressedLeading, decompressedLeading.size());
    auto end = std::chrono::high_resolution_clock::now();
    pi.leading_time = std::chrono::duration<double>(end - start).count();

    start = std::chrono::high_resolution_clock::now();
    decompressWithZstd(compressedContent, decompressedContent, decompressedContent.size());
    end = std::chrono::high_resolution_clock::now();
    pi.content_time = std::chrono::duration<double>(end - start).count();

    start = std::chrono::high_resolution_clock::now();
    decompressWithZstd(compressedTrailing, decompressedTrailing, decompressedTrailing.size());
    end = std::chrono::high_resolution_clock::now();
    pi.trailing_time = std::chrono::duration<double>(end - start).count();



    // Reconstruct decompressed data
     reconstructFromComponents(decompressedLeading, decompressedContent, decompressedTrailing, reconstructedData);
    // Verify decompressed data
    if (!verifyDataMatch(globalByteArray, reconstructedData)) {
        std::cerr << "Error: Decompressed data doesn't match the original." << std::endl;
    }
}

// Parallel compression with decomposition
size_t zstdDecomposedParallel(const std::vector<uint8_t>& data, ProfilingInfo &pi, std::vector<uint8_t>& compressedLeading, std::vector<uint8_t>& compressedContent, std::vector<uint8_t>& compressedTrailing) {
    std::vector<uint8_t> leading, content, trailing;
    splitBytesIntoComponents(data, leading, content, trailing);
    double compressedSize_l=0, compressedSize_c=0,compressedSize_t=0;

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            auto start = std::chrono::high_resolution_clock::now();
            size_t compressedSize_l=compressWithZstd(leading, compressedLeading, 3);
            auto end = std::chrono::high_resolution_clock::now();
            pi.leading_time = std::chrono::duration<double>(end - start).count();
        }

        #pragma omp section
        {
            auto start = std::chrono::high_resolution_clock::now();
            size_t compressedSize_c=compressWithZstd(content, compressedContent, 3);
            auto end = std::chrono::high_resolution_clock::now();
            pi.content_time = std::chrono::duration<double>(end - start).count();
        }

        #pragma omp section
        {
            auto start = std::chrono::high_resolution_clock::now();
            size_t compressedSize_t=compressWithZstd(trailing, compressedTrailing, 3);
            auto end = std::chrono::high_resolution_clock::now();
            pi.trailing_time = std::chrono::duration<double>(end - start).count();
        }
    }


    pi.type = "Parallel Decomposition";
  return( compressedLeading.size() + compressedContent.size() + compressedTrailing.size());
}

// Parallel decompression with decomposition
void zstdDecomposedParallelDecompression(const std::vector<uint8_t>& compressedLeading, const std::vector<uint8_t>& compressedContent, const std::vector<uint8_t>& compressedTrailing, ProfilingInfo &pi) {
    std::vector<uint8_t> decompressedLeading(globalByteArray.size() / 4);
    std::vector<uint8_t> decompressedContent(globalByteArray.size() / 2);
    std::vector<uint8_t> decompressedTrailing(globalByteArray.size() / 4);
    std::vector<uint8_t> reconstructedData;

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            auto start = std::chrono::high_resolution_clock::now();
            decompressWithZstd(compressedLeading, decompressedLeading, decompressedLeading.size());
            auto end = std::chrono::high_resolution_clock::now();
            pi.leading_time = std::chrono::duration<double>(end - start).count();
        }

        #pragma omp section
        {
            auto start = std::chrono::high_resolution_clock::now();
            decompressWithZstd(compressedContent, decompressedContent, decompressedContent.size());
            auto end = std::chrono::high_resolution_clock::now();
            pi.content_time = std::chrono::duration<double>(end - start).count();
        }

        #pragma omp section
        {
            auto start = std::chrono::high_resolution_clock::now();
            decompressWithZstd(compressedTrailing, decompressedTrailing, decompressedTrailing.size());
            auto end = std::chrono::high_resolution_clock::now();
            pi.trailing_time = std::chrono::duration<double>(end - start).count();
        }
    }




  // Reconstruct decompressed data
  reconstructFromComponents(decompressedLeading, decompressedContent, decompressedTrailing, reconstructedData);
  //Verify decompressed data
  if (!verifyDataMatch(globalByteArray, reconstructedData)) {
    std::cerr << "Error: Decompressed data doesn't match the original." << std::endl;
  }

}
double calculateCompressionRatio(size_t originalSize, size_t compressedSize) {
  return static_cast<double>(originalSize) / static_cast<double>(compressedSize);
}
// Main function
int main() {
    std::string datasetPath = "/home/jamalids/Documents/2D/data1/Fcbench/HPC/H/num_brain_f64.tsv";
    std::vector<float> floatArray = loadTSVDataset(datasetPath);
    if (floatArray.empty()) {
        std::cerr << "Failed to load dataset from " << datasetPath << std::endl;
        return 1;
    }

    // Convert float array to byte array
    globalByteArray = convertFloatToBytes(floatArray);

    // Profiling for multiple iterations
    int num_iter = 10;
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

    // Save results to CSV
    std::ofstream file("/home/jamalids/Documents/compression-part4/new1/pp.csv");
  file << "Iteration,Type,CompressionRatio,TotalTimeCompressed,TotalTimeDecompressed,LeadingTime,ContentTime,TrailingTime\n";
  for (size_t i = 0; i < pi_array.size(); ++i) {
    pi_array[i].printCSV(file, (i / 2) + 1); // Adjust iteration numbering
  }
  file.close();

    std::cout << "Profiling completed and results saved." << std::endl;
    return 0;
}