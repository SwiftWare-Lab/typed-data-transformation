// Created by jamalids on 31/10/24.
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

struct ProfilingInfo {
    double com_ratio = 0.0;
    double total_time_compressed = 0.0;
    double total_time_decompressed = 0.0;
    std::string type;
    double leading_time = 0.0;
    double content_time = 0.0;
    double trailing_time = 0.0;
    size_t leading_bytes = 0;
    size_t content_bytes = 0;
    size_t trailing_bytes = 0;

    void printCSV(std::ofstream &file, int iteration) {
        file << iteration << ","
             << type << ","
             << com_ratio << ","
             << total_time_compressed << ","
             << total_time_decompressed << ","
             << leading_time << ","
             << content_time << ","
             << trailing_time << ","
             << leading_bytes << ","
             << content_bytes << ","
             << trailing_bytes << "\n";
    }
};

std::vector<uint8_t> globalByteArray;

std::vector<float> loadTSVDataset(const std::string& filePath, size_t maxRows = 8000000) {
    std::vector<float> floatArray;
    std::ifstream file(filePath);
    std::string line;

    if (file.is_open()) {
        size_t rowCount = 0;
        while (std::getline(file, line) && rowCount < maxRows) {
            std::stringstream ss(line);
            std::string value;
            std::getline(ss, value, '\t'); // Assume first column is not needed

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

void splitBytesIntoComponents1(const std::vector<uint8_t>& byteArray, std::vector<uint8_t>& leading, std::vector<uint8_t>& content, std::vector<uint8_t>& trailing, size_t leadingBytes, size_t contentBytes, size_t trailingBytes) {
    size_t numElements = byteArray.size() / (leadingBytes + contentBytes + trailingBytes);
    leading.clear();
    content.clear();
    trailing.clear();

    for (size_t i = 0; i < numElements; ++i) {
        size_t base = i * (leadingBytes + contentBytes + trailingBytes);
        leading.insert(leading.end(), byteArray.begin() + base, byteArray.begin() + base + leadingBytes);
        content.insert(content.end(), byteArray.begin() + base + leadingBytes, byteArray.begin() + base + leadingBytes + contentBytes);
        trailing.insert(trailing.end(), byteArray.begin() + base + leadingBytes + contentBytes, byteArray.begin() + base + leadingBytes + contentBytes + trailingBytes);
    }
}

void splitBytesIntoComponents(const std::vector<uint8_t>& byteArray,
                              std::vector<uint8_t>& leading,
                              std::vector<uint8_t>& content,
                              std::vector<uint8_t>& trailing,
                              size_t leadingBytes,
                              size_t contentBytes,
                              size_t trailingBytes) {
    size_t numElements = byteArray.size() / (leadingBytes + contentBytes + trailingBytes);

    // Resize the vectors to accommodate the exact number of bytes
    leading.resize(numElements * leadingBytes);
    content.resize(numElements * contentBytes);
    trailing.resize(numElements * trailingBytes);

    // Parallel loop to split the byteArray into components
#pragma omp parallel for
    for (size_t i = 0; i < numElements; ++i) {
        size_t base = i * (leadingBytes + contentBytes + trailingBytes);
        memcpy(&leading[i * leadingBytes], &byteArray[base], leadingBytes);
        memcpy(&content[i * contentBytes], &byteArray[base + leadingBytes], contentBytes);
        memcpy(&trailing[i * trailingBytes], &byteArray[base + leadingBytes + contentBytes], trailingBytes);
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
// Full compression without decomposition
size_t zstdCompression(const std::vector<uint8_t>& data, ProfilingInfo &pi, std::vector<uint8_t>& compressedData) {

    size_t compressedSize = compressWithZstd(data, compressedData, 3);

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
// Sequential compression with decomposition that takes dynamic byte sizes as parameters
size_t zstdDecomposedSequential(const std::vector<uint8_t>& data, ProfilingInfo &pi,
                                std::vector<uint8_t>& compressedLeading,
                                std::vector<uint8_t>& compressedContent,
                                std::vector<uint8_t>& compressedTrailing,
                                size_t leadingBytes, size_t contentBytes, size_t trailingBytes) {
    std::vector<uint8_t> leading, content, trailing;
    splitBytesIntoComponents(data, leading, content, trailing, leadingBytes, contentBytes, trailingBytes);

    auto start = std::chrono::high_resolution_clock::now();
    size_t compressedSize_l = compressWithZstd(leading, compressedLeading, 3);
    auto end = std::chrono::high_resolution_clock::now();
    pi.leading_time = std::chrono::duration<double>(end - start).count();

    start = std::chrono::high_resolution_clock::now();
    size_t compressedSize_c = compressWithZstd(content, compressedContent, 3);
    end = std::chrono::high_resolution_clock::now();
    pi.content_time = std::chrono::duration<double>(end - start).count();

    start = std::chrono::high_resolution_clock::now();
    size_t compressedSize_t = compressWithZstd(trailing, compressedTrailing, 3);
    end = std::chrono::high_resolution_clock::now();
    pi.trailing_time = std::chrono::duration<double>(end - start).count();

    pi.type = "Sequential Decomposition";
    return compressedSize_l + compressedSize_c + compressedSize_t;
}

// Sequential decompression with decomposition that  takes dynamic byte sizes as parameters
void zstdDecomposedSequentialDecompression(const std::vector<uint8_t>& compressedLeading,
                                           const std::vector<uint8_t>& compressedContent,
                                           const std::vector<uint8_t>& compressedTrailing,
                                           ProfilingInfo &pi,
                                           size_t leadingBytes, size_t contentBytes, size_t trailingBytes) {
    size_t totalSize = globalByteArray.size();  // Ensure globalByteArray is appropriately defined and accessible
    size_t floatCount = totalSize / (leadingBytes + contentBytes + trailingBytes);

    std::vector<uint8_t> reconstructedData(totalSize);

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<uint8_t> tempLeading(floatCount * leadingBytes);
    decompressWithZstd(compressedLeading, tempLeading, floatCount * leadingBytes);
    for (size_t i = 0; i < floatCount; ++i) {
        for (size_t j = 0; j < leadingBytes; ++j) {
            reconstructedData[i * (leadingBytes + contentBytes + trailingBytes) + j] = tempLeading[i * leadingBytes + j];
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    pi.leading_time = std::chrono::duration<double>(end - start).count();

    start = std::chrono::high_resolution_clock::now();
    std::vector<uint8_t> tempContent(floatCount * contentBytes);
    decompressWithZstd(compressedContent, tempContent, floatCount * contentBytes);
    for (size_t i = 0; i < floatCount; ++i) {
        for (size_t j = 0; j < contentBytes; ++j) {
            reconstructedData[i * (leadingBytes + contentBytes + trailingBytes) + leadingBytes + j] = tempContent[i * contentBytes + j];
        }
    }
    end = std::chrono::high_resolution_clock::now();
    pi.content_time = std::chrono::duration<double>(end - start).count();

    start = std::chrono::high_resolution_clock::now();
    std::vector<uint8_t> tempTrailing(floatCount * trailingBytes);
    decompressWithZstd(compressedTrailing, tempTrailing, floatCount * trailingBytes);
    for (size_t i = 0; i < floatCount; ++i) {
        for (size_t j = 0; j < trailingBytes; ++j) {
            reconstructedData[i * (leadingBytes + contentBytes + trailingBytes) + leadingBytes + contentBytes + j] = tempTrailing[i * trailingBytes + j];
        }
    }
    end = std::chrono::high_resolution_clock::now();
    pi.trailing_time = std::chrono::duration<double>(end - start).count();

    // Verify decompressed data
    if (!verifyDataMatch(globalByteArray, reconstructedData)) {
        std::cerr << "Error: Decompressed data doesn't match the original." << std::endl;
    }
}
// Parallel compression with decomposition that takes dynamic byte sizes as parameters
size_t zstdDecomposedParallel(const std::vector<uint8_t>& data, ProfilingInfo &pi,
                              std::vector<uint8_t>& compressedLeading,
                              std::vector<uint8_t>& compressedContent,
                              std::vector<uint8_t>& compressedTrailing,
                              size_t leadingBytes, size_t contentBytes, size_t trailingBytes) {
    std::vector<uint8_t> leading, content, trailing;
    splitBytesIntoComponents(data, leading, content, trailing, leadingBytes, contentBytes, trailingBytes);

    double compressedSize_l = 0, compressedSize_c = 0, compressedSize_t = 0;

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            auto start = std::chrono::high_resolution_clock::now();
            compressedSize_l = compressWithZstd(leading, compressedLeading, 3);
            auto end = std::chrono::high_resolution_clock::now();
            pi.leading_time = std::chrono::duration<double>(end - start).count();
        }

        #pragma omp section
        {
            auto start = std::chrono::high_resolution_clock::now();
            compressedSize_c = compressWithZstd(content, compressedContent, 3);
            auto end = std::chrono::high_resolution_clock::now();
            pi.content_time = std::chrono::duration<double>(end - start).count();
        }

        #pragma omp section
        {
            auto start = std::chrono::high_resolution_clock::now();
            compressedSize_t = compressWithZstd(trailing, compressedTrailing, 3);
            auto end = std::chrono::high_resolution_clock::now();
            pi.trailing_time = std::chrono::duration<double>(end - start).count();
        }
    }

    pi.type = "Parallel Decomposition";
    return (compressedLeading.size() + compressedContent.size() + compressedTrailing.size());
}

// Parallel decompression with decomposition that  takes dynamic byte sizes as parameters
void zstdDecomposedParallelDecompression(const std::vector<uint8_t>& compressedLeading,
                                         const std::vector<uint8_t>& compressedContent,
                                         const std::vector<uint8_t>& compressedTrailing,
                                         ProfilingInfo &pi,
                                         size_t leadingBytes, size_t contentBytes, size_t trailingBytes) {
    size_t totalSize = globalByteArray.size();
    size_t floatCount = totalSize / (leadingBytes + contentBytes + trailingBytes);
    std::vector<uint8_t> reconstructedData(totalSize);

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            auto start = std::chrono::high_resolution_clock::now();
            std::vector<uint8_t> tempLeading(floatCount * leadingBytes);
            decompressWithZstd(compressedLeading, tempLeading, floatCount * leadingBytes);
            for (size_t i = 0; i < floatCount; ++i) {
                memcpy(&reconstructedData[i * (leadingBytes + contentBytes + trailingBytes)], &tempLeading[i * leadingBytes], leadingBytes);
            }
            auto end = std::chrono::high_resolution_clock::now();
            pi.leading_time = std::chrono::duration<double>(end - start).count();
        }

        #pragma omp section
        {
            auto start = std::chrono::high_resolution_clock::now();
            std::vector<uint8_t> tempContent(floatCount * contentBytes);
            decompressWithZstd(compressedContent, tempContent, floatCount * contentBytes);
            for (size_t i = 0; i < floatCount; ++i) {
                memcpy(&reconstructedData[i * (leadingBytes + contentBytes + trailingBytes) + leadingBytes], &tempContent[i * contentBytes], contentBytes);
            }
            auto end = std::chrono::high_resolution_clock::now();
            pi.content_time = std::chrono::duration<double>(end - start).count();
        }

        #pragma omp section
        {
            auto start = std::chrono::high_resolution_clock::now();
            std::vector<uint8_t> tempTrailing(floatCount * trailingBytes);
            decompressWithZstd(compressedTrailing, tempTrailing, floatCount * trailingBytes);
            for (size_t i = 0; i < floatCount; ++i) {
                memcpy(&reconstructedData[i * (leadingBytes + contentBytes + trailingBytes) + leadingBytes + contentBytes], &tempTrailing[i * trailingBytes], trailingBytes);
            }
            auto end = std::chrono::high_resolution_clock::now();
            pi.trailing_time = std::chrono::duration<double>(end - start).count();
        }
    }

    // Verify decompressed data
    if (!verifyDataMatch(globalByteArray, reconstructedData)) {
        std::cerr << "Error: Decompressed data doesn't match the original." << std::endl;
    }
}
double calculateCompressionRatio(size_t originalSize, size_t compressedSize) {
    return static_cast<double>(originalSize) / static_cast<double>(compressedSize);
}
int main() {
    std::string datasetPath = "/home/jamalids/Documents/2D/data1/Fcbench/HPC/H/wave_f32.tsv";
    std::vector<float> floatArray = loadTSVDataset(datasetPath);
    if (floatArray.empty()) {
        std::cerr << "Failed to load dataset from " << datasetPath << std::endl;
        return 1;
    }

    // Convert float array to byte array
    globalByteArray = convertFloatToBytes(floatArray);

    // Define byte segment sizes
    size_t leadingBytes = 1;  // size in bytes for leading segment
    size_t contentBytes =2;  // size in bytes for content segment
    size_t trailingBytes =1; // size in bytes for trailing segment

    // Profiling setup
    int num_iter = 10;
    std::vector<ProfilingInfo> pi_array;
    double com_ratio;
    double compressedSize;

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

    // Save profiling data to CSV
    std::ofstream file("/home/jamalids/Documents/compression-part4/new1/num_brain_f64.csv");
    file << "Iteration,Type,CompressionRatio,TotalTimeCompressed,TotalTimeDecompressed,LeadingTime,ContentTime,TrailingTime,leading_bytes,content_bytes,trailing_bytes\n";

    for (size_t i = 0; i < pi_array.size(); ++i) {
        pi_array[i].printCSV(file, (i / 3) + 1); // Correct iteration numbering for three tests per iteration
    }
    file.close();

    std::cout << "Profiling completed and results saved." << std::endl;
    return 0;
}
