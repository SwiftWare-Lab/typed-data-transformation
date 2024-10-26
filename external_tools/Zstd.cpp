#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <zstd.h>
#include <chrono>  // For measuring time
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include "common.h"

// Function to load a TSV dataset into a float vector
std::vector<float> loadTSVDataset(const std::string& filePath, size_t maxRows = 4000000) {
    std::vector<float> floatArray;
    std::ifstream file(filePath);
    std::string line;

    // Skip the header if present
    if (file.is_open()) {
        size_t rowCount = 0;
        while (std::getline(file, line) && rowCount < maxRows) {
            std::stringstream ss(line);
            std::string value;
            // Skip the first column
            std::getline(ss, value, '\t');

            // Read the rest of the row
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

// Function to convert a float array to a uint8_t vector
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
void splitBytesIntoComponents(const std::vector<uint8_t>& byteArray,
                              std::vector<uint8_t>& leading,
                              std::vector<uint8_t>& content,
                              std::vector<uint8_t>& trailing) {
    size_t floatCount = byteArray.size() / 4;

    for (size_t i = 0; i < floatCount; i++) {
        leading.push_back(byteArray[i * 4]);
        content.push_back(byteArray[i * 4 + 1]);
        content.push_back(byteArray[i * 4 + 2]);
        trailing.push_back(byteArray[i * 4 + 3]);
    }
}

// Compress a byte vector with Zstd and return compressed size
size_t compressWithZstd(const std::vector<uint8_t>& data, std::vector<uint8_t>& compressedData, int compressionLevel) {
    size_t const cBuffSize = ZSTD_compressBound(data.size());
    compressedData.resize(cBuffSize);

    size_t const cSize = ZSTD_compress(compressedData.data(), cBuffSize, data.data(), data.size(), compressionLevel);
    CHECK_ZSTD(cSize);

    compressedData.resize(cSize);
    return cSize;
}

// Decompress a byte vector with Zstd and return decompressed size
size_t decompressWithZstd(const std::vector<uint8_t>& compressedData, std::vector<uint8_t>& decompressedData, size_t originalSize) {
    decompressedData.resize(originalSize);

    size_t const dSize = ZSTD_decompress(decompressedData.data(), originalSize, compressedData.data(), compressedData.size());
    CHECK_ZSTD(dSize);

    return dSize;
}

// Verify if arrays are equal
bool verifyBytes(const std::vector<uint8_t>& original, const std::vector<uint8_t>& reconstructed) {
    if (original.size() != reconstructed.size()) {
        return false;
    }

    for (size_t i = 0; i < original.size(); i++) {
        if (original[i] != reconstructed[i]) {
            printf("Byte mismatch at index %zu: original = %d, reconstructed = %d\n",
                   i, original[i], reconstructed[i]);
            return false;
        }
    }
    return true;
}

// Compression ratio calculation
double calculateCompressionRatio(size_t originalSize, size_t compressedSize) {
    return static_cast<double>(originalSize) / static_cast<double>(compressedSize);
}

int main() {
    // Dataset path
    std::string datasetPath = "/home/jamalids/Documents/2D/data1/Fcbench/HPC/H/num_brain_f64.tsv";

    // Load the dataset
    std::vector<float> floatArray = loadTSVDataset(datasetPath);
    if (floatArray.empty()) {
        std::cerr << "Failed to load dataset from " << datasetPath << std::endl;
        return 1;
    }

    // Convert float array to byte array
    std::vector<uint8_t> byteArray = convertFloatToBytes(floatArray);

    // Compress the entire  array before decomposition
    std::vector<uint8_t> compressedData;
    auto start = std::chrono::high_resolution_clock::now();
    size_t compressedSize = compressWithZstd(byteArray, compressedData, 3);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> compressionTime = end - start;

    // Decompress the entire byte array
    std::vector<uint8_t> decompressedData;
    start = std::chrono::high_resolution_clock::now();
    decompressWithZstd(compressedData, decompressedData, byteArray.size());
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> decompressionTime = end - start;

    // comp ratio before decomposition
    double totalCompRatioBeforeDecomp = calculateCompressionRatio(byteArray.size(), compressedSize);

    printf("Compression Ratio Before Decomposition: %.2f\n", totalCompRatioBeforeDecomp);
    printf("Compression Time Before Decomposition: %.2f seconds\n", compressionTime.count());
    printf("Decompression Time Before Decomposition: %.2f seconds\n", decompressionTime.count());

    // Split into components
    std::vector<uint8_t> leading, content, trailing;
    splitBytesIntoComponents(byteArray, leading, content, trailing);

    std::vector<uint8_t> compressedLeading, compressedContent, compressedTrailing;
    std::vector<uint8_t> decompressedLeading, decompressedContent, decompressedTrailing;

    //  compression level
    int zstdLevel = 3;

    // Measure compression time for each component
    start = std::chrono::high_resolution_clock::now();
    size_t leadingCompressedSize = compressWithZstd(leading, compressedLeading, zstdLevel);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> leadingCompTime = end - start;

    start = std::chrono::high_resolution_clock::now();
    size_t contentCompressedSize = compressWithZstd(content, compressedContent, zstdLevel);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> contentCompTime = end - start;

    start = std::chrono::high_resolution_clock::now();
    size_t trailingCompressedSize = compressWithZstd(trailing, compressedTrailing, zstdLevel);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> trailingCompTime = end - start;

    // Measure decompression time for each component
    start = std::chrono::high_resolution_clock::now();
    decompressWithZstd(compressedLeading, decompressedLeading, leading.size());
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> leadingDecompTime = end - start;

    start = std::chrono::high_resolution_clock::now();
    decompressWithZstd(compressedContent, decompressedContent, content.size());
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> contentDecompTime = end - start;

    start = std::chrono::high_resolution_clock::now();
    decompressWithZstd(compressedTrailing, decompressedTrailing, trailing.size());
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> trailingDecompTime = end - start;

    // Reconstruct full  from decompressed components
    std::vector<uint8_t> reconstructedBytes(byteArray.size());
    size_t floatCount = byteArray.size() / 4;

    for (size_t i = 0; i < floatCount; i++) {
        reconstructedBytes[i * 4] = decompressedLeading[i];
        reconstructedBytes[i * 4 + 1] = decompressedContent[i * 2];
        reconstructedBytes[i * 4 + 2] = decompressedContent[i * 2 + 1];
        reconstructedBytes[i * 4 + 3] = decompressedTrailing[i];
    }

    // Verify reconstructed matches the original
    bool byteMatch = verifyBytes(byteArray, reconstructedBytes);

    // Calculate total comp ratio after decomposition
    size_t totalOriginalSize = leading.size() + content.size() + trailing.size();
    size_t totalCompressedSize = leadingCompressedSize + contentCompressedSize + trailingCompressedSize;
    double totalCompRatioAfterDecomp = calculateCompressionRatio(totalOriginalSize, totalCompressedSize);


    printf("Compression Ratios After Decomposition - Leading: %.2f, Content: %.2f, Trailing: %.2f\n",
           calculateCompressionRatio(leading.size(), leadingCompressedSize),
           calculateCompressionRatio(content.size(), contentCompressedSize),
           calculateCompressionRatio(trailing.size(), trailingCompressedSize));
    printf("Total Compression Ratio After Decomposition: %.2f\n", totalCompRatioAfterDecomp);

    printf("Compression Times - Leading: %.2f, Content: %.2f, Trailing: %.2f (seconds)\n",
           leadingCompTime.count(), contentCompTime.count(), trailingCompTime.count());
    printf("Decompression Times - Leading: %.2f, Content: %.2f, Trailing: %.2f (seconds)\n",
           leadingDecompTime.count(), contentDecompTime.count(), trailingDecompTime.count());

    if (byteMatch) {
        printf("Reconstructed byte array matches the original data.\n");
    } else {
        printf("Reconstructed byte array does not match the original.\n");
    }

    return 0;
}
