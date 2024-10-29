#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <zstd.h>
#include <chrono>
#include <cstdint>


std::vector<uint8_t> globalByteArray;

std::vector<float> loadTSVDataset(const std::string& filePath, size_t maxRows = 8000000);

//  verify the original data matches the reconstructed data
bool verifyDataMatch(const std::vector<uint8_t>& original, const std::vector<uint8_t>& reconstructed) {
    // Check if sizes are the same
    if (original.size() != reconstructed.size()) {
        std::cerr << "Size mismatch: Original size = " << original.size()
                  << ", Reconstructed size = " << reconstructed.size() << std::endl;
        return false;
    }

    // Compare each byte of the original and reconstructed data
    for (size_t i = 0; i < original.size(); ++i) {
        if (original[i] != reconstructed[i]) {
            std::cerr << "Data mismatch at index " << i << ": Original = "
                      << static_cast<int>(original[i]) << ", Reconstructed = "
                      << static_cast<int>(reconstructed[i]) << std::endl;
            return false;
        }
    }


    return true;
}


//  load a TSV dataset into a float vector
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

// Compress  Zstd
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

// Decompress  with Zstd
size_t decompressWithZstd(const std::vector<uint8_t>& compressedData, std::vector<uint8_t>& decompressedData, size_t originalSize) {
    decompressedData.resize(originalSize);

    size_t const dSize = ZSTD_decompress(decompressedData.data(), originalSize, compressedData.data(), compressedData.size());
    if (ZSTD_isError(dSize)) {
        std::cerr << "Zstd decompression error: " << ZSTD_getErrorName(dSize) << std::endl;
        return 0;
    }

    return dSize;
}

//  compression ratio
double calculateCompressionRatio(size_t originalSize, size_t compressedSize) {
    return static_cast<double>(originalSize) / static_cast<double>(compressedSize);
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
//  save results to a CSV file
void saveResultsToCSV(const std::string& filename, const std::vector<std::vector<std::string>>& results) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return;
    }


    file << "Iteration,CompRatio,CompressedTime,"
         << "DecompressedTime,CompRatio_Decomposition,decompositionTime,"
         << "Leading_CompressedTime,Leading_DecompressedTime,Content_CompressedTime,Content_DecompressedTime,"
         << "Trailing_CompressedTime,Trailing_DecompressedTime,ReconstructionTime,Decomposition_CompressedTime,Decomposition_DecompressedTime\n";


    for (const auto& row : results) {
        for (size_t i = 0; i < row.size(); ++i) {
            file << row[i];
            if (i < row.size() - 1) file << ",";
        }
        file << "\n";
    }

    file.close();
    std::cout << "Results saved to " << filename << std::endl;
}

int main() {

    std::string datasetPath = "/home/jamalids/Documents/2D/data1/Fcbench/HPC/H/num_brain_f64.tsv";
    std::vector<float> floatArray = loadTSVDataset(datasetPath);

    if (floatArray.empty()) {
        std::cerr << "Failed to load dataset from " << datasetPath << std::endl;
        return 1;
    }

    // Convert float array to byte array
    globalByteArray = convertFloatToBytes(floatArray);

    // Split global byte array into components
    std::vector<uint8_t> leading, content, trailing;
    std::chrono::duration<double> decompositionTime;
    auto start = std::chrono::high_resolution_clock::now();
    splitBytesIntoComponents(globalByteArray, leading, content, trailing);
    auto end = std::chrono::high_resolution_clock::now();
    decompositionTime = end - start;

    std::vector<uint8_t> compressedLeading, compressedContent, compressedTrailing;
    std::vector<uint8_t> decompressedLeading(leading.size()), decompressedContent(content.size()), decompressedTrailing(trailing.size());
    std::vector<uint8_t> reconstructedData;

    double compRatioBeforeDecomp = 0.0, totalCompRatioAfterDecomp = 0.0; ;
    std::chrono::duration<double> compressionTimeBeforeDecomp, decompressionTimeBeforeDecomp;
    std::chrono::duration<double> leadingCompTime, contentCompTime, trailingCompTime;
    std::chrono::duration<double> leadingDecompTime, contentDecompTime, trailingDecompTime;
    std::chrono::duration<double> reconstructionTime;
    std::chrono::duration<double>totalCompressedTime;
    std::chrono::duration<double>totalDecompressedTime;

    // Variables to track max decomposition compression and decompression times
    std::chrono::duration<double> maxDecompositionCompTime = std::chrono::seconds(0);
    std::chrono::duration<double> maxDecompositionDecompTime = std::chrono::seconds(0);


    std::vector<std::vector<std::string>> results;


    for (int i = 0; i < 10; ++i) {
        // Compress the global byte array
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<uint8_t> compressedData;
        size_t compressedSize = compressWithZstd(globalByteArray, compressedData, 3);
        auto end = std::chrono::high_resolution_clock::now();
        compressionTimeBeforeDecomp = end - start;
        compRatioBeforeDecomp = calculateCompressionRatio(globalByteArray.size(), compressedSize);

        // Decompress full data
        std::vector<uint8_t> decompressedData(globalByteArray.size());
        start = std::chrono::high_resolution_clock::now();
        decompressWithZstd(compressedData, decompressedData, globalByteArray.size());
        end = std::chrono::high_resolution_clock::now();
        decompressionTimeBeforeDecomp = end - start;

        // Compression for each component
        start = std::chrono::high_resolution_clock::now();
        size_t leadingCompressedSize = compressWithZstd(leading, compressedLeading, 3);
        end = std::chrono::high_resolution_clock::now();
        leadingCompTime = end - start;

        start = std::chrono::high_resolution_clock::now();
        size_t contentCompressedSize = compressWithZstd(content, compressedContent, 3);
        end = std::chrono::high_resolution_clock::now();
        contentCompTime = end - start;

        start = std::chrono::high_resolution_clock::now();
        size_t trailingCompressedSize = compressWithZstd(trailing, compressedTrailing, 3);
        end = std::chrono::high_resolution_clock::now();
        trailingCompTime = end - start;

        // Decompression for each component
        start = std::chrono::high_resolution_clock::now();
        decompressWithZstd(compressedLeading, decompressedLeading, leading.size());
        end = std::chrono::high_resolution_clock::now();
        leadingDecompTime = end - start;

        start = std::chrono::high_resolution_clock::now();
        decompressWithZstd(compressedContent, decompressedContent, content.size());
        end = std::chrono::high_resolution_clock::now();
        contentDecompTime = end - start;

        start = std::chrono::high_resolution_clock::now();
        decompressWithZstd(compressedTrailing, decompressedTrailing, trailing.size());
        end = std::chrono::high_resolution_clock::now();
        trailingDecompTime = end - start;

        // Reconstruction
        start = std::chrono::high_resolution_clock::now();
        reconstructFromComponents(decompressedLeading, decompressedContent, decompressedTrailing, reconstructedData);
        end = std::chrono::high_resolution_clock::now();
        reconstructionTime = end - start;

        // Verify the reconstructed data matches the original data
        if (!verifyDataMatch(globalByteArray, reconstructedData)) {
            std::cerr << "Error: Reconstructed data doesn't match the original." << std::endl;
            return 1;
        }
        else {
            std::cerr << " Reconstructed data matchs the original." << std::endl;

        }


        size_t totalOriginalSize = leading.size() + content.size() + trailing.size();
        size_t totalCompressedSize = leadingCompressedSize + contentCompressedSize + trailingCompressedSize;
        totalCompRatioAfterDecomp = calculateCompressionRatio(totalOriginalSize, totalCompressedSize);
        totalCompressedTime = leadingCompTime+contentCompTime +trailingCompTime +decompositionTime;
        totalDecompressedTime=leadingDecompTime+contentDecompTime+trailingDecompTime +reconstructionTime;

        // Update max decomposition compression and decompression times
        maxDecompositionCompTime = std::max(maxDecompositionCompTime, totalCompressedTime);
        maxDecompositionDecompTime = std::max(maxDecompositionDecompTime, totalDecompressedTime);

        // Store results for CSV
        results.push_back({
            std::to_string(i + 1),
            std::to_string(compRatioBeforeDecomp),
            std::to_string(compressionTimeBeforeDecomp.count()),
            std::to_string(decompressionTimeBeforeDecomp.count()),
            std::to_string(totalCompRatioAfterDecomp),
            std::to_string(decompositionTime.count()),
            std::to_string(leadingCompTime.count()),
            std::to_string(leadingDecompTime.count()),
            std::to_string(contentCompTime.count()),
            std::to_string(contentDecompTime.count()),
            std::to_string(trailingCompTime.count()),
            std::to_string(trailingDecompTime.count()),
            std::to_string(reconstructionTime.count()),
            std::to_string(totalCompressedTime.count()),
            std::to_string(totalDecompressedTime.count()),


        });
    }


    saveResultsToCSV("/home/jamalids/Documents/compression-part4/new/compression_results.csv", results);


    return 0;
}