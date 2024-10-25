// utils.h
#ifndef UTILS_H
#define UTILS_H

#include <bitset>
#include <cstddef>  // for size_t

// Convert binary to float
inline float binaryToFloat(const std::bitset<32>& binaryData) {
    uint32_t intBits = static_cast<uint32_t>(binaryData.to_ulong());
    float floatVal = *reinterpret_cast<float*>(&intBits);
    return floatVal;
}
// Convert binary back to float
float binaryToFloat(const std::bitset<32>& binaryData);

#endif // UTILS_H
