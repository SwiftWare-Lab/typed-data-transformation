#ifndef PROFILING_INFO_H
#define PROFILING_INFO_H

#include <string>
#include <fstream>


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

#endif // PROFILING_INFO_H

