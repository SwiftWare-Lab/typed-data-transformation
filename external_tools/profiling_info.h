#ifndef PROFILING_INFO_H
#define PROFILING_INFO_H

#include <string>
#include <fstream>

#define PROFILING_INFO_H



struct ProfilingInfo {
  double com_ratio = 0.0;
  double com_ratio_leading = 0.0;
  double com_ratio_content = 0.0;
  double com_ratio_trailing = 0.0;
  double total_time_compressed = 0.0;
  double total_time_decompressed = 0.0;
  std::string type;
  double leading_time = 0.0;
  double content_time = 0.0;
  double trailing_time = 0.0;
  double compression_throughput = 0.0;
  double decompression_throughput = 0.0;
  double total_values = 0.0;
  double total_entropy=0;
  double leading_entropy=0;
  double content_entropy=0;
  double trailing_entropy=0;

  // Updated printCSV function to include the new fields
  void printCSV(std::ofstream &file, int iteration) {
    file << iteration << ","
         << type << ","
         << com_ratio << ","
         << com_ratio_leading << ","
    << com_ratio_content << ","
    << com_ratio_trailing << ","
         << total_time_compressed << ","
         << total_time_decompressed << ","
         << leading_time << ","
         << content_time << ","
         << trailing_time << ","
         << compression_throughput<<","
         << decompression_throughput<<","
         << total_values<<","
         << total_entropy<<","
         << leading_entropy<<","
         << content_entropy<<","
          << trailing_entropy <<"\n";
  }
};

#endif // PROFILING_INFO_H
