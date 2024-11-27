#ifndef PROFILING_INFO_H
#define PROFILING_INFO_H

#include <string>
#include <fstream>

#define PROFILING_INFO_H



struct ProfilingInfo1 {
  double com_ratio = 0.0;
  double total_time_compressed = 0.0;
  double total_time_decompressed = 0.0;
  std::string type;
  double leading_time = 0.0;
  double content_time = 0.0;
  double trailing_time = 0.0;
  double compression_throughput = 0.0;
  double decompression_throughput = 0.0;
  double total_values = 0.0;

  // Updated printCSV function to include the new fields
  void printCSV1(std::ofstream &file, int iteration) {
    file << iteration << ","
         << type << ","
         << com_ratio << ","
         << total_time_compressed << ","
         << total_time_decompressed << ","
         << leading_time << ","
         << content_time << ","
         << trailing_time << ","
         << compression_throughput<<","
         << decompression_throughput<<","
         << total_values<<"\n";
  }
};
struct ProfilingInfo {
  double com_ratio = 0.0;
  double total_time_compressed = 0.0;
  double total_time_decompressed = 0.0;
  std::string type;
  std::vector<double> component_times; // Store times for each component
  double compression_throughput = 0.0;
  double decompression_throughput = 0.0;
  double total_values = 0.0;

  // Constructor to initialize `component_times` dynamically
  ProfilingInfo(size_t num_components = 0)
      : component_times(num_components, 0.0) {}

  // Updated printCSV function to include times for all components
  void printCSV(std::ofstream &file, int iteration) {
    file << iteration << ","
         << type << ","
         << com_ratio << ","
         << total_time_compressed << ","
         << total_time_decompressed << ",";

    // Append component times dynamically
    for (size_t i = 0; i < component_times.size(); ++i) {
      file << component_times[i];
      if (i < component_times.size() - 1) {
        file << ",";
      }
    }

    file << "," << compression_throughput << ","
         << decompression_throughput << ","
         << total_values << "\n";
  }
};


#endif // PROFILING_INFO_H
