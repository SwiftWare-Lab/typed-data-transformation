#include <fcntl.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstring>
#include <iomanip>
#include <cstdlib>
#include <cstdint>
#include <unistd.h>
#include <dirent.h>
#include <sys/stat.h>
#include <map>
#include <functional>
#include <algorithm>

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "nvcomp.hpp"
#include "nvcomp/lz4.hpp"

using namespace nvcomp;

//---------------------------------------------------------
// CUDA error checking
//---------------------------------------------------------
#define CUDA_CHECK(func)                                                       \
  do {                                                                         \
    cudaError_t rt = (func);                                                   \
    if (rt != cudaSuccess) {                                                   \
      std::cerr << "CUDA API call failure '" #func "' with error: "           \
                << cudaGetErrorString(rt) << " at " << __FILE__                \
                << ":" << __LINE__ << std::endl;                              \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

//---------------------------------------------------------
// Helpers: TSV load, conversion
//---------------------------------------------------------
template <typename T>
std::pair<std::vector<T>, size_t> loadTSVDataset(const std::string &fp) {
  std::vector<T> arr;
  std::ifstream f(fp);
  std::string line;
  size_t rows = 0;
  if (!f.is_open()) throw std::runtime_error("Open failed: " + fp);
  while (std::getline(f, line)) {
    std::stringstream ss(line);
    std::string val;
    std::getline(ss, val, '\t'); // skip first column
    while (std::getline(ss, val, '\t')) {
      arr.push_back(static_cast<T>(std::stod(val)));
    }
    ++rows;
  }
  return {arr, rows};
}

std::vector<uint8_t> toBytes(const std::vector<float> &v) {
  std::vector<uint8_t> b(v.size() * sizeof(float));
  std::memcpy(b.data(), v.data(), b.size());
  return b;
}
std::vector<uint8_t> toBytes(const std::vector<double> &v) {
  std::vector<uint8_t> b(v.size() * sizeof(double));
  std::memcpy(b.data(), v.data(), b.size());
  return b;
}

//---------------------------------------------------------
// Convert config to string
//---------------------------------------------------------
std::string configToString(const std::vector<std::vector<size_t>> &cfg) {
  std::ostringstream ss;
  ss << '"';
  for (size_t i = 0; i < cfg.size(); ++i) {
    ss << '[';
    for (size_t j = 0; j < cfg[i].size(); ++j) {
      ss << cfg[i][j];
      if (j + 1 < cfg[i].size()) ss << ',';
    }
    ss << ']';
    if (i + 1 < cfg.size()) ss << ',';
  }
  ss << '"';
  return ss.str();
}

//---------------------------------------------------------
// Split bytes into components
//---------------------------------------------------------
void splitBytesIntoComponents(
  const std::vector<uint8_t> &in,
  std::vector<std::vector<uint8_t>> &out,
  const std::vector<std::vector<size_t>> &cfg,
  int numThreads)
{
  size_t elemBytes = 0;
  for (auto &g : cfg) elemBytes += g.size();
  size_t elems = in.size() / elemBytes;

  out.assign(cfg.size(), {});
  for (size_t i = 0; i < cfg.size(); ++i)
    out[i].resize(elems * cfg[i].size());

#ifdef _OPENMP
  #pragma omp parallel for num_threads(numThreads)
#endif
  for (size_t e = 0; e < elems; ++e) {
    for (size_t c = 0; c < cfg.size(); ++c) {
      for (size_t j = 0; j < cfg[c].size(); ++j) {
        size_t idx = e * elemBytes + (cfg[c][j] - 1);
        out[c][e * cfg[c].size() + j] = in[idx];
      }
    }
  }
}

//---------------------------------------------------------
// Timing helper
//---------------------------------------------------------
float measureCudaTime(std::function<void(cudaStream_t)> fn, cudaStream_t s) {
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start, s));
  fn(s);
  CUDA_CHECK(cudaEventRecord(stop, s));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms;
}

//---------------------------------------------------------
// Validation kernel
//---------------------------------------------------------
__global__
void compareBuffers(const uint8_t *a, const uint8_t *b, int *invalid, size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (; idx < n; idx += stride) {
    if (a[idx] != b[idx]) *invalid = 1;
  }
}

//---------------------------------------------------------
// Directory utils
//---------------------------------------------------------
std::vector<std::string> getAllTsvFiles(const std::string &folder) {
  std::vector<std::string> paths;
  DIR *dp = opendir(folder.c_str());
  if (!dp) return paths;
  struct dirent *de;
  while ((de = readdir(dp))) {
    std::string f(de->d_name);
    if (f.size() > 4 && f.substr(f.size() - 4) == ".tsv") {
      std::string full = folder + "/" + f;
      struct stat sb;
      if (stat(full.c_str(), &sb) == 0 && S_ISREG(sb.st_mode))
        paths.push_back(full);
    }
  }
  closedir(dp);
  return paths;
}
bool isDirectory(const std::string &p) {
  struct stat sb;
  return stat(p.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode);
}

//---------------------------------------------------------
// Process one dataset
//---------------------------------------------------------
int runSingleDataset(const std::string &path, int precisionBits) {
  // Load dataset and convert to bytes
  std::vector<uint8_t> globalBytes;
  if (precisionBits == 64) {
    auto tmp = loadTSVDataset<double>(path);
    globalBytes = toBytes(tmp.first);
  } else {
    auto tmp = loadTSVDataset<float>(path);
    globalBytes = toBytes(tmp.first);
  }
  size_t totalBytes = globalBytes.size();

  // Derive dataset name & CSV filename
  std::string datasetName = path;
  if (auto p = datasetName.find_last_of("/\\"); p != std::string::npos)
    datasetName = datasetName.substr(p + 1);
  if (auto d = datasetName.find_last_of('.'); d != std::string::npos)
    datasetName = datasetName.substr(0, d);
  std::string csvFilename = "/home/jamalids/Documents/" + datasetName + ".csv";


  // Simple component config (replace with real ones as needed)
  std::vector<std::vector<std::vector<size_t>>> candidateConfigs = {
    {{1}, {2}, {3}, {4}}
  };

  // --- Part I: Whole-dataset Compression/Decompression ---
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  uint8_t *d_in = nullptr, *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, totalBytes));
  CUDA_CHECK(cudaMalloc(&d_out, totalBytes));
  CUDA_CHECK(cudaMemcpy(d_in, globalBytes.data(), totalBytes, cudaMemcpyHostToDevice));

  LZ4Manager wholeMgr(1 << 16, nvcompBatchedLZ4Opts_t{NVCOMP_TYPE_CHAR}, stream);
  auto wholeCfg = wholeMgr.configure_compression(totalBytes);
  size_t maxWhole = ((wholeCfg.max_compressed_buffer_size - 1) / 4096 + 1) * 4096;
  uint8_t *d_whole = nullptr;
  CUDA_CHECK(cudaMalloc(&d_whole, maxWhole));

  float tC1 = measureCudaTime([&](cudaStream_t s) {
    wholeMgr.compress(d_in, d_whole, wholeCfg);
  }, stream);
  size_t out1 = wholeMgr.get_compressed_output_size(d_whole);

  auto wholeDecompCfg = wholeMgr.configure_decompression(wholeCfg);
  float tD1 = measureCudaTime([&](cudaStream_t s) {
    wholeMgr.decompress(d_out, d_whole, wholeDecompCfg);
  }, stream);

  int *h_invalid = nullptr;
  CUDA_CHECK(cudaMallocHost(&h_invalid, sizeof(int)));
  *h_invalid = 0;
  compareBuffers<<<64, 256, 0, stream>>>(d_in, d_out, h_invalid, totalBytes);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  std::cout << (*h_invalid ? "Whole FAIL\n" : "Whole PASS\n");

  double thrC1 = (totalBytes / 1e6) / tC1;
  double thrD1 = (totalBytes / 1e6) / tD1;
  double rat1 = double(totalBytes) / out1;

  wholeMgr.deallocate_gpu_mem();
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree(d_whole));
  CUDA_CHECK(cudaFreeHost(h_invalid));
  CUDA_CHECK(cudaStreamDestroy(stream));

  // Prepare containers for CSV rows
  std::vector<std::string> compRows, blockRows;

  // --- Part II: Component-based Compression/Decompression ---
  for (auto &cfg : candidateConfigs) {
    std::vector<std::vector<uint8_t>> components;
    splitBytesIntoComponents(globalBytes, components, cfg, 16);

    double totalC2 = 0.0, totalD2 = 0.0;
    size_t sum2 = 0;

    for (auto &blk : components) {
      cudaStream_t s2; CUDA_CHECK(cudaStreamCreate(&s2));
      LZ4Manager mgr(1<<16, nvcompBatchedLZ4Opts_t{NVCOMP_TYPE_CHAR}, s2);
      auto cCfg = mgr.configure_compression(blk.size());
      size_t mx = ((cCfg.max_compressed_buffer_size - 1) / 4096 + 1) * 4096;
      uint8_t *d_buf = nullptr;
      CUDA_CHECK(cudaMalloc(&d_buf, mx));
      uint8_t *d_i = nullptr, *d_o2 = nullptr;
      CUDA_CHECK(cudaMalloc(&d_i, blk.size()));
      CUDA_CHECK(cudaMalloc(&d_o2, blk.size()));
      CUDA_CHECK(cudaMemcpy(d_i, blk.data(), blk.size(), cudaMemcpyHostToDevice));

      float ctime = measureCudaTime([&](cudaStream_t s3) {
        mgr.compress(d_i, d_buf, cCfg);
      }, s2);
      size_t cbytes = mgr.get_compressed_output_size(d_buf);

      auto dCfg = mgr.configure_decompression(cCfg);
      float dtime = measureCudaTime([&](cudaStream_t s3) {
        mgr.decompress(d_o2, d_buf, dCfg);
      }, s2);

      totalC2 += ctime;
      totalD2 += dtime;
      sum2 += cbytes;

      mgr.deallocate_gpu_mem();
      CUDA_CHECK(cudaFree(d_i));
      CUDA_CHECK(cudaFree(d_o2));
      CUDA_CHECK(cudaFree(d_buf));
      CUDA_CHECK(cudaStreamDestroy(s2));
    }

    double thrC2 = (totalBytes / 1e6) / totalC2;
    double thrD2 = (totalBytes / 1e6) / totalD2;
    double rat2 = double(totalBytes) / sum2;

    std::ostringstream row2;
    row2 << datasetName
         << ",Component," << totalBytes << "," << sum2 << ","
         << std::fixed << std::setprecision(2) << rat2 << ","
         << totalC2 << "," << totalD2 << ","
         << thrC2 << "," << thrD2 << ","
         << configToString(cfg);
    compRows.push_back(row2.str());
  }

  // --- Part III: Block‑aggregated Component Compression/Decompression ---
  // const int numBlocks = 4;
  // size_t blockSize = (totalBytes + numBlocks - 1) / numBlocks;

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  int numBlocks = prop.multiProcessorCount;
  numBlocks=numBlocks/8;
  std::cout << "Launching with numBlocks = " << numBlocks << std::endl;



  // Now split the data into 'numBlocks'
  size_t blockSize = (totalBytes + numBlocks - 1) / numBlocks;


  for (auto &cfg : candidateConfigs) {
    double totalC3 = 0.0, totalD3 = 0.0;
    size_t sum3 = 0;

    for (int b = 0; b < numBlocks; ++b) {
      size_t off = b * blockSize;
      size_t sz  = std::min(blockSize, totalBytes - off);
      std::vector<uint8_t> sub(globalBytes.begin() + off,
                                globalBytes.begin() + off + sz);
      std::vector<std::vector<uint8_t>> comps;
      splitBytesIntoComponents(sub, comps, cfg, 16);

      for (auto &blk : comps) {
        cudaStream_t s3; CUDA_CHECK(cudaStreamCreate(&s3));
        LZ4Manager mgr(1<<16, nvcompBatchedLZ4Opts_t{NVCOMP_TYPE_CHAR}, s3);
        auto cCfg = mgr.configure_compression(blk.size());
        size_t mx = ((cCfg.max_compressed_buffer_size - 1) / 4096 + 1) * 4096;
        uint8_t *d_buf = nullptr;
        CUDA_CHECK(cudaMalloc(&d_buf, mx));
        uint8_t *d_i = nullptr, *d_o3 = nullptr;
        CUDA_CHECK(cudaMalloc(&d_i, blk.size()));
        CUDA_CHECK(cudaMalloc(&d_o3, blk.size()));
        CUDA_CHECK(cudaMemcpy(d_i, blk.data(), blk.size(), cudaMemcpyHostToDevice));

        float ctime = measureCudaTime([&](cudaStream_t s4) {
          mgr.compress(d_i, d_buf, cCfg);
        }, s3);
        size_t cbytes = mgr.get_compressed_output_size(d_buf);

        auto dCfg = mgr.configure_decompression(cCfg);
        float dtime = measureCudaTime([&](cudaStream_t s4) {
          mgr.decompress(d_o3, d_buf, dCfg);
        }, s3);

        totalC3 += ctime;
        totalD3 += dtime;
        sum3    += cbytes;

        mgr.deallocate_gpu_mem();
        CUDA_CHECK(cudaFree(d_i));
        CUDA_CHECK(cudaFree(d_o3));
        CUDA_CHECK(cudaFree(d_buf));
        CUDA_CHECK(cudaStreamDestroy(s3));
      }
    }

    double thrC3 = (totalBytes / 1e6) / totalC3;
    double thrD3 = (totalBytes / 1e6) / totalD3;
    double rat3  = double(totalBytes) / sum3;

    std::ostringstream row3;
    row3 << datasetName
         << ",BlockAggregated," << totalBytes << "," << sum3 << ","
         << std::fixed << std::setprecision(2) << rat3 << ","
         << totalC3 << "," << totalD3 << ","
         << thrC3 << "," << thrD3 << ","
         << configToString(cfg);
    blockRows.push_back(row3.str());
  }

  // --- Write all results to CSV ---
  std::ofstream csv(csvFilename);
  csv << "Dataset,Mode,TotalBytes,CompressedBytes,Ratio,CompTime,DecompTime,"
         "CompThroughput,DecompThroughput,Config\n";
  csv << datasetName << ",Whole," << totalBytes << "," << out1 << ","
      << std::fixed << std::setprecision(2) << rat1 << ","
      << tC1 << "," << tD1 << ","
      << thrC1 << "," << thrD1 << ",\"whole\"\n";
  for (auto &r : compRows)  csv << r << "\n";
  for (auto &r : blockRows) csv << r << "\n";
  csv.close();

  return EXIT_SUCCESS;
}

//---------------------------------------------------------
// Main entry
//---------------------------------------------------------
int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <file|folder> <32|64>\n";
    return EXIT_FAILURE;
  }
  std::string path = argv[1];
  int prec = std::stoi(argv[2]);

  if (isDirectory(path)) {
    for (auto &f : getAllTsvFiles(path)) {
      std::cout << "=== Processing " << f << " ===\n";
      runSingleDataset(f, prec);
    }
  } else {
    runSingleDataset(path, prec);
  }
  return 0;
}