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
#include <memory>

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


  //  component config
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


  std::vector<std::string> compRows, blockRows;

 // --- Part II: Component-based Compression/Decompression ---

for (auto &cfg : candidateConfigs) {
  // 1) Split host bytes into components
  std::vector<std::vector<uint8_t>> components;
  splitBytesIntoComponents(globalBytes, components, cfg, /*threads=*/16);
  size_t N = components.size();

  // 2) Create one CUDA stream + LZ4Manager per component
  std::vector<cudaStream_t> streams(N);
  std::vector<std::unique_ptr<LZ4Manager>> mgrs;
  mgrs.reserve(N);
  nvcompBatchedLZ4Opts_t opts;
  opts.data_type = NVCOMP_TYPE_CHAR;
  for (size_t i = 0; i < N; ++i) {
    CUDA_CHECK(cudaStreamCreate(&streams[i]));
    mgrs.emplace_back(
      new LZ4Manager(
        /*batchBytes=*/1 << 16,
        opts,
        streams[i]
      )
    );
  }

  // 3) Configure compression for each component
  using CompCfg = decltype(mgrs[0]->configure_compression(0));
  std::vector<CompCfg> cCfgs;
  cCfgs.reserve(N);
  for (size_t i = 0; i < N; ++i) {
    cCfgs.push_back(
      mgrs[i]->configure_compression(components[i].size()));
  }

  // 4) Allocate device input, output, and compressed buffers
  std::vector<uint8_t*> d_in(N), d_out(N), d_buf(N);
  for (size_t i = 0; i < N; ++i) {
    size_t sz     = components[i].size();
    size_t maxBuf = ((cCfgs[i].max_compressed_buffer_size - 1) / 4096 + 1) * 4096;

    CUDA_CHECK(cudaMalloc(&d_in[i],  sz));
    CUDA_CHECK(cudaMalloc(&d_out[i], sz));
    CUDA_CHECK(cudaMalloc(&d_buf[i], maxBuf));

    CUDA_CHECK(cudaMemcpy(
      d_in[i],
      components[i].data(),
      sz,
      cudaMemcpyHostToDevice
    ));
  }

  // 5) Time all N compress calls in one go
  float tC2 = measureCudaTime([&](cudaStream_t) {
    for (size_t i = 0; i < N; ++i) {
      mgrs[i]->compress(d_in[i], d_buf[i], cCfgs[i]);
    }
  }, streams[0]);

  // 6) Sum compressed sizes
  size_t sum2 = 0;
  for (size_t i = 0; i < N; ++i) {
    sum2 += mgrs[i]->get_compressed_output_size(d_buf[i]);
  }

  // 7) Time all N decompress calls in one go
  float tD2 = measureCudaTime([&](cudaStream_t) {
    for (size_t i = 0; i < N; ++i) {
      auto dCfg = mgrs[i]->configure_decompression(cCfgs[i]);
      mgrs[i]->decompress(d_out[i], d_buf[i], dCfg);
    }
  }, streams[0]);

  // 8) Compute throughput and ratio
  double thrC2 = (totalBytes / 1e6) / tC2;
  double thrD2 = (totalBytes / 1e6) / tD2;
  double rat2  = double(totalBytes) / sum2;

  // 9) Emit CSV row
  std::ostringstream row2;
  row2 << datasetName
       << ",Component," << totalBytes << "," << sum2 << ","
       << std::fixed << std::setprecision(2) << rat2 << ","
       << tC2 << "," << tD2 << ","
       << thrC2 << "," << thrD2 << ","
       << configToString(cfg);
  compRows.push_back(row2.str());

  // 10) Cleanup
  for (size_t i = 0; i < N; ++i) {
    mgrs[i]->deallocate_gpu_mem();
    CUDA_CHECK(cudaFree(d_in[i]));
    CUDA_CHECK(cudaFree(d_out[i]));
    CUDA_CHECK(cudaFree(d_buf[i]));
    CUDA_CHECK(cudaStreamDestroy(streams[i]));
  }
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