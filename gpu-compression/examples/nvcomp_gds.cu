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
int runSingleDataset(const std::string &path, int precisionBits, int runId){

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
  // std::vector<std::vector<std::vector<size_t>>> candidateConfigs = {
  //   {{1}, {2}, {3}, {4}}
  // };

using ComponentConfig = std::vector<std::vector<std::vector<size_t>>>;

std::vector<std::pair<std::string, ComponentConfig>> candidateConfigs = {
  { "acs_wht_f32", {
      { {1,2}, {3},   {4}   }

    }
  },
  { "citytemp_f32", {
      { {1,2}, {3}, {4} }
    }
  },
  { "hdr_night_f32", {
      { {1,2}, {3}, {4} }
    }
  },
  { "hdr_palermo_f32", {
      { {1,2}, {3}, {4} }
    }
  },
  { "hst_wfc3_ir_f32", {
      { {1,2}, {3}, {4} }
    }
  },
  { "hst_wfc3_uvis_f32", {
      { {1,2,3}, {4} }
    }
  },
  { "jw_mirimage_f32", {
      { {1,2,3}, {4} }
    }
  },
  { "rsim_f32", {
      { {1,2,3}, {4} }
    }
  },
  { "solar_wind_f32", {
      { {1,2}, {3}, {4} }
    }
  },
  { "spitzer_irac_f32", {
      { {1,2,3}, {4} }
    }
  },
  { "tpcds_catalog_f32", {
      { {1}, {2}, {3}, {4} }
    }
  },
  { "tpcds_store_f32", {
      { {1,2}, {3}, {4} }
    }
  },
  { "tpcds_web_f32", {
      { {1}, {2}, {3}, {4} }
    }
  },
  { "tpch_lineitem_f32", {
      { {1,2,3}, {4} }
    }
  },
  // wave_f32 has two possible groupings
  { "wave_f32", {
      { {1,2,3}, {4} }

    }
  },
  // default fallback
  { "default", {
      { {1}, {2}, {3}, {4} }
    }
  },

  // -- 64‑bit datasets --
  { "astro_mhd_f64", {
      { {1,2,3,4,5,6}, {7,8} }
    }
  },
  { "tpch_order_f64", {
      { {5,6}, {1,2,3,4}, {7}, {8} }
    }
  },
  { "astro_pt_f64", {
      { {4,5}, {1,2,3}, {6}, {7}, {8} }
    }
  },
  { "wesad_chest_f64", {
      { {1,2,3,4,8}, {5,6,7} }
    }
  },
  { "phone_gyro_f64", {
      { {1,2,3,4,5,6,7}, {8} }
    }
  },
  { "tpcxbb_store_f64", {
      { {6,7}, {1,2,3,4,5}, {8} }
    }
  },
  { "num_brain_f64", {
      { {2,5}, {1,3,4}, {6}, {7}, {8} },
      // alternative grouping:
      { {1,2}, {3}, {6}, {4}, {5}, {7}, {8} }
    }
  },
  { "msg_bt_f64", {
      { {2,3}, {1}, {4}, {5}, {6}, {7}, {8} }
    }
  },
  { "tpcxbb_web_f64", {
      { {6,7}, {1,2,3,4,5}, {8} }
    }
  },
  { "nyc_taxi2015_f64", {
      { {1,2,3,8}, {4,5,6,7} }
    }
  }
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

// compress
float tC1 = measureCudaTime([&](cudaStream_t s) {
  wholeMgr.compress(d_in, d_whole, wholeCfg);
}, stream);
size_t out1 = wholeMgr.get_compressed_output_size(d_whole);

// decompress
auto wholeDecompCfg = wholeMgr.configure_decompression(wholeCfg);
float tD1 = measureCudaTime([&](cudaStream_t s) {
  wholeMgr.decompress(d_out, d_whole, wholeDecompCfg);
}, stream);

// verify
int *h_invalid = nullptr;
CUDA_CHECK(cudaMallocHost(&h_invalid, sizeof(int)));
*h_invalid = 0;
compareBuffers<<<64,256,0,stream>>>(d_in, d_out, h_invalid, totalBytes);
CUDA_CHECK(cudaStreamSynchronize(stream));
std::cout << (*h_invalid ? "Whole FAIL\n" : "Whole PASS\n");

double thrC1 = (totalBytes/1e6)/tC1;
double thrD1 = (totalBytes/1e6)/tD1;
double rat1  = double(totalBytes)/out1;

wholeMgr.deallocate_gpu_mem();
CUDA_CHECK(cudaFree(d_in));
CUDA_CHECK(cudaFree(d_out));
CUDA_CHECK(cudaFree(d_whole));
CUDA_CHECK(cudaFreeHost(h_invalid));
CUDA_CHECK(cudaStreamDestroy(stream));

std::vector<std::string> compRows, blockRows;


// --- Part II: Component-based Compression/Decompression ---

using ComponentConfig = std::vector<std::vector<std::vector<size_t>>>;


const ComponentConfig *cfgListPtr = nullptr;
for (auto &e : candidateConfigs) {
  if (e.first == datasetName) {
    cfgListPtr = &e.second;
    break;
  }
}
if (!cfgListPtr) {
  // fallback
  for (auto &e : candidateConfigs) {
    if (e.first == "default") {
      cfgListPtr = &e.second;
      break;
    }
  }
}
const auto &cfgList = *cfgListPtr;

// Iterate each component grouping
for (auto &cfg : cfgList) {
  std::cout << "Selected config: " << configToString(cfg) << std::endl;

  // 1) Split into N host‐side byte‐arrays
  std::vector<std::vector<uint8_t>> components;
  splitBytesIntoComponents(globalBytes, components, cfg, /*threads=*/16);
  size_t N = components.size();

  // 2) One stream + LZ4Manager per part
  std::vector<cudaStream_t> streams(N);
  std::vector<std::unique_ptr<LZ4Manager>> mgrs;
  mgrs.reserve(N);
  nvcompBatchedLZ4Opts_t opts{NVCOMP_TYPE_CHAR};
  for (size_t i = 0; i < N; ++i) {
    CUDA_CHECK(cudaStreamCreate(&streams[i]));
    mgrs.emplace_back(new LZ4Manager(1<<16, opts, streams[i]));
  }

  // 3) Configure each
  using CompCfg = decltype(mgrs[0]->configure_compression(0));
  std::vector<CompCfg> cCfgs; cCfgs.reserve(N);
  for (size_t i = 0; i < N; ++i) {
    cCfgs.push_back(mgrs[i]->configure_compression(components[i].size()));
  }

  // 4) Allocate & copy
  std::vector<uint8_t*> d_in(N), d_buf(N), d_out(N);
  for (size_t i = 0; i < N; ++i) {
    size_t sz     = components[i].size();
    size_t maxBuf = ((cCfgs[i].max_compressed_buffer_size - 1)/4096 + 1)*4096;
    CUDA_CHECK(cudaMalloc(&d_in[i],  sz));
    CUDA_CHECK(cudaMalloc(&d_buf[i], maxBuf));
    CUDA_CHECK(cudaMalloc(&d_out[i], sz));
    CUDA_CHECK(cudaMemcpy(d_in[i], components[i].data(), sz, cudaMemcpyHostToDevice));
  }

  // 5) Compress all N in one timed region
  cudaEvent_t cStart, cEnd;
  CUDA_CHECK(cudaEventCreate(&cStart));
  CUDA_CHECK(cudaEventCreate(&cEnd));
  CUDA_CHECK(cudaEventRecord(cStart,0));
  for (size_t i=0;i<N;++i) {
    mgrs[i]->compress(d_in[i],d_buf[i],cCfgs[i]);
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(cEnd,0));
  CUDA_CHECK(cudaEventSynchronize(cEnd));
  float tC2=0;
  CUDA_CHECK(cudaEventElapsedTime(&tC2,cStart,cEnd));
  CUDA_CHECK(cudaEventDestroy(cStart));
  CUDA_CHECK(cudaEventDestroy(cEnd));

  // 6) Sum sizes
  size_t sum2=0;
  for (size_t i=0;i<N;++i) {
    sum2 += mgrs[i]->get_compressed_output_size(d_buf[i]);
  }

  // 7) Decompress all N in one timed region
  cudaEvent_t dStart, dEnd;
  CUDA_CHECK(cudaEventCreate(&dStart));
  CUDA_CHECK(cudaEventCreate(&dEnd));
  CUDA_CHECK(cudaEventRecord(dStart,0));
  for (size_t i=0;i<N;++i) {
    auto dCfg = mgrs[i]->configure_decompression(cCfgs[i]);
    mgrs[i]->decompress(d_out[i],d_buf[i],dCfg);
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(dEnd,0));
  CUDA_CHECK(cudaEventSynchronize(dEnd));
  float tD2=0;
  CUDA_CHECK(cudaEventElapsedTime(&tD2,dStart,dEnd));
  CUDA_CHECK(cudaEventDestroy(dStart));
  CUDA_CHECK(cudaEventDestroy(dEnd));

  // 8) Verify
  int *d_invalid=nullptr, h_invalid=0;
  CUDA_CHECK(cudaMalloc(&d_invalid,sizeof(int)));
  CUDA_CHECK(cudaMemset(d_invalid,0,sizeof(int)));
  for (size_t i=0;i<N;++i) {
    size_t sz = components[i].size();
    int blocks = (sz+255)/256;
    compareBuffers<<<blocks,256,0,streams[i]>>>(d_in[i],d_out[i],d_invalid,sz);
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(&h_invalid,d_invalid,sizeof(int),cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_invalid));
  if (h_invalid) {
    std::cerr<<"ERROR: Component-based mismatch!\n";
  } else {
    std::cout<<"Component-based decompression verified.\n";
  }

  // 9) Record metrics
  double thrC2 = (totalBytes/1e6)/tC2;
  double thrD2 = (totalBytes/1e6)/tD2;
  double rat2  = double(totalBytes)/sum2;
  std::ostringstream row2;
  row2<<runId<<","<<datasetName<<",Component,"<<totalBytes<<","<<sum2<<","
      <<std::fixed<<std::setprecision(2)<<rat2<<","
      <<tC2<<","<<tD2<<","<<thrC2<<","<<thrD2<<","
      <<configToString(cfg);
  compRows.push_back(row2.str());

  // 10) Cleanup
  for (size_t i=0;i<N;++i) {
    mgrs[i]->deallocate_gpu_mem();
    CUDA_CHECK(cudaFree(d_in[i]));
    CUDA_CHECK(cudaFree(d_buf[i]));
    CUDA_CHECK(cudaFree(d_out[i]));
    CUDA_CHECK(cudaStreamDestroy(streams[i]));
  }
}

// --- Part III: Block‑aggregated Component Compression/Decompression ---




// --- Part III: Block‑aggregated Component Compression/Decompression ---

  // --- Write all results to CSV ---
  std::ofstream csv(csvFilename, std::ios::app);
  csv << runId << "," << datasetName << ",Whole," << totalBytes << "," << out1 << ","
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

  const int runs = 5;

  if (isDirectory(path)) {
    for (auto &f : getAllTsvFiles(path)) {
      std::string datasetName = f;
      if (auto p = datasetName.find_last_of("/\\"); p != std::string::npos)
        datasetName = datasetName.substr(p + 1);
      if (auto d = datasetName.find_last_of('.'); d != std::string::npos)
        datasetName = datasetName.substr(0, d);
      std::string csvFilename = "/home/jamalids/Documents/" + datasetName + ".csv";

      std::ofstream csv(csvFilename);
      csv << "RUN,Dataset,Mode,TotalBytes,CompressedBytes,Ratio,CompTime,DecompTime,"
             "CompThroughput,DecompThroughput,Config\n";
      csv.close();

      for (int r = 1; r <= runs; ++r) {
        std::cout << "=== Run " << r << " for " << f << " ===\n";
        runSingleDataset(f, prec, r);
      }
    }
  } else {
    std::string datasetName = path;
    if (auto p = datasetName.find_last_of("/\\"); p != std::string::npos)
      datasetName = datasetName.substr(p + 1);
    if (auto d = datasetName.find_last_of('.'); d != std::string::npos)
      datasetName = datasetName.substr(0, d);
    std::string csvFilename = "/home/jamalids/Documents/" + datasetName + ".csv";

    std::ofstream csv(csvFilename);
    csv << "RUN,Dataset,Mode,TotalBytes,CompressedBytes,Ratio,CompTime,DecompTime,"
           "CompThroughput,DecompThroughput,Config\n";
    csv.close();

    for (int r = 1; r <= runs; ++r) {
      std::cout << "=== Run " << r << " for " << path << " ===\n";
      runSingleDataset(path, prec, r);
    }
  }
  return 0;
}
