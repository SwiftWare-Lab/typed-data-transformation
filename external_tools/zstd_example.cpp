//
// Created by Kazem on 2024-10-24.
//

#include <benchmark/benchmark.h>

#include <cstdio>     // printf
#include <cstdlib>    // free
#include <cstring>    // strlen, strcat, memset
#include <zstd.h>      // presumes zstd library is installed
#include "common.h"    // Helper functions, CHECK(), and CHECK_ZSTD()
#include "zstd.h"


static void compress_orDie_random(const size_t fSize, const char* oname){
  // Create a random array of floats
  float* const fBuff = (float*)malloc(fSize * sizeof(float));
  for (size_t i = 0; i < fSize; i++) fBuff[i] = (float)i;
  size_t const cBuffSize = ZSTD_compressBound(fSize * sizeof(float));
  void* const cBuff = malloc_orDie(cBuffSize);

  // Compress
  size_t const cSize = ZSTD_compress(cBuff, cBuffSize, fBuff, fSize * sizeof(float), 1);
  CHECK_ZSTD(cSize);

  // Save the compressed data
  saveFile_orDie(oname, cBuff, cSize);

  // Success
  printf("Random data compressed to %s\n", oname);

  // Clean up
  free(fBuff);
  free(cBuff);
}


static void compress_orDie(const char* fname, const char* oname)
{
  size_t fSize;
  void* const fBuff = mallocAndLoadFile_orDie(fname, &fSize);
  size_t const cBuffSize = ZSTD_compressBound(fSize);
  void* const cBuff = malloc_orDie(cBuffSize);

  /* Compress.
   * If you are doing many compressions, you may want to reuse the context.
   * See the multiple_simple_compression.c example.
   */
  size_t const cSize = ZSTD_compress(cBuff, cBuffSize, fBuff, fSize, 1);
  CHECK_ZSTD(cSize);

  saveFile_orDie(oname, cBuff, cSize);

  /* success */
  printf("%25s : %6u -> %7u - %s \n", fname, (unsigned)fSize, (unsigned)cSize, oname);

  free(fBuff);
  free(cBuff);
}

static void decompress(const char* fname)
{
  size_t cSize;
  void* const cBuff = mallocAndLoadFile_orDie(fname, &cSize);
  /* Read the content size from the frame header. For simplicity we require
   * that it is always present. By default, zstd will write the content size
   * in the header when it is known. If you can't guarantee that the frame
   * content size is always written into the header, either use streaming
   * decompression, or ZSTD_decompressBound().
   */
  unsigned long long const rSize = ZSTD_getFrameContentSize(cBuff, cSize);
  CHECK(rSize != ZSTD_CONTENTSIZE_ERROR, "%s: not compressed by zstd!", fname);
  CHECK(rSize != ZSTD_CONTENTSIZE_UNKNOWN, "%s: original size unknown!", fname);

  void* const rBuff = malloc_orDie((size_t)rSize);

  /* Decompress.
   * If you are doing many decompressions, you may want to reuse the context
   * and use ZSTD_decompressDCtx(). If you want to set advanced parameters,
   * use ZSTD_DCtx_setParameter().
   */
  size_t const dSize = ZSTD_decompress(rBuff, rSize, cBuff, cSize);
  CHECK_ZSTD(dSize);
  /* When zstd knows the content size, it will error if it doesn't match. */
  CHECK(dSize == rSize, "Impossible because zstd will check this condition!");

  /* success */
  //printf("%25s : %6u -> %7u \n", fname, (unsigned)cSize, (unsigned)rSize);

  free(rBuff);
  free(cBuff);
}


//static char* createOutFilename_orDie(const char* filename)
//{
//  size_t const inL = strlen(filename);
//  size_t const outL = inL + 5;
//  void* const outSpace = malloc_orDie(outL);
//  memset(outSpace, 0, outL);
//  strcat(outSpace, filename);
//  strcat(outSpace, ".zst");
//  return (char*)outSpace;
//}

//int main(int argc, const char** argv)
//{
//  const char* const exeName = argv[0];
//
//  if (argc!=3) {
//    printf("wrong arguments\n");
//    printf("usage:\n");
//    printf("%s FILE\n", exeName);
//    return 1;
//  }
//
//  const char* const inFilename = argv[1];
//  const size_t fSize = atoi(argv[2]);
//
//  //char* const outFilename = createOutFilename_orDie(inFilename);
////  char *outFilename = (char *)malloc(strlen(inFilename)+5);
////  compress_orDie(inFilename, outFilename);
////  free(outFilename);
//  compress_orDie_random(fSize, inFilename);
//  decompress(inFilename);
//
//  return 0;
//}

void *compImp(int a) {
  compress_orDie("test_file", "test_file.zst");
  return NULL;
}

static void ZSTD_BENCH(benchmark::State &state, void (* compImp1)(int)) {
  int fSize = state.range(0);
  int zstdLevel = state.range(1);
  int numThreads = state.range(2);
  std::string fName = "test_file";

  float* const fBuff = (float*)malloc(fSize * sizeof(float));
  for (size_t i = 0; i < fSize; i++) fBuff[i] = (float)i;
  size_t const cBuffSize = ZSTD_compressBound(fSize * sizeof(float));
  void* const cBuff = malloc_orDie(cBuffSize);

  size_t cSize;

// set number of threads
  //ZSTD_setParameter(ZSTD_c_compressionLevel, zstdLevel);
  ZSTD_CCtx* cctx = ZSTD_createCCtx();
  ZSTD_CCtx_setParameter(cctx, ZSTD_c_nbWorkers, numThreads);
  for (auto _: state) {
    // Compress
    //cSize = ZSTD_compress(cBuff, cBuffSize, fBuff, fSize * sizeof(float), zstdLevel);
    cSize = ZSTD_compressCCtx(cctx, cBuff, cBuffSize, fBuff, fSize * sizeof(float), zstdLevel);
    CHECK_ZSTD(cSize);
  }

  // Save the compressed data
  saveFile_orDie(fName.c_str(), cBuff, cSize);
  // TODO verify the compressed file
  decompress(fName.c_str());
  // compare uncompressed and original data
}

// Register the function as a benchmark
int main() {
  benchmark::RegisterBenchmark("ZSTD_BENCH", ZSTD_BENCH, reinterpret_cast<void (*)(int)>(compImp))
  ->Args({1000000, 22, 1})->Args({2000000, 22, 1})->Args({1000000, 3, 1})->Args({2000000, 3, 1})->Args({1000000, 22, 1})->Args({2000000, 1, 1})
  ->Args({1000000, 22, 4})->Args({2000000, 22, 4})->Args({1000000, 3, 4})->Args({2000000, 3, 4})->Args({1000000, 22, 4})->Args({2000000, 1, 4});
  benchmark::RunSpecifiedBenchmarks();
}
//BENCHMARK_CAPTURE(ZSTD_BENCH, test_case, reinterpret_cast<void (*)(int)>(compImp))->Args({1000000});
