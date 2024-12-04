set(CMAKE_CXX_STANDARD 17)
project(zstd-test-project CXX C)

# Set the C++ standard globally
set(CMAKE_CXX_STANDARD 17)

FetchContent_Declare(
        cxxopts
        GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
        GIT_TAG v2.2.1
)
FetchContent_GetProperties(cxxopts)
if(NOT cxxopts_POPULATED)
    FetchContent_Populate(cxxopts)
    set(CXXOPTS_BUILD_TESTS OFF CACHE BOOL "Enable building of tests." FORCE)  # Disable test building for cxxopts
    add_subdirectory(${cxxopts_SOURCE_DIR} ${cxxopts_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

# Fetch LZ4 for compression
FetchContent_Declare(
        lz4
        GIT_REPOSITORY https://github.com/lz4/lz4.git
        GIT_TAG dev #  branch
)
FetchContent_GetProperties(lz4)
if(NOT lz4_POPULATED)
    FetchContent_Populate(lz4)
    add_subdirectory(${lz4_SOURCE_DIR}/build/cmake ${lz4_BINARY_DIR})
endif()

add_executable(parallel-test
      # parralel-zstd.cpp
     # zstd_parallel.h
       # profiling_info.h
     #lz4_parallel.h
      # parralel-zstd.cpp
      snappy_parallel.h
      parallel-snappy.cpp
       # parallel-snappy.cpp
       #  parallel-lz4.cpp
)
# Fetch Snappy for compression
FetchContent_Declare(
        snappy
        GIT_REPOSITORY https://github.com/google/snappy.git
        GIT_TAG main
)
FetchContent_GetProperties(snappy)
if(NOT snappy_POPULATED)
    FetchContent_Populate(snappy)
    set(SNAPPY_BUILD_BENCHMARKS OFF CACHE BOOL "Disable Snappy benchmarks" FORCE)
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable Google Benchmark tests" FORCE)
    set(SNAPPY_BUILD_TESTS OFF CACHE BOOL "Disable Snappy tests" FORCE)
    add_subdirectory(${snappy_SOURCE_DIR} ${snappy_BINARY_DIR})
endif()

# Fetch Google Benchmark (ensure tests are disabled)
FetchContent_Declare(
        benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG main
)
FetchContent_GetProperties(benchmark)
if(NOT benchmark_POPULATED)
    FetchContent_Populate(benchmark)
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable Google Benchmark tests" FORCE)
    add_subdirectory(${benchmark_SOURCE_DIR} ${benchmark_BINARY_DIR})
endif()

target_link_libraries(parallel-test
        PRIVATE
        libzstd_static
        benchmark::benchmark
        OpenMP::OpenMP_CXX
        lz4
        snappy
        # argparse can be replaced or used alongside cxxopts if required
)

target_include_directories(parallel-test
        PRIVATE
        ${zstd_SOURCE_DIR}/lib
        # Include cxxopts only for this target if desired
        ${cxxopts_SOURCE_DIR}/include
        ${lz4_SOURCE_DIR}/lib
)