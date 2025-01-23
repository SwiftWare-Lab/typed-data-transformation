cmake_minimum_required( VERSION 3.20)


project(big-data CXX C )


# ZSTD cloning
include(FetchContent)

set(ZSTD_BUILD_STATIC ON)
set(ZSTD_BUILD_SHARED OFF)
#set multithread on
set(ZSTD_MULTITHREAD_SUPPORT ON)

FetchContent_Declare(
        zstd
        URL "https://github.com/facebook/zstd/releases/download/v1.5.6/zstd-1.5.6.tar.gz"
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        SOURCE_SUBDIR build/cmake
)

FetchContent_MakeAvailable(zstd)

include(FetchContent)

FetchContent_Declare(
        lz4
        GIT_REPOSITORY https://github.com/lz4/lz4.git
        GIT_TAG        v1.10.0 # Or a specific tag/commit
)

FetchContent_MakeAvailable(lz4)


# add deflate
FetchContent_Declare(
        deflate
        GIT_REPOSITORY https://github.com/ebiggers/libdeflate
        GIT_TAG        v1.23
)

FetchContent_MakeAvailable(deflate)

message("===> ${deflate_SOURCE_DIR}")

FetchContent_Declare(
        zlib
        GIT_REPOSITORY https://github.com/madler/zlib.git
        GIT_TAG        v1.2.13 # Or a specific tag/commit
)

FetchContent_MakeAvailable(zlib)


#target_link_libraries(
#        ${PROJECT_NAME}
#        PRIVATE
#        libzstd_static
#)
#
## On windows and macos this is needed
#target_include_directories(
#        ${PROJECT_NAME}
#        PRIVATE
#        ${zstd_SOURCE_DIR}/lib
#)



#option(PROFILING_ENABLED "Enables Modules for Profiling, needs LIBFPM4" OFF)
#
#FetchContent_Declare(
#        googlebenchmark
#        GIT_REPOSITORY https://github.com/google/benchmark.git
#        GIT_TAG v1.9.0
#)
## enable BENCHMARK_DOWNLOAD_DEPENDENCIES
#set(BENCHMARK_DOWNLOAD_DEPENDENCIES ON)
#if (PROFILING_ENABLED)
#    # enable PFM
#    set(BENCHMARK_ENABLE_LIBPFM OFF)
#endif()
## disable tests
##set(BENCHMARK_ENABLE_TESTING OFF)
#
#FetchContent_MakeAvailable(googlebenchmark)

## Argparse
include(FetchContent)
FetchContent_Declare(
        argparse
        GIT_REPOSITORY https://github.com/p-ranav/argparse.git
)
FetchContent_MakeAvailable(argparse)


# find openmp
find_package(OpenMP REQUIRED)


add_subdirectory(external_tools)
#add_compile_definitions(gpu-compression)


set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)

project(gpu-compression CUDA)

# Find CUDA (this is KEY)
#find_package(CUDA REQUIRED COMPONENTS npp nvrtc)

#set(CMAKE_PREFIX_PATH "/usr/lib/sbsa-linux-gnu/nvcomp/")

find_package(nvcomp REQUIRED)

#find_package(LZ4 REQUIRED)


# include nvcomp found directory



set(CMAKE_CUDA_STANDARD 20)

add_executable(gpu-compression gpu-compression/nvcomp_demo.cu)

# include zlib header

message("----> NVIDIA HEADERS: ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
message("===> ${zlib_SOURCE_DIR}")

# include nvidia headrs
target_include_directories(gpu-compression PRIVATE
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        /usr/include/nvcomp_12/
        #${zlib_SOURCE_DIR}/
        ${deflate_SOURCE_DIR}
        #${LZ4_INCLUDE_DIR}
)

#set_target_properties(gpu-compression PROPERTIES
#        CUDA_SEPARABLE_COMPILATION ON
##        nvcomp::nvcomp
##        CUDA::cudart
#        )

target_link_libraries(gpu-compression PRIVATE
        CUDA::cudart
        nvcomp::nvcomp
        #CUDA::npp
        CUDA::nvrtc
        #${LZ4_LIBRARY}
        ${deflate_SOURCE_DIR}/lib
        libdeflate::libdeflate_static
        zlibstatic

        )

