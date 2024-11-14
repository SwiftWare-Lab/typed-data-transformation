set(CMAKE_CXX_STANDARD 17)
project(zstd-test-project CXX C)

# Set the C++ standard globally
set(CMAKE_CXX_STANDARD 17)
set(ZLIB_ROOT "/usr")
find_package(ZLIB REQUIRED)
FetchContent_Declare(
        cxxopts
        GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
        GIT_TAG v2.2.1
)
include(FetchContent)


FetchContent_GetProperties(cxxopts)
if(NOT cxxopts_POPULATED)
    FetchContent_Populate(cxxopts)
    set(CXXOPTS_BUILD_TESTS OFF CACHE BOOL "Enable building of tests." FORCE)  # Disable test building for cxxopts
    add_subdirectory(${cxxopts_SOURCE_DIR} ${cxxopts_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()





add_executable(parallel-test
      # parralel-zstd.cpp
       # zstd_parallel.h
       zlib.h
        gzip.cpp

        profiling_info.h
       # gzip.cpp
        # Add other cpp files or headers here if needed
)

target_link_libraries(parallel-test
        PRIVATE
        libzstd_static
        benchmark::benchmark
        OpenMP::OpenMP_CXX
        ZLIB::ZLIB
        # argparse can be replaced or used alongside cxxopts if required
)

target_include_directories(parallel-test
        PRIVATE
        ${zstd_SOURCE_DIR}/lib
        # Include cxxopts only for this target if desired
        ${cxxopts_SOURCE_DIR}/include
)