set(CMAKE_CXX_STANDARD 17)
project(zstd-test-project CXX C)

add_executable(zstd2-test
        Zstd2.cpp
        # add other cpp files here if needed
)
target_link_libraries( zstd2-test
        PRIVATE
        libzstd_static
        benchmark::benchmark
        OpenMP::OpenMP_CXX
)

# On windows and macos this is needed
target_include_directories( zstd2-test
        PRIVATE
        ${zstd_SOURCE_DIR}/lib
)


add_executable(zstd-test
        #  zstd_example.cpp
        Zstd.cpp
)
target_link_libraries( zstd-test
        PRIVATE
        libzstd_static
        benchmark::benchmark
        OpenMP::OpenMP_CXX
)

# On windows and macos this is needed
target_include_directories( zstd-test
        PRIVATE
        ${zstd_SOURCE_DIR}/lib
)


add_executable(parallel-test
       # parallel.cpp
        #parallel.cpp
        parralel-zstd.cpp
        # add other cpp files here if needed
)
target_link_libraries( parallel-test
        PRIVATE
        libzstd_static
        benchmark::benchmark
        OpenMP::OpenMP_CXX
)

# On windows and macos this is needed
target_include_directories( parallel-test
        PRIVATE
        ${zstd_SOURCE_DIR}/lib
)
