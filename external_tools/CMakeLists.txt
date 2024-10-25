set(CMAKE_CXX_STANDARD 17)
project(zstd-test-project CXX C)

add_executable(zstd-test
        zstd_example.cpp
        zstd_test_com_decom.cpp
        zstd_decompress.cpp
        utils.cpp

)

target_link_libraries( zstd-test
        PRIVATE
        libzstd_static
        benchmark::benchmark
)

# On windows and macos this is needed
target_include_directories( zstd-test
        PRIVATE
        ${zstd_SOURCE_DIR}/lib
)