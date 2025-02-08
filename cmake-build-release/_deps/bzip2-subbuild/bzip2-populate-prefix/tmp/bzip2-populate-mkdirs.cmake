# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/jamalids/Documents/2D/final results/big-data-compression/cmake-build-release/_deps/bzip2-src"
  "/home/jamalids/Documents/2D/final results/big-data-compression/cmake-build-release/_deps/bzip2-build"
  "/home/jamalids/Documents/2D/final results/big-data-compression/cmake-build-release/_deps/bzip2-subbuild/bzip2-populate-prefix"
  "/home/jamalids/Documents/2D/final results/big-data-compression/cmake-build-release/_deps/bzip2-subbuild/bzip2-populate-prefix/tmp"
  "/home/jamalids/Documents/2D/final results/big-data-compression/cmake-build-release/_deps/bzip2-subbuild/bzip2-populate-prefix/src/bzip2-populate-stamp"
  "/home/jamalids/Documents/2D/final results/big-data-compression/cmake-build-release/_deps/bzip2-subbuild/bzip2-populate-prefix/src"
  "/home/jamalids/Documents/2D/final results/big-data-compression/cmake-build-release/_deps/bzip2-subbuild/bzip2-populate-prefix/src/bzip2-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/jamalids/Documents/2D/final results/big-data-compression/cmake-build-release/_deps/bzip2-subbuild/bzip2-populate-prefix/src/bzip2-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/jamalids/Documents/2D/final results/big-data-compression/cmake-build-release/_deps/bzip2-subbuild/bzip2-populate-prefix/src/bzip2-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
