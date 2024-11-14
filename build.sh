#!/bin/bash

#SBATCH --cpus-per-task=64
#SBATCH --export=ALL
#SBATCH --job-name="fusion"
#SBATCH --mail-type=begin  # email me when the job starts
#SBATCH --mail-type=end    # email me when the job finishes
#SBATCH --mail-user=jamalids@mcmaster.ca
#SBATCH --nodes=1
#SBATCH --output="fusion.%j.%N.out"
#SBATCH --constraint=rome
#SBATCH --mem=254000M
#SBATCH -t 47:59:00


module load StdEnv/2023
module load gcc/13.3
module load cmake/3.27.7
ZLIB_DIR=$PWD/zlib_install
if ! ldconfig -p | grep -q "libz.so"; then
    echo "zlib not found on system; installing locally..."

    # Download and install zlib if not available
    if [ ! -d "$ZLIB_DIR" ]; then
        wget https://zlib.net/fossils/zlib-1.2.13.tar.gz  # Updated URL
        tar -xzf zlib-1.2.13.tar.gz
        cd zlib-1.2.13
        ./configure --prefix=$ZLIB_DIR
        make -j40
        make install
        cd ..
    fi

    # Set zlib paths for CMake
    export ZLIB_LIBRARY=$ZLIB_DIR/lib/libz.so
    export ZLIB_INCLUDE_DIR=$ZLIB_DIR/include
    ZLIB_CMAKE_FLAGS="-DZLIB_LIBRARY=$ZLIB_LIBRARY -DZLIB_INCLUDE_DIR=$ZLIB_INCLUDE_DIR"
else
    echo "zlib found on system."
    ZLIB_CMAKE_FLAGS=""
fi


# build the program
mkdir build
cd build
#/home/jamalids/.local/share/JetBrains/Toolbox/apps/clion/bin/cmake/linux/x64/bin/cmake -DCMAKE_BUILD_TYPE=Release ..
/home/jamalids/programs/cmake-3.31.0-rc3-linux-x86_64/bin/cmake -DCMAKE_BUILD_TYPE=Release ..
make

cd ..

# run the program
./build/external_tools/parallel-test




