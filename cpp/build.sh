#!/bin/bash

BUILD_DIR="./build"

if [ -d "${BUILD_DIR}" ]; then
  echo "Found old build directory, deleting it..."
  rm -rf ${BUILD_DIR};
  echo "Done."
fi


echo "Building TFTRT CPP converter..."
mkdir build
cd build
cmake ..
make
echo "Done"
