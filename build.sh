#!/bin/sh

BUILD_DIR=$1

if [[ -z $BUILD_DIR || $BUILD_DIR == "-D"* ]]; then
    BUILD_DIR=build
fi

rm -rf ${BUILD_DIR}
mkdir -p ${BUILD_DIR} && cd ${BUILD_DIR}
cmake $@ -S ..
make -j8
