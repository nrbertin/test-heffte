# Test HeFFTe
```
git clone --recursive https://github.com/nrbertin/test-heffte.git
cd test-heffte
```

CUDA build
```
./build.sh build_cuda -DHeffte_ENABLE_CUDA=On -DHeffte_DISABLE_GPU_AWARE_MPI=On
cd build_cuda
mpirun -np 1 ./test_heffte
```

ROCM build
```
./build.sh build_rocm -DCMAKE_CXX_COMPILER=hipcc -DHeffte_ENABLE_ROCM=On
cd build_rocm
mpirun -np 1 ./test_heffte
```

FFTW build
```
./build.sh -DHeffte_ENABLE_FFTW=On
cd build
mpirun -np 1 ./test_heffte
```
