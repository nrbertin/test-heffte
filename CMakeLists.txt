cmake_minimum_required(VERSION 3.14)

project(test_heffte LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

set(TEST_INCLUDE_DIRS "")
set(TEST_EXTERN_LIBS "")

# MPI
find_package(MPI REQUIRED)
set(TEST_INCLUDE_DIRS ${TEST_INCLUDE_DIRS} ${MPI_INCLUDE_PATH})
set(TEST_EXTERN_LIBS ${TEST_EXTERN_LIBS} MPI::MPI_CXX)

# FFT
set(FFT_LIB "")
set(FFT_INC "")
if(Heffte_ENABLE_FFTW)
    find_library(FFTW_LIB fftw3 HINTS ${FFTW_ROOT} REQUIRED)
    set(FFT_LIB ${FFT_LIB} ${FFTW_LIB})
endif()
if(Heffte_ENABLE_CUDA)
    #set(FFT_LIB ${FFT_LIB} cufft)
endif()
if(Heffte_ENABLE_ROCM)
    set(FFT_LIB ${FFT_LIB} hipfft)
endif()

message(STATUS "Using heFFTe")
add_subdirectory(heffte)
set(FFT_INC ${FFT_INC} ${CMAKE_CURRENT_SOURCE_DIR}/heffte/include)
set(FFT_LIB ${FFT_LIB} Heffte)

message(STATUS "FFT_INC = ${FFT_INC}")
message(STATUS "FFT_LIB = ${FFT_LIB}")

set(TEST_INCLUDE_DIRS ${TEST_INCLUDE_DIRS} ${FFT_INC})
set(TEST_EXTERN_LIBS ${TEST_EXTERN_LIBS} ${FFT_LIB})

add_executable(test_heffte test_heffte.cpp)
target_include_directories(test_heffte PRIVATE ${TEST_INCLUDE_DIRS})
target_link_libraries(test_heffte PRIVATE ${TEST_EXTERN_LIBS})
