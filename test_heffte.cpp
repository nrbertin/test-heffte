#include <stdio.h>
#include <mpi.h>
#include "heffte_fft3d.h"

#define MIN(a,b) (a<b?a:b)

int num_ranks = 1;
int rank_id = 0;

/*---------------------------------------------------------------------------
 *
 *    FFT definitions and utilities
 *
 *-------------------------------------------------------------------------*/
typedef std::complex<double> heffte_complex;

struct USE_FFTW {};
struct USE_CUFFT {};
struct USE_HIPFFT {};
struct USE_HEFFTE {};

enum fft_sign {FFT_FORWARD, FFT_BACKWARD};

struct FFTPlanBase {
    int local_Nx, local_Ny, local_Nz;
    int local_kx_start = 0;
    int local_ky_start = 0;
    int local_kz_start = 0;
    int local_size() { return local_Nx*local_Ny*local_Nz; }
};

template<class backend>
struct FFTPlan : FFTPlanBase {
    void initialize(int _Nx, int _Ny, int _Nz) {}
    void finalize() {}
};

template<class backend>
struct FFT3DTransform
{
    FFT3DTransform(FFTPlan<backend>& plan, void* in, void* out, int sign) {}
};

/*---------------------------------------------------------------------------
 *
 *    FFTW wrappers
 *
 *-------------------------------------------------------------------------*/
#if defined(Heffte_ENABLE_FFTW)
#include <fftw3.h>

template<>
struct FFTPlan<USE_FFTW> : FFTPlanBase {
    int Nx, Ny, Nz;
    void initialize(int _Nx, int _Ny, int _Nz) {
        Nx = local_Nx = _Nx;
        Ny = local_Ny = _Ny;
        Nz = local_Nz = _Nz;
        if (rank_id == 0) printf("Using FFTW on %d rank(s): %d x %d x %d\n", num_ranks, Nx, Ny, Nz);
    }
    void finalize() {}
};

template<>
struct FFT3DTransform<USE_FFTW>
{
    FFT3DTransform(FFTPlan<USE_FFTW>& plan, void* in, void* out, int sign)
    {
        int FFT_DIR = (sign == FFT_FORWARD) ? FFTW_FORWARD : FFTW_BACKWARD;
        fftw_plan p = fftw_plan_dft_3d(
            plan.Nx, plan.Ny, plan.Nz, 
            reinterpret_cast<fftw_complex*>(in),
            reinterpret_cast<fftw_complex*>(out),
            FFT_DIR, FFTW_ESTIMATE
        );
        fftw_execute(p);
        fftw_destroy_plan(p);
    }
};
#endif

/*---------------------------------------------------------------------------
 *
 *    cuFFT wrappers
 *
 *-------------------------------------------------------------------------*/
#if defined(Heffte_ENABLE_CUDA)
#include <cufft.h>
#include <cuComplex.h>

template<>
struct FFTPlan<USE_CUFFT> : FFTPlanBase {
    cufftHandle plan;
    void initialize(int Nx, int Ny, int Nz) {
        local_Nx = Nx;
        local_Ny = Ny;
        local_Nz = Nz;
        cufftPlan3d(&plan, Nx, Ny, Nz, CUFFT_Z2Z);
        if (rank_id == 0) printf("Using cuFFT on %d rank(s): %d x %d x %d\n", num_ranks, Nx, Ny, Nz);
    }
    void finalize() { cufftDestroy(plan); }
};

template<>
struct FFT3DTransform<USE_CUFFT>
{
    FFT3DTransform(FFTPlan<USE_CUFFT>& plan, void* in, void* out, int sign)
    {
        int FFT_DIR = (sign == FFT_FORWARD) ? CUFFT_FORWARD : CUFFT_INVERSE;
        cufftExecZ2Z(
            plan.plan,
            reinterpret_cast<cufftDoubleComplex*>(in),
            reinterpret_cast<cufftDoubleComplex*>(out),
            FFT_DIR
        );
        cudaDeviceSynchronize();
    }
};
#endif

/*---------------------------------------------------------------------------
 *
 *    hipFFT wrappers
 *
 *-------------------------------------------------------------------------*/
#if defined(Heffte_ENABLE_ROCM)
#include <hipfft/hipfft.h>

template<>
struct FFTPlan<USE_HIPFFT> : FFTPlanBase {
    hipfftHandle plan;
    void initialize(int Nx, int Ny, int Nz) {
        local_Nx = Nx;
        local_Ny = Ny;
        local_Nz = Nz;
        hipfftPlan3d(&plan, Nx, Ny, Nz, HIPFFT_Z2Z);
        if (rank_id == 0) printf("Using hipFFT on %d rank(s): %d x %d x %d\n", num_ranks, Nx, Ny, Nz);
    }
    void finalize() { hipfftDestroy(plan); }
};

template<>
struct FFT3DTransform<USE_HIPFFT>
{
    FFT3DTransform(FFTPlan<USE_HIPFFT>& plan, void* in, void* out, int sign)
    {
        int FFT_DIR = (sign == FFT_FORWARD) ? HIPFFT_FORWARD : HIPFFT_BACKWARD;
        hipfftExecZ2Z(
            plan.plan,
            reinterpret_cast<hipfftDoubleComplex*>(in),
            reinterpret_cast<hipfftDoubleComplex*>(out),
            FFT_DIR
        );
        (void)hipDeviceSynchronize();
    }
};
#endif

/*---------------------------------------------------------------------------
 *
 *    HeFFTe wrappers
 *
 *-------------------------------------------------------------------------*/
template<>
struct FFTPlan<USE_HEFFTE> : FFTPlanBase {
    int Nx, Ny, Nz;
    
#if defined(Heffte_ENABLE_CUDA)
    using heffte_backend = heffte::backend::cufft;
    std::string backend_name = "cuFFT";
#elif defined(Heffte_ENABLE_ROCM)
    using heffte_backend = heffte::backend::rocfft;
    std::string backend_name = "rocFFT";
#else
    using heffte_backend = heffte::backend::fftw;
    std::string backend_name = "FFTW";
#endif
    std::shared_ptr<heffte::fft3d<heffte_backend>> fft;
    std::shared_ptr<heffte::fft3d<heffte_backend>::buffer_container<heffte_complex>> workspace;
    
    void initialize(int _Nx, int _Ny, int _Nz) {
        Nx = _Nx; Ny = _Ny; Nz = _Nz;
        
        local_Nx = Nx / num_ranks + (rank_id < (Nx % num_ranks));
        local_Ny = Ny;
        local_Nz = Nz;
        local_kx_start = rank_id*(Nx / num_ranks) + MIN(rank_id, Nx % num_ranks);
        local_ky_start = 0;
        local_kz_start = 0;
        //printf("RANK[%d]: local_Nx = %d, local_kx_start = %d\n", rank_id, local_Nx, local_kx_start);

        heffte::box3d<> const box = {
            {local_kx_start, local_ky_start, local_kz_start},
            {local_kx_start+local_Nx-1, local_ky_start+local_Ny-1, local_kz_start+local_Nz-1}
        };

        if (rank_id == 0) printf("Using HeFFTe (%s backend) on %d rank(s): %d x %d x %d\n", backend_name.c_str(), num_ranks, Nx, Ny, Nz);
        //heffte::plan_options options = heffte::default_options<backend_tag>();
        fft = std::make_shared<heffte::fft3d<heffte_backend>>(box, box, MPI_COMM_WORLD/*, options*/);
        workspace = std::make_shared<heffte::fft3d<heffte_backend>::buffer_container<heffte_complex>>(fft->size_workspace());
    }
    
    void finalize() {}
};

template<>
struct FFT3DTransform<USE_HEFFTE>
{
    FFT3DTransform(FFTPlan<USE_HEFFTE>& plan, void* in, void* out, int sign)
    {
        if (sign == FFT_FORWARD) {
            plan.fft->forward(
                reinterpret_cast<heffte_complex*>(in),
                reinterpret_cast<heffte_complex*>(out),
                plan.workspace->data()
            );
        } else {
            plan.fft->backward(
                reinterpret_cast<heffte_complex*>(in),
                reinterpret_cast<heffte_complex*>(out),
                plan.workspace->data()
            );
        }
#if defined(Heffte_ENABLE_CUDA)
        cudaDeviceSynchronize();
#elif defined(Heffte_ENABLE_ROCM)
        (void)hipDeviceSynchronize();
#endif
    }
};

/*---------------------------------------------------------------------------
 *
 *    test_fft()
 *
 *-------------------------------------------------------------------------*/
template<class fft_backend>
void test_fft(int N)
{
    if constexpr (!std::is_same<fft_backend,USE_HEFFTE>::value) {
        if (num_ranks > 1) {
            //printf("Error: test_serial can only be run in serial\n");
            //exit(1);
            return;
        }
    }

    FFTPlan<fft_backend> plan;
    plan.initialize(N, N, N);

    int Nloc = plan.local_size();
    heffte_complex* array;
#if defined(Heffte_ENABLE_CUDA)
    cudaMallocManaged((void**)&array, sizeof(heffte_complex)*Nloc);
#elif defined(Heffte_ENABLE_ROCM)
    (void)hipMallocManaged(&array, sizeof(heffte_complex)*Nloc);
#else
    array = new heffte_complex[Nloc];
#endif

    for (int i = 0; i < plan.local_Nx; i++) {
        for (int j = 0; j < plan.local_Ny; j++) {
            for (int k = 0; k < plan.local_Nz; k++) {
                int kx = plan.local_kx_start+i;
                int ky = plan.local_ky_start+j;
                int kz = plan.local_kz_start+k;
                int ind_local = i*plan.local_Ny*plan.local_Nz+j*plan.local_Nz+k;
                int ind_global = kx*plan.local_Ny*plan.local_Nz+ky*plan.local_Nz+kz;
                array[ind_local] = heffte_complex(ind_global, 0.0);
            }
        }
    }

    FFT3DTransform<fft_backend>(plan, array, array, FFT_FORWARD);

    for (int i = 0; i < plan.local_Nx; i++) {
        for (int j = 0; j < plan.local_Ny; j++) {
            for (int k = 0; k < plan.local_Nz; k++) {
                int kx = plan.local_kx_start+i;
                int ky = plan.local_ky_start+j;
                int kz = plan.local_kz_start+k;
                int ind_local = i*plan.local_Ny*plan.local_Nz+j*plan.local_Nz+k;
                printf("array(%d,%d,%d) = %e\n", kx, ky, kz, array[ind_local].real());
            }
        }
    }

    plan.finalize();
#if defined(Heffte_ENABLE_CUDA)
    cudaFree(array);
#elif defined(Heffte_ENABLE_ROCM)
    (void)hipFree(array);
#else
    delete[] array;
#endif
}

/*---------------------------------------------------------------------------
 *
 *    main():   Perform a simple FFT transform and compare native
*               FFT libraries with HeFFTe results
 *
 *-------------------------------------------------------------------------*/
int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_id);

    int N = 2;
    if (argc > 1) N = atoi(argv[1]);

#if defined(Heffte_ENABLE_CUDA)
    test_fft<USE_CUFFT>(N);
#elif defined(Heffte_ENABLE_ROCM)
    test_fft<USE_HIPFFT>(N);
#else
    test_fft<USE_FFTW>(N);
#endif
    test_fft<USE_HEFFTE>(N);
    
    MPI_Finalize();
    return 0;
}
