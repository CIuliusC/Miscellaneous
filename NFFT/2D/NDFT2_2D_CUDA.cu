#include<stdio.h>
#include "Utilities.cuh"

#define BLOCKSIZE_NUDFT2_2D_X	16
#define BLOCKSIZE_NUDFT2_2D_Y	16

//#define DEBUG

#define pi 3.141592653589793238463

/*************************/
/* KERNEL MATRIX FILLING */
/*************************/
__global__ void Kernel_Matrix_Filling(const double * __restrict__ d_X, const double * __restrict__ d_Y, const double * __restrict__ d_u,
	const double * __restrict__ d_v, double2 * __restrict__ d_Kernel_Matrix, const int Nu, const int Nv,
	const int M, const int N)
{
	const int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	const int tidy = blockIdx.y * blockDim.y + threadIdx.y;

	// --- Evaluates the matrix filling index
	const int tid = tidy * M + tidx;

	if (tidx < M && tidy < N) {
		d_Kernel_Matrix[tid].x = cos(-2. * pi * ((d_u[tidy] * d_X[tidx]) / static_cast<double>(Nu)
			+ (d_v[tidy] * d_Y[tidx]) / static_cast<double>(Nv)));
		d_Kernel_Matrix[tid].y = sin(-2. * pi * ((d_u[tidy] * d_X[tidx]) / static_cast<double>(Nu)
			+ (d_v[tidy] * d_Y[tidx]) / static_cast<double>(Nv)));
		//printf("%f %f\n", d_Kernel_Matrix[tid].x, d_Kernel_Matrix[tid].y);
	}

}

/************/
/* NDFT2 2D */
/************/
extern "C" {
	__declspec(dllexport)
		void NDFT2_2D_GPU(const double * __restrict__ d_X, const double * __restrict__ d_Y, const double * __restrict__ d_u,
			const double * __restrict__ d_v, double2 * __restrict__ d_in, double2 * __restrict__ d_out,
			const int Nu, const int Nv, const int M, const int N) {

		// --- N:		length of d_u and d_v
		// --- M:		length of d_X and d_Y

		cublasHandle_t handle;  cublasSafeCall(cublasCreate(&handle));
		
		double2 *d_Kernel_Matrix;		gpuErrchk(cudaMalloc(&d_Kernel_Matrix, M * N * sizeof(double2)));

		// --- Filling the kernel matrix   
		dim3 dimBlock(BLOCKSIZE_NUDFT2_2D_X, BLOCKSIZE_NUDFT2_2D_Y);
		dim3 dimGrid(iDivUp(M, BLOCKSIZE_NUDFT2_2D_X), iDivUp(N, BLOCKSIZE_NUDFT2_2D_Y));

		Kernel_Matrix_Filling << <dimGrid, dimBlock >> > (d_X, d_Y, d_u, d_v, d_Kernel_Matrix, Nu, Nv, M, N);
#ifdef DEBUG
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif

		// --- Matrix multiplication
		double2 alpha;	alpha.x = 1.; alpha.y = 0.;
		double2 beta;	beta.x = 0.; beta.y = 0.;

		cublasSafeCall(cublasZgemv(handle, CUBLAS_OP_T, M, N, &alpha, d_Kernel_Matrix, M, d_in, 1, &beta, d_out, 1));

		// --- Freeing device memory
		gpuErrchk(cudaFree(d_Kernel_Matrix));
	}
}
