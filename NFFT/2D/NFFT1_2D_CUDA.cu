#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>  
#include <cuda_runtime.h>
#include <cufft.h>
#include <math.h>
#include <fstream>
#include <conio.h>
#include <iostream>
#include <iomanip>
#include <assert.h>

#include "cuFFT_auxiliary.cuh"
#include "Utilities.cuh"

using namespace std;

#define cc 2
#define K 6
const int block_size_Bessel = 32;

#define IDX2R(i,j,N) (((i)*(N))+(j))

#define pi_double	3.141592653589793238463

__constant__ double alfa = (2. - 1. / cc)*pi_double - 0.01;

cufftHandle plan;

/**************************/
/* cuFFT PLAN CALCULATION */
/**************************/
void Calculate_cuFFT_plan(const int N1, const int N2) { cufftSafeCall(cufftPlan2d(&plan, cc*N1, cc*N2, CUFFT_Z2Z)); }

/**************************/
/* cuFFT PLAN DESTRUCTION */
/**************************/
void Destroy_cuFFT_plan() { cufftSafeCall(cufftDestroy(plan)); }

/********************************************************/
/* MODIFIED BESSEL FUNCTION ZERO-TH ORDER - GPU VERSION */
/********************************************************/
static __device__ double bessi0_GPU(double x)
{
	// -- See paper
	// J.M. Blair, "Rational Chebyshev approximations for the modified Bessel functions I_0(x) and I_1(x)", 
	//				Math. Comput., vol. 28, n. 126, pp. 581-583, Apr. 1974.   

	double num, den, x2;

	x2 = abs(x*x);

	x = abs(x);

	if (x > 15.0)
	{
		den = 1.0 / x;
		num = -4.4979236558557991E+006;
		num = fma(num, den, 2.7472555659426521E+006);
		num = fma(num, den, -6.4572046640793153E+005);
		num = fma(num, den, 8.5476214845610564E+004);
		num = fma(num, den, -7.1127665397362362E+003);
		num = fma(num, den, 4.1710918140001479E+002);
		num = fma(num, den, -1.3787683843558749E+001);
		num = fma(num, den, 1.1452802345029696E+000);
		num = fma(num, den, 2.1935487807470277E-001);
		num = fma(num, den, 9.0727240339987830E-002);
		num = fma(num, den, 4.4741066428061006E-002);
		num = fma(num, den, 2.9219412078729436E-002);
		num = fma(num, den, 2.8050629067165909E-002);
		num = fma(num, den, 4.9867785050221047E-002);
		num = fma(num, den, 3.9894228040143265E-001);
		num = num * den;
		den = sqrt(x);
		num = num * den;
		den = exp(0.5 * x);  /* prevent premature overflow */
		num = num * den;
		num = num * den;
		return num;
	}
	else
	{
		num = -0.27288446572737951578789523409E+010;
		num = fma(num, x2, -0.6768549084673824894340380223E+009);
		num = fma(num, x2, -0.4130296432630476829274339869E+008);
		num = fma(num, x2, -0.11016595146164611763171787004E+007);
		num = fma(num, x2, -0.1624100026427837007503320319E+005);
		num = fma(num, x2, -0.1503841142335444405893518061E+003);
		num = fma(num, x2, -0.947449149975326604416967031E+000);
		num = fma(num, x2, -0.4287350374762007105516581810E-002);
		num = fma(num, x2, -0.1447896113298369009581404138E-004);
		num = fma(num, x2, -0.375114023744978945259642850E-007);
		num = fma(num, x2, -0.760147559624348256501094832E-010);
		num = fma(num, x2, -0.121992831543841162565677055E-012);
		num = fma(num, x2, -0.15587387207852991014838679E-015);
		num = fma(num, x2, -0.15795544211478823152992269E-018);
		num = fma(num, x2, -0.1247819710175804058844059E-021);
		num = fma(num, x2, -0.72585406935875957424755E-025);
		num = fma(num, x2, -0.28840544803647313855232E-028);

		den = -0.2728844657273795156746641315E+010;
		den = fma(den, x2, 0.5356255851066290475987259E+007);
		den = fma(den, x2, -0.38305191682802536272760E+004);
		den = fma(den, x2, 0.1E+001);

		return num / den;
	}
}

/***********************************************/
/* MODIFIED BESSEL FUNCTION CALCULATION KERNEL */
/***********************************************/
__global__ void Kernel_Bessel(double* __restrict__ Bessel_vector, const int N)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i<N) {
		double xi = (2 * pi_double*(i - (N / 2))) / (cc*N);
		Bessel_vector[i] = 1 / (bessi0_GPU(K*sqrt(alfa*alfa - xi*xi)));
		//Bessel_vector[i] = 1 / (cyl_bessel_i0f(K*sqrt(alfa*alfa - xi*xi)));
	}
}

/************************/
/* INTERPOLATION 2D NER */
/************************/
__global__ void Interpolation_2D_NER(const double2* __restrict__ U_d, const double* __restrict__ x1_d, const double* __restrict__ x2_d, double2* __restrict__ tr, const int N1, const int N2, const int N)
{

	int i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i<N)
	{
		int ind_i, ind_j;
		double phicap1, phicap2, tempd, p1, p2, expon;
		double2 temp = make_cuDoubleComplex(0.0, 0.0);

		for (int m1 = -K; m1 <= K; m1++)
			//for (int m1=-K; m1<=(-K+1); m1++)
		{
			ind_i = modulo((int)rint(cc*x1_d[i]) + m1 + cc*N1, cc*N1);

			expon = (cc*x1_d[i] - (rint(cc*x1_d[i]) + (double)m1));
			p1 = K*K - expon*expon;
			if (p1<0.) { tempd = rsqrt(-p1); phicap1 = (1. / pi_double)*((sin(alfa / tempd))*tempd); }
			else if (p1>0.) { tempd = rsqrt(p1); phicap1 = (1. / pi_double)*((sinh(alfa / tempd))*tempd); }
			else phicap1 = alfa / pi_double;
			//printf("%i %i %i\n",i,m1,ind_i);
			for (int m2 = -K; m2 <= K; m2++)
				//for (int m2  = -K; m2<=(-K+1); m2++)
			{
				ind_j = modulo((int)rint(cc*x2_d[i]) + m2 + cc*N2, cc*N2);

				expon = (cc*x2_d[i] - (rint(cc*x2_d[i]) + (double)m2));
				p2 = K*K - expon*expon;
				if (p2<0.) { tempd = rsqrt(-p2); phicap2 = (1. / pi_double)*((sin(alfa / tempd))*tempd); }
				else if (p2>0.) { tempd = rsqrt(p2); phicap2 = (1. / pi_double)*((sinh(alfa / tempd))*tempd); }
				else phicap2 = alfa / pi_double;

				// temp.x = temp.x+phicap1*phicap2*U_d[IDX2R(ind_j,ind_i,cc*N2)].x; 
				//temp.y = temp.y+phicap1*phicap2*U_d[IDX2R(ind_j,ind_i,cc*N2)].y; } }
				temp.x = temp.x + phicap1*phicap2*U_d[IDX2R(ind_i, ind_j, cc*N2)].x;
				temp.y = temp.y + phicap1*phicap2*U_d[IDX2R(ind_i, ind_j, cc*N2)].y;
			}
		}
		tr[i] = temp;
	}
}

/***********************************/
/* SCALING AND ZERO PADDING KERNEL */
/***********************************/
__global__ void ZeroPadding(const double2* __restrict__ data_d, double2* __restrict__ U_d, const double* __restrict__ Bessel_vector_x, const double* __restrict__ Bessel_vector_y, const int N1, const int N2)
{
	int i = threadIdx.y + blockDim.y * blockIdx.y;
	int j = threadIdx.x + blockDim.x * blockIdx.x;

	if (i <cc*N1 && j < cc*N2) {

		//     if((((i >= (cc-1)*N1/2)  && (i < (cc+1)*N1/2)) || ((j >= (cc-1)*N2/2)  && (j < (cc+1)*N2/2)))) {
		//         U_d[IDX2R(i,j,cc*N2)].x =0;
		//         U_d[IDX2R(i,j,cc*N2)].y =0;
		//     }else{
		//         int ind_i = (i+N1+(N1/2))&(N1-1); // change to (i+N1+(N1/2))%N1; if N1 is not power of 2
		//         int ind_j = (j+N2+(N2/2))&(N2-1); // change to (j+N2+(N2/2))%N2; if N2 is not power of 2

		//double a = Bessel_vector_x[ind_i]*Bessel_vector_y[ind_j];

		//U_d[IDX2R(i,j,cc*N2)].x = data_d[IDX2R(ind_i,ind_j,N2)].x*a;
		//U_d[IDX2R(i,j,cc*N2)].y = data_d[IDX2R(ind_i,ind_j,N2)].y*a;}

		if ((((i >= (cc - 1)*N1 / 2) && (i < (cc + 1)*N1 / 2)) && ((j >= (cc - 1)*N2 / 2) && (j < (cc + 1)*N2 / 2)))) {
			double a = Bessel_vector_x[i - (cc - 1)*N1 / 2] * Bessel_vector_y[j - (cc - 1)*N2 / 2];

			U_d[IDX2R(i, j, cc*N2)].x = data_d[IDX2R(i - (cc - 1)*N1 / 2, j - (cc - 1)*N2 / 2, N2)].x*a;
			U_d[IDX2R(i, j, cc*N2)].y = data_d[IDX2R(i - (cc - 1)*N1 / 2, j - (cc - 1)*N2 / 2, N2)].y*a;
		}
		else {
			U_d[IDX2R(i, j, cc*N2)].x = 0.;
			U_d[IDX2R(i, j, cc*N2)].y = 0.;
		}

	}
}

#define BLOCK_SIZE	 256
#define BLOCK_SIZE_x 16
#define BLOCK_SIZE_y 16

const int block_size_Interp = 256;
const int block_size = 16;

/***************************/
/* NUFFT NER 2D EVALUATION */
/***************************/
extern "C" {
	__declspec(dllexport)
	void NFFT1_2D_GPU(double2* __restrict__ tr, const double2* __restrict__ data_d, const double* __restrict__ x1_d, const double* __restrict__ x2_d, const int N1, const int N2, const int N)
	{
		double2 *U_d; cudaMalloc((void**)&U_d, sizeof(double2)*cc*N1*cc*N2);

		/* CALCULATION OF BESSEL FUNCTIONS */
		double* Bessel_vector_x; cudaMalloc((void **)&Bessel_vector_x, sizeof(double)*N1);
		double* Bessel_vector_y; cudaMalloc((void **)&Bessel_vector_y, sizeof(double)*N2);

		dim3 dimBlock01(block_size_Bessel, 1); dim3 dimGrid01(N1 / block_size_Bessel + (N1%block_size_Bessel == 0 ? 0 : 1), 1);
		Kernel_Bessel << <dimGrid01, dimBlock01 >> > (Bessel_vector_x, N1);
		dim3 dimBlock02(block_size_Bessel, 1); dim3 dimGrid02(N2 / block_size_Bessel + (N2%block_size_Bessel == 0 ? 0 : 1), 1);
		Kernel_Bessel << <dimGrid02, dimBlock02 >> > (Bessel_vector_y, N2);

		//double *test = (double*) malloc(sizeof(double)*N2);
		//cudaMemcpy(test, Bessel_vector_x, sizeof(double ) * N2, cudaMemcpyDeviceToHost);
		//for (int i = 0; i < N2; i++) cout << "test " << i << " " << test[i] << "\n";

		/* SCALING AND ZERO PADDING */
		dim3 dimBlock1(block_size, block_size);
		dim3 dimGrid1((cc*N2) / block_size + ((cc*N2) % block_size == 0 ? 0 : 1), (cc*N1) / block_size + ((cc*N1) % block_size == 0 ? 0 : 1));
		ZeroPadding << <dimGrid1, dimBlock1 >> > (data_d, U_d, Bessel_vector_x, Bessel_vector_y, N1, N2);

		//double2* test = (double2*) malloc(sizeof(double2)*cc*N1*cc*N2);
		//cudaMemcpy(test,U_d,sizeof(double2)*cc*N1*cc*N2,cudaMemcpyDeviceToHost);
		//for (int i=0; i<cc*N1*cc*N2; i++) cout << "test " << i << " " << test[i].x << " " << test[i].y << "\n";

		/* FFT */
		Calculate_cuFFT_plan(N1, N2);
		cufftExecZ2Z(plan, U_d, U_d, CUFFT_FORWARD);
		Destroy_cuFFT_plan();

		//double2* test = (double2*) malloc(sizeof(double2)*cc*N1*cc*N2);
		//cudaMemcpy(test,U_d,sizeof(double2)*cc*N1*cc*N2,cudaMemcpyDeviceToHost);
		//for (int i=0; i<cc*N1*cc*N2; i++) cout << "test " << i << " " << test[i].x << " " << test[i].y << "\n";

		/* FFTSHIFT 2D */
		dim3 dimBlockFFTSHIFT(block_size, block_size);
		dim3 dimGridFFTSHIFT((cc*N1) / block_size + ((cc*N1) % block_size == 0 ? 0 : 1), (cc*N2) / block_size + ((cc*N2) % block_size == 0 ? 0 : 1));
		fftshift_2D << <dimGridFFTSHIFT, dimBlockFFTSHIFT >> > (U_d, cc*N1, cc*N2);

		//double2* test = (double2*) malloc(sizeof(double2)*cc*N1*cc*N2);
		//cudaMemcpy(test,U_d,sizeof(double2)*cc*N1*cc*N2,cudaMemcpyDeviceToHost);
		//for (int i=0; i<cc*N1*cc*N2; i++) cout << "test " << i << " " << test[i].x << " " << test[i].y << "\n";

		//std::ofstream outfile;
		//outfile.open("fftshift_GPU.txt");
		//for(int i=0; i<cc*N1*cc*N2; i++) { outfile << std::setprecision(10) << U_d[i].x << "\n"; outfile << std::setprecision(10) << U_d[i].y << "\n"; }
		//outfile.close();

		/* INTERPOLATION */
		dim3 dimBlock2(block_size_Interp, 1);
		dim3 dimGrid2(N / block_size_Interp + (N%block_size_Interp == 0 ? 0 : 1), 1);
		Interpolation_2D_NER << <dimGrid2, dimBlock2 >> > (U_d, x1_d, x2_d, tr, N1, N2, N);

		//double2* test = (double2*) malloc(sizeof(double2)*N);
		//cudaMemcpy(test,tr,sizeof(double2)*N,cudaMemcpyDeviceToHost);
		//for (int i=0; i<N; i++) cout << "test " << i << " " << test[i].x << " " << test[i].y << "\n";

	}
}
