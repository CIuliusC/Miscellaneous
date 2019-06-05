#include <stdio.h>
#include <math.h>
#include <fstream>
#include <iostream>
#include "fftw3.h"

#include "TimingCPU.h"

static struct double2 {
	double x, y;
};

#define IDX2R(i,j,N) (((i)*(N))+(j))

#define cc 2
#define K 6

//#define TIMING

#define pi_double	3.141592653589793238462643383279502884197169399375105820974944592

static double alfa = (2. - 1. / cc) * pi_double - 0.01;

using namespace std;

/**********************/
/* FUSED MULTIPLY-ADD */
/**********************/
inline double fma(double x, double y, double z) { return x*y + z; }

/*******************************/
/* MODIFIED BESSEL FUNCTION I0 */
/*******************************/
static double bessi0_CPU(double x)
{
//	 -- See paper
//	 J.M. Blair, "Rational Chebyshev approximations for the modified Bessel functions I_0(x) and I_1(x)", Math. Comput., vol. 28, n. 126, pp. 581-583, Apr. 1974.   

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

/**************************************/
/* CALCULATION OF THE BESSEL FUNCTION */
/**************************************/
static void Calculate_Bessel(double* __restrict Bessel_vector, const int N)
{
	for (int i = 0; i<N; i++) {
		double xi = (2 * pi_double*(i - (N / 2))) / (cc*N);
		Bessel_vector[i] = 1 / (bessi0_CPU(K*sqrt(alfa*alfa - xi*xi)));
	}
}


/***********************************/
/* SCALING AND ZERO PADDING KERNEL */
/***********************************/
static void ZeroPadding_CPU(const double2* __restrict data, double2* __restrict U, const double* __restrict Bessel_vector_x, const double* __restrict Bessel_vector_y, const int N1, const int N2)
{
	double a;
	int ind_i, ind_j;

	//for (int i = 0; i<cc*N1; i++)
	//	for (int j = 0; j<cc*N2; j++)
	//	{
	//		if ((((i >= (cc - 1)*N1 / 2) && (i < (cc + 1)*N1 / 2)) || ((j >= (cc - 1)*N2 / 2) && (j < (cc + 1)*N2 / 2)))) {
	//			U[IDX2R(i, j, cc*N2)].x = 0;
	//			U[IDX2R(i, j, cc*N2)].y = 0;
	//		}
	//		else {
	//			ind_i = (i + N1 + (N1 / 2))&(N1 - 1); // change to (i+N1+(N1/2))%N1; if N1 is not power of 2
	//			ind_j = (j + N2 + (N2 / 2))&(N2 - 1); // change to (j+N2+(N2/2))%N2; if N2 is not power of 2

	//			a = Bessel_vector_x[ind_i] * Bessel_vector_y[ind_j];

	//			U[IDX2R(i, j, cc * N2)].x = data[IDX2R(ind_i, ind_j, N2)].x * a;
	//			U[IDX2R(i, j, cc * N2)].y = data[IDX2R(ind_i, ind_j, N2)].y * a;
	//			//U[IDX2R(j, i, cc * N1)].x = data[IDX2R(ind_j, ind_i, N1)].x*a;
	//			//U[IDX2R(j, i, cc * N1)].y = data[IDX2R(ind_j, ind_i, N1)].y*a;
	//		}
	//	}

	for (int i = 0; i<cc*N1; i++)
		for (int j = 0; j<cc*N2; j++)
		{
			if ((((i >= (cc - 1)*N1 / 2) && (i < (cc + 1)*N1 / 2)) && ((j >= (cc - 1)*N2 / 2) && (j < (cc + 1)*N2 / 2)))) {
				double a = Bessel_vector_x[i - (cc - 1)*N1 / 2] * Bessel_vector_y[j - (cc - 1)*N2 / 2];

				U[IDX2R(i, j, cc*N2)].x = data[IDX2R(i - (cc - 1)*N1 / 2, j - (cc - 1)*N2 / 2, N2)].x*a;
				U[IDX2R(i, j, cc*N2)].y = data[IDX2R(i - (cc - 1)*N1 / 2, j - (cc - 1)*N2 / 2, N2)].y*a;
			}
			else {
				U[IDX2R(i, j, cc*N2)].x = 0.;
				U[IDX2R(i, j, cc*N2)].y = 0.;
			}
		}
}

/*******************/
/* MODULO FUNCTION */
/*******************/
//int modulo_CPU(int val, int _mod)
//{
//	if (val > 0) return val&(_mod - 1);
//	else
//	{
//		int P = (-val)&(_mod - 1);
//		if (P > 0) return _mod - P;
//		else return 0;
//	}
//}
static int modulo_CPU(int val, int _mod)
{
	int P;
	if (val > 0) { (!(_mod & (_mod - 1)) ? P = val&(_mod - 1) : P = val % (_mod)); return P; }
	else
	{
		(!(_mod & (_mod - 1)) ? P = (-val)&(_mod - 1) : P = (-val) % (_mod));
		if (P > 0) return _mod - P;
		else return 0;
	}
}

/*****************/
/* RINT FUNCTION */
/*****************/
inline double rint(double x)
{
	int temp; temp = (x >= 0. ? (int)(x + 0.5) : (int)(x - 0.5));
	return (double)temp;
}

/************/
/* FFTSHIFT */
/************/
static void fftshift_2D(double2 * __restrict data, const int N1, const int N2)
{
	for (int i = 0; i < N1; i++)
		for (int j = 0; j < N2; j++) {
			data[i*N2 + j].x *= 1 - 2 * ((i + j) & 1);
			data[i*N2 + j].y *= 1 - 2 * ((i + j) & 1);
		}
}

/*****************/
/* INTERPOLATION */
/*****************/
//void Interpolation_CPU(const double2* __restrict U, const double* __restrict x1, const double* __restrict x2, double2* __restrict tr, const int N1, const int N2, const int N)
//{
//
//	for (int i = 0; i<N; i++)
//	{
//		int ind_i, ind_j;
//		double phicap1, phicap2, tempd, p1, p2, expon;
//		double2 temp; temp.x = 0.; temp.y = 0.;
//
//		for (int m1 = -K; m1 <= K; m1++)
//		{
//			ind_i = modulo_CPU((int)rint(cc*x1[i]) + m1 + cc*N1, cc*N1);
//
//			expon = (cc*x1[i] - (rint(cc*x1[i]) + (double)m1));
//			p1 = K*K - expon*expon;
//			if (p1<0.) { tempd = sqrt(-p1); phicap1 = (1. / pi_double)*((sin(alfa*tempd)) / tempd); }
//			else if (p1>0.) { tempd = sqrt(p1); phicap1 = (1. / pi_double)*((sinh(alfa*tempd)) / tempd); }
//			else phicap1 = alfa / pi_double;
//			for (int m2 = -K; m2 <= K; m2++)
//			{
//				ind_j = modulo_CPU((int)rint(cc*x2[i]) + m2 + cc*N2, cc*N2);
//
//				expon = (cc*x2[i] - (rint(cc*x2[i]) + (double)m2));
//				p2 = K*K - expon*expon;
//				if (p2<0.) { tempd = sqrt(-p2); phicap2 = (1. / pi_double)*((sin(alfa*tempd)) / tempd); }
//				else if (p2>0.) { tempd = sqrt(p2); phicap2 = (1. / pi_double)*((sinh(alfa*tempd)) / tempd); }
//				else phicap2 = alfa / pi_double;
//
//				temp.x = temp.x + phicap1*phicap2*U[IDX2R(ind_j, ind_i, cc*N2)].x;
//				temp.y = temp.y + phicap1*phicap2*U[IDX2R(ind_j, ind_i, cc*N2)].y;
//			}
//		}
//		tr[i] = temp;
//	}
//}
//void Interpolation_CPU(const double2* __restrict U_d, const double* __restrict x1_d, const double* __restrict x2_d, double2* __restrict tr, const int N1, const int N2, const int N)
//{
//
//	for (int i = 0; i < N; i++) {
//
//		int ind_i, ind_j;
//		double phicap1, phicap2, tempd, p1, p2, expon;
//		double2 temp; temp.x = 0.; temp.y = 0.;
//
//			for (int m1 = -K; m1 <= K; m1++)
//				//for (int m1=-K; m1<=(-K+1); m1++)
//			{
//				ind_i = modulo_CPU((int)rint(cc*x1_d[i]) + m1 + cc*N1, cc*N1);
//
//				expon = (cc*x1_d[i] - (rint(cc*x1_d[i]) + (double)m1));
//				p1 = K*K - expon*expon;
//				if (p1 < 0.) { tempd = 1. / sqrt(-p1); phicap1 = (1. / pi_double)*((sin(alfa / tempd))*tempd); }
//				else if (p1 > 0.) { tempd = 1. / sqrt(p1); phicap1 = (1. / pi_double)*((sinh(alfa / tempd))*tempd); }
//				else phicap1 = alfa / pi_double;
//				//printf("%i %i %i\n",i,m1,ind_i);
//				for (int m2 = -K; m2 <= K; m2++)
//					//for (int m2  = -K; m2<=(-K+1); m2++)
//				{
//					ind_j = modulo_CPU((int)rint(cc*x2_d[i]) + m2 + cc*N2, cc*N2);
//
//					expon = (cc*x2_d[i] - (rint(cc*x2_d[i]) + (double)m2));
//					p2 = K*K - expon*expon;
//					if (p2 < 0.) { tempd = 1. / sqrt(-p2); phicap2 = (1. / pi_double)*((sin(alfa / tempd))*tempd); }
//					else if (p2 > 0.) { tempd = 1. / sqrt(p2); phicap2 = (1. / pi_double)*((sinh(alfa / tempd))*tempd); }
//					else phicap2 = alfa / pi_double;
//
//					// temp.x = temp.x+phicap1*phicap2*U_d[IDX2R(ind_j,ind_i,cc*N2)].x; 
//					//temp.y = temp.y+phicap1*phicap2*U_d[IDX2R(ind_j,ind_i,cc*N2)].y; } }
//					temp.x = temp.x + phicap1*phicap2*U_d[IDX2R(ind_i, ind_j, cc*N2)].x;
//					temp.y = temp.y + phicap1*phicap2*U_d[IDX2R(ind_i, ind_j, cc*N2)].y;
//				}
//			}
//			tr[i] = temp;
//	}
//}
void Interpolation_CPU(const double2* __restrict U_d, const double* __restrict x1_d, const double* __restrict x2_d, double2* __restrict tr, const int N1, const int N2, const int N) {

	double invPi = (1. / pi_double);
	double phi12;

	for (int i = 0; i < N; i++) {

		int ind_i, ind_j;
		double phicap1, phicap2, tempd, p1, p2, expon;
		double2 temp; temp.x = 0.; temp.y = 0.;

		double	rintd1 = rint(cc * x1_d[i]);
		double  rintd2 = rint(cc * x2_d[i]);
		int		rinti1 = (int)rintd1;
		int     rinti2 = (int)rintd2;

		for (int m1 = -K; m1 <= K; m1++) {
			
			ind_i = modulo_CPU(rinti1 + m1 + cc * N1, cc * N1);

			expon = (cc*x1_d[i] - (rintd1 + (double)m1));
			p1 = K * K - expon * expon;
			if (p1 < 0.) { tempd = sqrt(-p1); phicap1 = invPi * ((sin(alfa * tempd)) / tempd); }
			else if (p1 > 0.) { tempd = sqrt(p1); phicap1 = invPi * ((sinh(alfa * tempd)) / tempd); }
			else phicap1 = invPi * alfa;
			
			for (int m2 = -K; m2 <= K; m2++) {
				ind_j = modulo_CPU(rinti2 + m2 + cc * N2, cc * N2);

				expon = (cc*x2_d[i] - (rintd2 + (double)m2));
				p2 = K*K - expon*expon;
				if (p2 < 0.) { tempd = sqrt(-p2); phicap2 = invPi * ((sin(alfa * tempd)) / tempd); }
				else if (p2 > 0.) { tempd = sqrt(p2); phicap2 = invPi * ((sinh(alfa * tempd)) / tempd); }
				else phicap2 = invPi * alfa;

				phi12 = phicap1 * phicap2;
				temp.x = temp.x + phi12 * U_d[IDX2R(ind_i, ind_j, cc*N2)].x;
				temp.y = temp.y + phi12 * U_d[IDX2R(ind_i, ind_j, cc*N2)].y;
			}
		}
		tr[i] = temp;
	}
}
/*************************/
/* NUFFT NED 2D FUNCTION */
/*************************/
extern "C" {
	__declspec(dllexport)
		void NFFT1_2D_CPU(double2* __restrict result, const double2* __restrict data, const double* __restrict x, const double* __restrict y, const int N1, const int N2, const int M)
	{

		TimingCPU timerCPU;
		
		//for (int i = 0; i<N1*N2; i++) std::cout << "test C " << i << " " << data[i].x << " " << data[i].y << "\n";
		//for (int i = 0; i < M; i++) std::cout << "test C " << i << " " << x[i] << " " << y[i] << "\n";

		/* CALCULATION OF BESSEL FUNCTIONS */
		double* Bessel_vector_x; Bessel_vector_x = (double*)malloc(N1 * sizeof(double));
		double* Bessel_vector_y; Bessel_vector_y = (double*)malloc(N2 * sizeof(double));

#ifdef TIMING
		timerCPU.StartCounter();
#endif
		Calculate_Bessel(Bessel_vector_x, N1);
		Calculate_Bessel(Bessel_vector_y, N2);
#ifdef TIMING
		printf("Calculate Bessel %f\n", timerCPU.GetCounter());
#endif

		//for (int k = 0; k < N1; k++) printf("test C %d %2.15f\n", k, Bessel_vector_x[k]);
		//for (int k = 0; k < N2; k++) printf("test C %d %2.15f\n", k, Bessel_vector_y[k]);

		/* ALLOCATIONS AND INITIALIZATIONS */
		//double2* temp_result_NUFFT;			temp_result_NUFFT = (double2*)malloc(cc * N1 * cc * N2 * sizeof(double2));
		//memset(temp_result_NUFFT, 0, cc * N1 * cc * N2 * sizeof(double2));
#ifdef TIMING
		timerCPU.StartCounter();
#endif
		double2* temp_result_NUFFT;			temp_result_NUFFT = (double2*)calloc(cc * N1 * cc * N2, sizeof(double2));
#ifdef TIMING
		printf("Allocation %f\n", timerCPU.GetCounter());
#endif

		/* SCALING AND ZERO PADDING */
#ifdef TIMING
		timerCPU.StartCounter();
#endif
		ZeroPadding_CPU(data, temp_result_NUFFT, Bessel_vector_x, Bessel_vector_y, N1, N2);
#ifdef TIMING
		printf("Zero Padding %f\n", timerCPU.GetCounter());
#endif
		//for (int i = 0; i<cc*N1*cc*N2; i++) std::cout << "test C " << i << " " << temp_result_NUFFT[i].x << " " << temp_result_NUFFT[i].y << "\n";
		
		/* FFT */
#ifdef TIMING
		timerCPU.StartCounter();
#endif
		fftw_plan plan;
		plan = fftw_plan_dft_2d(cc * N1, cc * N2, (fftw_complex*)temp_result_NUFFT, (fftw_complex*)temp_result_NUFFT, FFTW_FORWARD, FFTW_ESTIMATE);
#ifdef TIMING
		printf("plan creation %f\n", timerCPU.GetCounter());
		timerCPU.StartCounter();
#endif
		fftw_execute(plan);
#ifdef TIMING
		printf("FFT %f\n", timerCPU.GetCounter());
		timerCPU.StartCounter();
#endif
		fftw_destroy_plan(plan);
#ifdef TIMING
		printf("Plan destruction %f\n", timerCPU.GetCounter());
#endif
		//for (int i = 0; i<cc*N1*cc*N2; i++) std::cout << "test C " << i << " " << temp_result_NUFFT[i].x << " " << temp_result_NUFFT[i].y << "\n";

#ifdef TIMING
		timerCPU.StartCounter();
#endif
		fftshift_2D(temp_result_NUFFT, cc * N1, cc * N2);
#ifdef TIMING
		printf("FFT shift %f\n", timerCPU.GetCounter());
#endif
		//for (int i = 0; i<cc*N1*cc*N2; i++) std::cout << "test C " << i << " " << temp_result_NUFFT[i].x << " " << temp_result_NUFFT[i].y << "\n";

		/* INTERPOLATION */
#ifdef TIMING
		timerCPU.StartCounter();
#endif
		Interpolation_CPU(temp_result_NUFFT, x, y, result, N1, N2, M);
#ifdef TIMING
		printf("Interpolation %f\n", timerCPU.GetCounter());
		//for (int i = 0; i < M; i++) std::cout << "test C " << i << " " << result[i].x << " " << result[i].y << "\n";
		printf("--------------------------------------------------------\n");
#endif
	}
}
