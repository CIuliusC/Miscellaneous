#include <stdio.h>
#include <math.h>
#include <fstream>
#include "fftw3.h"
//#include <cuda.h>  
//#include <cuda_runtime.h>
#include <conio.h> 
#include <iostream>
#include <iomanip>

static struct double2 {
	double x, y;
}; 

using namespace std;

#define IDX2R(i,j,N) (((i)*(N))+(j))

#define pi_double	3.141592653589793238463

// --- NUFFT NED 2D parameters
#define cc 2
#define K 6
double alfa_CPU = (2. - 1. / cc)*pi_double - 0.01;

fftw_plan plan;
double2* temp_result_NUFFT;

/**************************/
/* FFTW PLAN CALCULATION */
/**************************/
void Calculate_FFTW_plan(const int N1, const int N2) {

	//temp_result_NUFFT = (double2*)malloc(cc*N1*cc*N2 * sizeof(double2));
	plan = fftw_plan_dft_2d(cc*N1, cc*N2, (fftw_complex*)temp_result_NUFFT, (fftw_complex*)temp_result_NUFFT, FFTW_FORWARD, FFTW_ESTIMATE);
}

/*************************/
/* FFTW PLAN DESTRUCTION */
/*************************/
void Destroy_FFTW_plan() {
	fftw_destroy_plan(plan);
}

/*******************/
/* MODULO FUNCTION */
/*******************/
static int modulo_CPU(int val, int _mod)
{
	//	if(val > 0) return val&(_mod-1);
	int P;
	// printf("%i %i\n",_mod,!(_mod & (_mod - 1)));
	if (val > 0) { (!(_mod & (_mod - 1)) ? P = val&(_mod - 1) : P = val % (_mod)); return P; }
	// if(val > 0) return val%(_mod);
	else
	{
		//		int P = (-val)&(_mod-1);
		// int P = (-val)%(_mod);
		//printf("%i\n",!(-val == 0) && !(val & (val - 1)));
		(!(_mod & (_mod - 1)) ? P = (-val)&(_mod - 1) : P = (-val) % (_mod));
		if (P > 0) return _mod - P;
		else return 0;
	}
}

/*****************/
/* RINT FUNCTION */
/*****************/
//double rint(double a)
//{
//	const double two_to_52 = 4.5035996273704960e+15;
//	double fa = fabs(a);
//	double r = two_to_52 + fa;
//	if (fa >= two_to_52) {
//		r = a;
//	}
//	else {
//		r = r - two_to_52;
//		r = _copysign(r, a);
//	}
//	return r;
//}

/**********************/
/* FUSED MULTIPLY-ADD */
/**********************/
inline double fma(double x, double y, double z) { return x*y + z; }

/*******************************/
/* MODIFIED BESSEL FUNCTION I0 */
/*******************************/
static double bessi0_CPU(double x)
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
		num = -0.28840544803647313855232E-028;
		num = fma(num, x2, -0.72585406935875957424755E-025);
		num = fma(num, x2, -0.1247819710175804058844059E-021);
		num = fma(num, x2, -0.15795544211478823152992269E-018);
		num = fma(num, x2, -0.15587387207852991014838679E-015);
		num = fma(num, x2, -0.121992831543841162565677055E-012);
		num = fma(num, x2, -0.760147559624348256501094832E-010);
		num = fma(num, x2, -0.375114023744978945259642850E-007);
		num = fma(num, x2, -0.1447896113298369009581404138E-004);
		num = fma(num, x2, -0.4287350374762007105516581810E-002);
		num = fma(num, x2, -0.947449149975326604416967031E+000);
		num = fma(num, x2, -0.1503841142335444405893518061E+003);
		num = fma(num, x2, -0.1624100026427837007503320319E+005);
		num = fma(num, x2, -0.11016595146164611763171787004E+007);
		num = fma(num, x2, -0.4130296432630476829274339869E+008);
		num = fma(num, x2, -0.6768549084673824894340380223E+009);
		num = fma(num, x2, -0.27288446572737951578789523409E+010);

		den = 0.1E+001;
		den = fma(den, x2, -0.38305191682802536272760E+004);
		den = fma(den, x2, 0.5356255851066290475987259E+007);
		den = fma(den, x2, -0.2728844657273795156746641315E+010);

		return num / den;
	}
}

/*****************/
/* INTERPOLATION */
/*****************/
//void Interpolation_NED_CPU(const double2* __restrict data, double2* __restrict result, const double* __restrict x, const double* __restrict y, const int N1, const int N2, int M)
//{
//
//	for (int i = 0; i<M; i++) {
//
//		double cc_points1 = cc*x[i];
//		double r_cc_points1 = rint(cc_points1);
//		const double cc_diff1 = cc_points1 - r_cc_points1;
//
//		double cc_points2 = cc*y[i];
//		double r_cc_points2 = rint(cc_points2);
//		const double cc_diff2 = cc_points2 - r_cc_points2;
//
//		double P1, P2, tempd;
//		int PP1, PP2;
//
//		double phi_cap1, phi_cap2;
//
//		for (int m = 0; m<(2 * K + 1); m++) {
//
//			P1 = K*K - (cc_diff1 - (m - K))*(cc_diff1 - (m - K));
//
//			PP1 = modulo_CPU((r_cc_points1 + (m - K) + N1*cc / 2), (cc*N1));
//
//			if (P1<0.) { tempd = sqrt(-P1); phi_cap1 = (1. / pi_double)*((sin(alfa_CPU*tempd)) / tempd); }
//			else if (P1>0.) { tempd = sqrt(P1); phi_cap1 = (1. / pi_double)*((sinh(alfa_CPU*tempd)) / tempd); }
//			else phi_cap1 = alfa_CPU / pi_double;
//
//			for (int n = 0; n<(2 * K + 1); n++) {
//
//				P2 = K*K - (cc_diff2 - (n - K))*(cc_diff2 - (n - K));
//
//				PP2 = modulo_CPU((r_cc_points2 + (n - K) + N2*cc / 2), (cc*N2));
//
//				if (P2<0.) { tempd = sqrt(-P2); phi_cap2 = (1. / pi_double)*((sin(alfa_CPU*tempd)) / tempd); }
//				else if (P2>0.) { tempd = sqrt(P2); phi_cap2 = (1. / pi_double)*((sinh(alfa_CPU*tempd)) / tempd); }
//				else phi_cap2 = alfa_CPU / pi_double;
//
//				result[IDX2R(PP1, PP2, cc*N2)].x = result[IDX2R(PP1, PP2, cc*N2)].x + data[i].x*phi_cap1*phi_cap2;
//				result[IDX2R(PP1, PP2, cc*N2)].y = result[IDX2R(PP1, PP2, cc*N2)].y + data[i].y*phi_cap1*phi_cap2;
//
//			}
//		}
//
//	}
//
//}
void Interpolation_NED_CPU(const double2* __restrict data, double2* __restrict result, const double* __restrict x, const double* __restrict y, const int N1, const int N2, int M)
{
	
	for (int i = 0; i < M; i++) {

		double cc_points1 = cc*x[i];
		double r_cc_points1 = rint(cc_points1);				// It is the mu in Fourmont's paper
		const double cc_diff1 = cc_points1 - r_cc_points1;

		double cc_points2 = cc*y[i];
		double r_cc_points2 = rint(cc_points2);				// It is the mu in Fourmont's paper
		const double cc_diff2 = cc_points2 - r_cc_points2;

		int PP1, PP2;
		double P, tempd;

		double phi_cap1, phi_cap2;

		if (i < M) {

			for (int m = 0; m < 2 * K + 1; m++) {

				P = K * K - (cc_diff1 - (m - K))*(cc_diff1 - (m - K));

				PP1 = modulo_CPU((r_cc_points1 + (m - K) + N1*cc / 2), cc * N1);

				if (P < 0.) { tempd = 1. / sqrt(-P); phi_cap1 = (1. / pi_double) * ((sin(alfa_CPU / tempd))*tempd); }
				else if (P > 0.) { tempd = 1. / sqrt(P); phi_cap1 = (1. / pi_double) * ((sinh(alfa_CPU / tempd))*tempd); }
				else phi_cap1 = alfa_CPU / pi_double;

				for (int n = 0; n < 2 * K + 1; n++) {

					P = K*K - (cc_diff2 - (n - K))*(cc_diff2 - (n - K));

					PP2 = modulo_CPU((r_cc_points2 + (n - K) + N2*cc / 2), cc*N2);

					if (P < 0.) { tempd = 1. / sqrt(-P); phi_cap2 = phi_cap1*(1. / pi_double)*((sin(alfa_CPU / tempd))*tempd); }
					else if (P > 0.) { tempd = 1. / sqrt(P); phi_cap2 = phi_cap1*(1. / pi_double)*((sinh(alfa_CPU / tempd))*tempd); }
					else phi_cap2 = phi_cap1*(alfa_CPU / pi_double);

					result[IDX2R(PP1, PP2, cc*N2)].x = result[IDX2R(PP1, PP2, cc*N2)].x + data[i].x*phi_cap2;
					result[IDX2R(PP1, PP2, cc*N2)].y = result[IDX2R(PP1, PP2, cc*N2)].y + data[i].y*phi_cap2;

				}
			}
		}
	}

}

/***************/
/* FFTSHIFT 2D */
/***************/
static void fftshift_2D_CPU(double2 *data, int N1, int N2)
{
	for (int i = 0; i<N1; i++)
		for (int j = 0; j<N2; j++) {
			//data[j*N1+i].x *= 1.-2*((i+j)&1);
			//data[j*N1+i].y *= 1.-2*((i+j)&1);
			data[i*N2 + j].x *= 1. - 2 * ((i + j) & 1);
			data[i*N2 + j].y *= 1. - 2 * ((i + j) & 1);
		}
}

/**************************************/
/* CALCULATION OF THE BESSEL FUNCTION */
/**************************************/
static void Calculate_Bessel(double* __restrict Bessel_vector, const int N)
{
	for (int i = 0; i<N; i++) {
		double xi = (2 * pi_double*(i - (N / 2))) / (cc*N);
		Bessel_vector[i] = 1 / (bessi0_CPU(K*sqrt(alfa_CPU*alfa_CPU - xi*xi)));
	}
}

/**************************/
/* DECIMATION AND SCALING */
/**************************/
void Decimation_and_Scaling_CPU(const double2* __restrict data, double2* __restrict result, const double* __restrict Bessel_vector_x, const double* __restrict Bessel_vector_y, const int N1, const int N2)
{
	double a, xi1, xi2;
	for (int i = 0; i<cc*N1; i++)
	{
		for (int j = 0; j<cc*N2; j++)
		{
			if ((((i >= (cc - 1)*N1 / 2) && (i < (cc + 1)*N1 / 2)) && ((j >= (cc - 1)*N2 / 2) && (j < (cc + 1)*N2 / 2))))
			{
				xi1 = 2.*pi_double*(i - (cc - 1)*N1 / 2 - N1 / 2) / (cc*N1);
				xi2 = 2.*pi_double*(j - (cc - 1)*N2 / 2 - N2 / 2) / (cc*N2);

				a = Bessel_vector_x[i - (cc - 1)*N1 / 2] * Bessel_vector_y[j - (cc - 1)*N2 / 2];

				result[IDX2R(i - (cc - 1)*N1 / 2, j - (cc - 1)*N2 / 2, N2)].x = data[IDX2R(i, j, cc*N2)].x*a;
				result[IDX2R(i - (cc - 1)*N1 / 2, j - (cc - 1)*N2 / 2, N2)].y = data[IDX2R(i, j, cc*N2)].y*a;
			}
		}
	}
}

/****************************/
/* SCALING AND ZERO PADDING */
/****************************/
static void ZeroPadding_CPU(const double2* __restrict data, double2* __restrict U, const double* __restrict Bessel_vector_x, const double* __restrict Bessel_vector_y, const int N1, const int N2)
{
	double a;
	int ind_i, ind_j;

	//for (int i=0; i<cc*N1; i++)
	//	for (int j=0; j<cc*N2; j++)
	//	{
	//		if((((i >= (cc-1)*N1/2)  && (i < (cc+1)*N1/2)) || ((j >= (cc-1)*N2/2)  && (j < (cc+1)*N2/2)))) {
	//           U[IDX2R(i,j,cc*N2)].x =0;
	//           U[IDX2R(i,j,cc*N2)].y =0;
	//       }else{
	//           ind_i = (i+N1+(N1/2))&(N1-1); // change to (i+N1+(N1/2))%N1; if N1 is not power of 2
	//           ind_j = (j+N2+(N2/2))&(N2-1); // change to (j+N2+(N2/2))%N2; if N2 is not power of 2

	//		a = Bessel_vector_x[ind_i]*Bessel_vector_y[ind_j];

	//		U[IDX2R(i,j,cc*N2)].x = data[IDX2R(ind_i,ind_j,N2)].x*a;
	//		U[IDX2R(i,j,cc*N2)].y = data[IDX2R(ind_i,ind_j,N2)].y*a;}
	//   }

	for (int i = 0; i<cc*N1; i++)
		for (int j = 0; j<cc*N2; j++)
			if ((((i >= (cc - 1)*N1 / 2) && (i < (cc + 1)*N1 / 2)) && ((j >= (cc - 1)*N2 / 2) && (j < (cc + 1)*N2 / 2)))) {
				a = Bessel_vector_x[i - (cc - 1)*N1 / 2] * Bessel_vector_y[j - (cc - 1)*N2 / 2];

				//printf("%i %i %i %i %f %f %f\n",i,(cc-1)*N1/2,i-(cc-1)*N1/2,j-(cc-1)*N2/2,data[IDX2R(i-(cc-1)*N1/2,j-(cc-1)*N2/2,N2)].x,data[IDX2R(i-(cc-1)*N1/2,j-(cc-1)*N2/2,N2)].y,1./a);

				U[IDX2R(i, j, cc*N2)].x = data[IDX2R(i - (cc - 1)*N1 / 2, j - (cc - 1)*N2 / 2, N2)].x*a;
				U[IDX2R(i, j, cc*N2)].y = data[IDX2R(i - (cc - 1)*N1 / 2, j - (cc - 1)*N2 / 2, N2)].y*a;
				//U[IDX2R(i,j,cc*N2)].x = data[IDX2R(j-(cc-1)*N2/2,i-(cc-1)*N1/2,N1)].x*a;
				//U[IDX2R(i,j,cc*N2)].y = data[IDX2R(j-(cc-1)*N2/2,i-(cc-1)*N1/2,N1)].y*a;
			}
			else {
				U[IDX2R(i, j, cc*N2)].x = 0.;
				U[IDX2R(i, j, cc*N2)].y = 0.;
			}

}

/*************************/
/* NUFFT NED 2D FUNCTION */
/*************************/
extern "C" {
	__declspec(dllexport)
		void NFFT2_2D_CPU(double2* __restrict result, const double2* __restrict data, const double* __restrict x, const double* __restrict y, const int N1, const int N2, const int M)
	{

		//for (int i = 0; i<M; i++) std::cout << "test C " << i << " " << data[i].x << " " << data[i].y << "\n";

		/* CALCULATION OF BESSEL FUNCTIONS */
		double* Bessel_vector_x; Bessel_vector_x = (double*)malloc(N1 * sizeof(double));
		double* Bessel_vector_y; Bessel_vector_y = (double*)malloc(N2 * sizeof(double));

		Calculate_Bessel(Bessel_vector_x, N1);
		Calculate_Bessel(Bessel_vector_y, N2);

		//std::ofstream outfile;
		//outfile.open("Besselx_CPU.txt");
		//for(int i=0; i<N1; i++) { outfile << std::setprecision(10) << Bessel_vector_x[i] << "\n"; }
		//outfile.close();

		//for (int i = 0; i < N1; i++) printf("Bessel vector x CPU %d = %2.23f\n", i, Bessel_vector_x[i]);
		//for (int i=0; i<N2; i++) printf("Bessel vector y CPU %d = %2.23f\n", i, Bessel_vector_y[i]);

		/* ALLOCATIONS AND INITIALIZATIONS */
		temp_result_NUFFT = (double2 *)malloc(cc*N1*cc*N2 * sizeof(double2));
		memset(temp_result_NUFFT, 0, cc*N1*cc*N2 * sizeof(double2));

		/* INTERPOLATION */
		Interpolation_NED_CPU(data, temp_result_NUFFT, x, y, N1, N2, M);
		//for (int i = 0; i<cc*N1*cc*N2; i++) std::cout << "test C " << i << " " << temp_result_NUFFT[i].x << " " << temp_result_NUFFT[i].y << "\n";

		//ofstream outfile;
		//outfile.open("test.txt");
		//for (int i = 0; i < 200; i++) { outfile << std::setprecision(10) << temp_result_NUFFT[i].x << "\n"; }
		//outfile.close();

		/* FFTSHIFT 2D */
		fftshift_2D_CPU(temp_result_NUFFT, cc*N1, cc*N2);
		//for (int i = 0; i<cc*N1*cc*N2; i++) std::cout << "test C " << i << " " << temp_result_NUFFT[i].x << " " << temp_result_NUFFT[i].y << "\n";

		/* FFT */
		Calculate_FFTW_plan(N1, N2);
		fftw_execute(plan);
		Destroy_FFTW_plan();
		//for (int i = 0; i<cc*N1*cc*N2; i++) std::cout << "test C " << i << " " << temp_result_NUFFT[i].x << " " << temp_result_NUFFT[i].y << "\n";

		/* FFTSHIFT 2D */
		fftshift_2D_CPU(temp_result_NUFFT, cc*N1, cc*N2);

		/* DECIMATION AND SCALING */
		Decimation_and_Scaling_CPU(temp_result_NUFFT, result, Bessel_vector_x, Bessel_vector_y, N1, N2);

	}
}

