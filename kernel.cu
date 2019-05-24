#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <fstream>		// --- ifstream
#include <sstream>		// --- stringstream
#include <string>

#include <helper_cuda.h>

typedef unsigned int uint;
typedef unsigned char uchar;

#include "bicubicTexture_kernel.cuh"

float tx = 100.0f, ty = 100.0f;		// --- Image translation
float scale = 1.0f / 8.0f;		// --- Image scale
float xCenter, yCenter;			// --- Image centre
										
/******************/
/* ERROR CHECKING */
/******************/
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

cudaArray *d_imageArray = 0;

enum eFilterMode
{
	MODE_NEAREST,
	MODE_BILINEAR,
	MODE_BICUBIC,
	MODE_FAST_BICUBIC,
	MODE_CATMULL_ROM,
	NUM_MODES,
	MODE_CATROM
}; 

/**************************/
/* TEXTURE INITIALIZATION */
/**************************/
void initTexture(int imageWidth, int imageHeight, uchar *h_data)
{
	// --- Allocate CUDA array and copy image data
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
	gpuErrchk(cudaMallocArray(&d_imageArray, &channelDesc, imageWidth, imageHeight));
	uint size = imageWidth * imageHeight * sizeof(uchar);
	gpuErrchk(cudaMemcpyToArray(d_imageArray, 0, 0, h_data, size, cudaMemcpyHostToDevice));
	free(h_data);

	// --- Set up texture
	tex.addressMode[0] = cudaAddressModeClamp;
	tex.addressMode[1] = cudaAddressModeClamp;
	tex.filterMode = cudaFilterModeLinear;
	tex.normalized = false;    

	// --- Texture binding
	gpuErrchk(cudaBindTextureToArray(tex, d_imageArray));

}

extern "C"
void freeTexture()
{
	checkCudaErrors(cudaFreeArray(d_imageArray));
}


// render image using CUDA
//void render(int width, int height, float tx, float ty, float scale, float xCenter, float yCenter,
//	dim3 blockSize, dim3 gridSize, int filter_mode, uchar4 *output)
//{
//	// call CUDA kernel, writing results to PBO memory
//	switch (filter_mode)
//	{
//	case MODE_NEAREST:
//		tex.filterMode = cudaFilterModePoint;
//		d_render << <gridSize, blockSize >> >(output, width, height, tx, ty, scale, xCenter, yCenter);
//		break;
//
//	case MODE_BILINEAR:
//		tex.filterMode = cudaFilterModeLinear;
//		d_render << <gridSize, blockSize >> >(output, width, height, tx, ty, scale, xCenter, yCenter);
//		break;
//
//	case MODE_BICUBIC:
//		tex.filterMode = cudaFilterModePoint;
//		d_renderBicubic << <gridSize, blockSize >> >(output, width, height, tx, ty, scale, xCenter, yCenter);
//		break;
//
//	case MODE_FAST_BICUBIC:
//		tex.filterMode = cudaFilterModeLinear;
//		d_renderFastBicubic << <gridSize, blockSize >> >(output, width, height, tx, ty, scale, xCenter, yCenter);
//		break;
//
//	case MODE_CATROM:
//		tex.filterMode = cudaFilterModePoint;
//		d_renderCatRom << <gridSize, blockSize >> >(output, width, height, tx, ty, scale, xCenter, yCenter);
//		break;
//	}
//
//	getLastCudaError("kernel failed");
//}


// CUDA system and GL includes
#include <cuda_runtime.h>

// Helper functions
#include <helper_functions.h> // CUDA SDK Helper functions
#include <helper_cuda.h>      // CUDA device initialization helper functions


typedef unsigned int uint;
typedef unsigned char uchar;

//void initTexture(int imageWidth, int imageHeight, uchar *h_data);

//uint width = 512, height = 512;
uint imageWidth, imageHeight;
dim3 blockSize(16, 16);
// dim3 gridSize(width / blockSize.x, height / blockSize.y);
//dim3 gridSize(imageWidth / blockSize.x, imageHeight / blockSize.y);

//eFilterMode g_FilterMode = MODE_FAST_BICUBIC;
//eFilterMode g_FilterMode = MODE_NEAREST;
eFilterMode g_FilterMode = MODE_BILINEAR;

/******************/
/* LOAD PGM IMAGE */
/******************/
void loadPGMImageAndInitTexture(const char *inputFilename) {

	std::cout << "Opening file " << inputFilename << std::endl;
	std::ifstream infile(inputFilename, std::ios::binary);
	std::stringstream ss;
	std::string inputLine = "";

	// --- Read the first line
	getline(infile, inputLine);
	if (inputLine.compare("P5") != 0) std::cerr << "Version error" << std::endl;
	std::cout << "Version : " << inputLine << std::endl;

	std::string identifier;
	std::stringstream::pos_type pos = ss.tellg();
	ss >> identifier;
	if (identifier == "#") {
		// --- If second line is a comment, display the comment
		std::cout << "Comment: " << identifier << std::endl;
	}
	else {
		// --- If second line is not a comment, rewind
		ss.clear();
		ss.seekg(pos, ss.beg);
	}

	// --- Read the third line : width, height
	ss << infile.rdbuf(); 
	ss >> imageWidth >> imageHeight;
	std::cout << "Image size is " << imageWidth << " columns and " << imageHeight << " rows" << std::endl;
	// --- Maximum intensity value
	int max_val;
	ss >> max_val;
	std::cout << "Image maximum intensity is " << max_val << std::endl;

	uchar pixel;
	uchar *h_data = (uchar *)malloc(imageHeight * imageWidth * sizeof(uchar));

	for (int row = 0; row < imageHeight; row++) {//record the pixel values
		for (int col = 0; col < imageWidth; col++) {
			ss.read((char *)&pixel, 1);
			h_data[row * imageWidth + col] = pixel;
		}
	}

	xCenter = imageWidth * 0.5f;
	yCenter = imageHeight * 0.5f;

	// --- Texture initialization
	initTexture(imageWidth, imageHeight, h_data);

}

/*******************/
/* WRITE PGM IMAGE */
/*******************/
void writePGMImage(const char *outputFilename, uchar *h_data, const int imageWidth, const int imageHeight) {

	std::ofstream f(outputFilename, std::ios_base::out | std::ios_base::binary | std::ios_base::trunc);

	int maxColorValue = 255;
	f << "P5\n" << imageWidth << " " << imageHeight << "\n" << maxColorValue << "\n";

	for (int i = 0; i < imageHeight; ++i)
		f.write(reinterpret_cast<const char*>(&h_data[i * imageWidth]), imageWidth);

	//if (wannaFlush)
		f << std::flush;
}

//extern "C" void render(int width, int height, float tx, float ty, float scale, float xCenter, float yCenter,
//	dim3 blockSize, dim3 gridSize, eFilterMode filter_mode, uchar4 *output);

// w0, w1, w2, and w3 are the four cubic B-spline basis functions
float bspline_w0(float a)
{
	return (1.0f / 6.0f)*(-a*a*a + 3.0f*a*a - 3.0f*a + 1.0f);
}

float bspline_w1(float a)
{
	return (1.0f / 6.0f)*(3.0f*a*a*a - 6.0f*a*a + 4.0f);
}

float bspline_w2(float a)
{
	return (1.0f / 6.0f)*(-3.0f*a*a*a + 3.0f*a*a + 3.0f*a + 1.0f);
}

__host__ __device__
float bspline_w3(float a)
{
	return (1.0f / 6.0f)*(a*a*a);
}

int iDivUp(int a, int b)
{
	return (a % b != 0) ? (a / b + 1) : (a / b);
}

/************************/
/* NERP AND LERP KERNEL */
/************************/
__global__ void nalKernel(uchar * __restrict__ d_output, const uint width, const uint height, const float tx, const float ty,
	const float scale, const float cx, const float cy)
{
	const int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	const int tidy = blockIdx.y * blockDim.y + threadIdx.y;

	float u = (tidx - cx) * scale + cx + tx;
	float v = (tidy - cy) * scale + cy + ty;

	if ((tidx < width) && (tidy < height))
	{
		float c = tex2D(tex, u, v);
		//float c = tex2DBilinear<uchar, float>(tex, u, v);
		//float c = tex2DBilinearGather<uchar, uchar4>(tex2, u, v, 0) / 255.0f;
		d_output[tidy * width + tidx] = c * 0xff;
	}
}

/*******************************/
/* BICUBIC WITH TEXTURE LOOKUP */
/*******************************/
template<class T> __device__
T cubicFilter(float x, T c0, T c1, T c2, T c3)
{
	T r;
	r = c0 * w0(x);
	r += c1 * w1(x);
	r += c2 * w2(x);
	r += c3 * w3(x);
	return r;
}

template<class T, class R> 
__device__ R tex2DBicubic(const texture<T, 2, cudaReadModeNormalizedFloat> texref, float x, float y)
{
	x -= 0.5f;
	y -= 0.5f;
	float px = floor(x);
	float py = floor(y);
	float fx = x - px;
	float fy = y - py;

	return cubicFilter<R>(fy,
		cubicFilter<R>(fx, tex2D(texref, px - 1, py - 1), tex2D(texref, px, py - 1), tex2D(texref, px + 1, py - 1), tex2D(texref, px + 2, py - 1)),
		cubicFilter<R>(fx, tex2D(texref, px - 1, py),     tex2D(texref, px, py),     tex2D(texref, px + 1, py),     tex2D(texref, px + 2, py)),
		cubicFilter<R>(fx, tex2D(texref, px - 1, py + 1), tex2D(texref, px, py + 1), tex2D(texref, px + 1, py + 1), tex2D(texref, px + 2, py + 1)),
		cubicFilter<R>(fx, tex2D(texref, px - 1, py + 2), tex2D(texref, px, py + 2), tex2D(texref, px + 1, py + 2), tex2D(texref, px + 2, py + 2)));
}

__global__ void d_renderBicubic(uchar *d_output, const uint width, const uint height, const float tx, const float ty,
	const float scale, const float cx, const float cy)
{
	const int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	const int tidy = blockIdx.y * blockDim.y + threadIdx.y;

	float u = (tidx - cx) * scale + cx + tx;
	float v = (tidy - cy) * scale + cy + ty;

	if ((tidx < width) && (tidy < height))
	{
		float c = tex2DBicubic<uchar, float>(tex, u, v);
		d_output[tidy * width + tidx] = c * 0xff;
	}
}

/********/
/* MAIN */
/********/
// --- https://www.geeksforgeeks.org/c-program-to-write-an-image-in-pgm-format/
// --- https://stackoverflow.com/questions/39797167/distorted-image-in-pgm-binary-file-manipulation
// --- http://people.sc.fsu.edu/~jburkardt/data/pgmb/pgmb.html
// --- http://www.cplusplus.com/forum/general/2393/
int main()
{
	char *inputFilename  = "D:\\Project\\Packt\\bicubicTexture\\data\\barbara.pgm";
	char *outputFilename = "D:\\Project\\Packt\\bicubicTexture\\data\\interpolationResult.pgm";

	loadPGMImageAndInitTexture(inputFilename);

	//uchar4 *d_output; checkCudaErrors(cudaMalloc(&d_output, imageWidth*imageHeight * 4));
	//unsigned int *h_result = (unsigned int *)malloc(imageWidth * imageHeight * sizeof(unsigned int));

	uchar *d_output; gpuErrchk(cudaMalloc(&d_output, imageWidth * imageHeight * sizeof(uchar)));
	uchar *h_result = (uchar *)malloc(imageWidth * imageHeight * sizeof(uchar));

	dim3 gridSize(imageWidth / blockSize.x, imageHeight / blockSize.y);
	//render(imageWidth, imageHeight,
	//	tx, ty, scale, xCenter, yCenter,
	//	blockSize, gridSize, g_FilterMode, d_output);

	// --- Nearest neighbor filtering
	//tex.filterMode = cudaFilterModePoint;
	tex.filterMode = cudaFilterModeLinear;
	nalKernel << <gridSize, blockSize >> >(d_output, imageWidth, imageHeight, tx, ty, scale, xCenter, yCenter);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	// check if kernel execution generated an error
	getLastCudaError("Error: render (bicubicTexture) Kernel execution FAILED");
	checkCudaErrors(cudaDeviceSynchronize());

	//cudaMemcpy(h_result, d_output, imageWidth*imageHeight * 4, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_result, d_output, imageWidth * imageHeight * sizeof(uchar), cudaMemcpyDeviceToHost);

	//uchar *h_result1channel = (uchar *)malloc(imageWidth * imageHeight * sizeof(uchar));
	//uchar *h_tmp = reinterpret_cast<uchar *>(h_result);

	//for (int k = 0; k < imageWidth * imageHeight; k++) h_result1channel[k] = h_tmp[4 * k];

	//writePGMImage(outputFilename, h_result1channel, imageWidth, imageHeight);
	writePGMImage(outputFilename, h_result, imageWidth, imageHeight);

	checkCudaErrors(cudaFree(d_output));
	free(h_result);


	return 0;
}
