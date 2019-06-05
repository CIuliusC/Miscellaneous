import os
import sys
import numpy    as      np
import cmath
import ctypes
from   ctypes   import  *
import matplotlib.pyplot  as plt    
from time import time

import pycuda.driver      as     drv
import pycuda.gpuarray    as     gpuarray
import pycuda.autoinit

FFTWFile    = 'D:\\FFTW64\\libfftw3-3.dll'
FFTW64      = cdll.LoadLibrary(FFTWFile)

lib = cdll.LoadLibrary('D:\\CEM\\ICECOM2019\\NUFFT_Library\\x64\\Release\\NUFFT_Library.dll')

#############
# NUDFT2_2D #
#############
def NUDFT2_2D(h_x, h_y, h_data, M, N):

    h_data   = np.reshape(h_data, (h_data.size, 1))
    
    h_u = np.arange(-N / 2, N / 2, 1)
    h_v = np.arange(-M / 2, M / 2, 1)

    h_U, h_V = np.meshgrid(h_u, h_v)
    h_U      = np.reshape(h_U, (N * M, 1))
    h_V      = np.reshape(h_V, (N * M, 1))

    h_X, h_U = np.meshgrid(h_x, h_U)
    h_Y, h_V = np.meshgrid(h_y, h_V)

    Kernel   = np.exp(-1j * 2. * np.pi * h_X * h_U / N) * np.exp(-1j * 2. * np.pi * h_Y * h_V / M)

    transf   = np.transpose(np.reshape(np.matmul(Kernel, h_data), (M, N)))
    transf   = np.reshape(np.reshape(transf, (M * N, 1)), (M, N))
   
    return transf

######################
# PROBLEM PARAMETERS #
######################
lamb            = 1

maxExp          = 10
minExp          = 4
timingCPU       = np.zeros((1, maxExp - minExp), dtype = np.float64)
timingGPU       = np.zeros((1, maxExp - minExp), dtype = np.float64)
speedUp         = np.zeros((1, maxExp - minExp), dtype = np.float64)
errNorms        = np.zeros((1, maxExp - minExp), dtype = np.float64)
for k in range(minExp, maxExp):

    # --- M_x * M_y input points
    M_x         = np.power(2, k)
    M_y         = np.power(2, k)

    # --- N x M output points
    N           = np.power(2, k)
    M           = np.power(2, k)

    h_x         = M_x * (lamb / 2.) * (np.random.rand(1, M_x * M_y) - 0.5)
    h_y         = M_y * (lamb / 2.) * (np.random.rand(1, M_x * M_y) - 0.5)
    h_data      = (np.random.rand(1, M_x * M_y) - 0.5) + 1j * (np.random.rand(1, M_x * M_y) - 0.5)

    d_x         = gpuarray.to_gpu(h_x)
    d_y         = gpuarray.to_gpu(h_y)
    d_data      = gpuarray.to_gpu(h_data)

    d_result    = gpuarray.zeros((M, N), dtype = np.complex128)

    lib.NFFT2_2D_GPU.argtypes = [POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), c_int, c_int, c_int]
    time1 = time()
    lib.NFFT2_2D_GPU(ctypes.cast(d_result.ptr, POINTER(c_double)), 
                     ctypes.cast(d_data.ptr, POINTER(c_double)), 
                     ctypes.cast(d_x.ptr, POINTER(c_double)), 
                     ctypes.cast(d_y.ptr, POINTER(c_double)), 
                     N, 
                     M, 
                     M_x * M_y)
    time2 = time()
    timingGPU[0, k - minExp] = (time2 - time1)

    h_result_C      = np.zeros((M, N), dtype = np.complex128)
    lib.NFFT2_2D_CPU.argtypes = [POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), c_int, c_int, c_int]
    doublep = ctypes.POINTER(ctypes.c_double)
    time1 = time()
    lib.NFFT2_2D_CPU(h_result_C.ctypes.data_as(doublep), 
                  h_data.ctypes.data_as(doublep), 
                   h_x.ctypes.data_as(doublep), 
                     h_y.ctypes.data_as(doublep), 
                     N, 
                     M,
                     M_x * M_y)
    time2 = time()
    timingCPU[0, k - minExp] = (time2 - time1)

    h_result_CUDA = d_result.get()
    exponent = 2 * np.ones(h_result_CUDA.shape, dtype = np.int32)
    errNorms[0, k - minExp] = 100 * np.sqrt(np.sum(np.float_power(np.abs(h_result_CUDA - h_result_C), exponent)) / np.sum(np.float_power(np.abs(h_result_C), exponent)))

    speedUp[0, k - minExp] = timingCPU[0, k - minExp] / timingGPU[0, k - minExp]

    #print(timingCPU[0, k - minExp], timingGPU[0, k - minExp], (k - minExp) / (maxExp - minExp))

#h_result = NUDFT2_2D(h_x, h_y, h_data, M, N)

#h_result_CUDA = np.reshape(d_result.get(), (M, N))

print(speedUp)
print(errNorms)
plt.plot(speedUp)
plt.show()

""" exponent = np.ones((M, N), dtype = np.int32)
errNorm1 = 100 * np.sqrt(np.sum(np.float_power(np.abs(h_result - h_result_CUDA), exponent)) / np.sum(np.float_power(np.abs(h_result), exponent)))
#print(h_result_C)
#print(h_result_CUDA)
errNorm2 = 100 * np.sqrt(np.sum(np.float_power(np.abs(h_result - h_result_C), exponent)) / np.sum(np.float_power(np.abs(h_result), exponent)))
print(errNorm1)
print(errNorm2)
 """
