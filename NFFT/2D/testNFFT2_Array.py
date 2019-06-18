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

beta            = 2 * np.pi / lamb

# --- M_x * M_y input points
M_x         = 20
M_y         = 20

Deltax      = lamb / 2
Deltay      = lamb / 2

h_csi       = np.linspace(-1., 1., M_x, dtype = np.float64)
h_eta       = np.linspace(-1., 1., M_y, dtype = np.float64)

h_x         = Deltax * M_x * h_csi * h_csi
h_y         = Deltay * M_y * h_eta * h_eta

h_X, h_Y    = np.meshgrid(h_x, h_y)
h_x         = np.reshape(h_X, (1, M_x * M_y))
h_y         = np.reshape(h_Y, (1, M_x * M_y))

#h_x         = (Deltax * M_x) * (np.random.rand(1, M_x * M_y) - 0.5) * 2 / lamb
#h_y         = (Deltax * M_y) * (np.random.rand(1, M_x * M_y) - 0.5) * 2 / lamb

# --- N x M output points
N           = 4 * 40
M           = 4 * 40

a           = np.log10(np.exp(1)) / (M_x * Deltax / 2)
h_data      = np.exp(-a * np.sqrt(h_x * h_x + h_y * h_y)) + 1j * np.zeros(h_x.shape, dtype = np.float64)

d_x         = gpuarray.to_gpu(h_x)
d_y         = gpuarray.to_gpu(h_y)
d_data      = gpuarray.to_gpu(h_data)

d_result    = gpuarray.zeros((M, N), dtype = np.complex128)
lib.NFFT2_2D_GPU.argtypes = [POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), c_int, c_int, c_int]
lib.NFFT2_2D_GPU(ctypes.cast(d_result.ptr, POINTER(c_double)), 
                ctypes.cast(d_data.ptr, POINTER(c_double)), 
                ctypes.cast(d_x.ptr, POINTER(c_double)), 
                ctypes.cast(d_y.ptr, POINTER(c_double)), 
                N, 
                M, 
                M_x * M_y)
h_result_CUDA   = d_result.get()

h_result_py     = NUDFT2_2D(h_x, h_y, h_data, M, N)

errNorm         = 100 * np.sqrt(np.sum(np.float_power(np.abs(h_result_CUDA - h_result_py), 2)) / np.sum(np.float_power(np.abs(h_result_py), 2)))

h_n         = np.linspace(-1., 1., N, dtype = np.float64)
h_m         = np.linspace(-1., 1., M, dtype = np.float64)

h_N, h_M    = np.meshgrid(h_n, h_m)

spectralFilter = np.zeros((M, N), dtype = np.float64)
indices = np.where((h_N * h_N + h_M * h_M) < 1.)
spectralFilter[indices] = 1.

print(errNorm)
CS = plt.contour(20. * np.log10(np.abs(h_result_CUDA * spectralFilter) / np.max(np.abs(h_result_CUDA * spectralFilter))), levels = [-50, -40, -30, -20, -10, -7, -5, -3, -1, 0], 
            extent = [-1, 1, -1, 1])
plt.xlabel(r'$u/\beta$')
plt.ylabel(r'$v/\beta$')
# Make a colorbar for the ContourSet returned by the contourf call.
cbar = plt.colorbar(CS)
#cbar.ax.set_ylabel('verbosity coefficient')
# Add the contour line levels to the colorbar
cbar.add_lines(CS)

plt.show()
