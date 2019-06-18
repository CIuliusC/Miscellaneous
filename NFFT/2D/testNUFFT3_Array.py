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
# NUDFT3_2D #
#############
def NUDFT3_2D(h_x, h_y, h_s, h_t, h_f):

    h_x = np.reshape(h_x, (1, h_x.size))
    h_y = np.reshape(h_y, (1, h_y.size))
    h_s = np.reshape(h_s, (h_s.size, 1))
    h_t = np.reshape(h_t, (h_t.size, 1))
    h_f = np.reshape(h_f, (h_f.size, 1))
#    F = np.transpose(np.exp(-1j * np.dot((np.dot(np.transpose(h_s), h_x) + np.dot(np.transpose(h_t), h_y)), np.transpose(h_f))))
    F = np.matmul(np.exp(-1j * (np.matmul(h_s, h_x) + np.matmul(h_t, h_y))), h_f)

    return F

######################
# PROBLEM PARAMETERS #
######################
lamb            = 1.                                            # --- Wavelength
beta            = 2. * np.pi / lamb                             # --- Wavenumber

eps             = 1e-3

# --- M_x * M_y input points
M_x             = 20
M_y             = 20
N               = M_x * M_y
#N              = M_x

Deltax          = lamb / 2
Deltay          = lamb / 2

h_csi           = np.linspace(-1., 1., M_x, dtype = np.float64)
h_eta           = np.linspace(-1., 1., M_y, dtype = np.float64)

h_x             = Deltax * M_x * h_csi * h_csi
h_y             = Deltay * M_y * h_eta * h_eta

h_X, h_Y        = np.meshgrid(h_x, h_y)
h_x             = np.reshape(h_X, (1, M_x * M_y))
h_y             = np.reshape(h_Y, (1, M_x * M_y))

#h_x         = (Deltax * M_x) * (np.random.rand(1, M_x * M_y) - 0.5) * 2 / lamb
#h_y         = (Deltax * M_y) * (np.random.rand(1, M_x * M_y) - 0.5) * 2 / lamb

# --- M_s * M_t output points
M_s             = 4 * 20
M_t             = 4 * 20
M               = M_s * M_t
#M              = M_s

h_s             = (2. * beta / M) * (np.arange(0, M, dtype = np.float64) - M / 2)
h_t             = 0. * (2. * beta / M) * (np.arange(0, M, dtype = np.float64) - M / 2)
#h_s             = beta * np.cos(np.linspace(0., 2. * np.pi, num = M)) 
#h_t             = beta * np.sin(np.linspace(0., 2. * np.pi, num = M))

""" h_x     = beta * (np.random.rand(1, M_x * M_y) - 0.5)
h_y     = beta * (np.random.rand(1, M_x * M_y) - 0.5)
 """

a               = np.log10(np.exp(1)) / (M_x * Deltax / 2)
h_f             = np.exp(-a * np.sqrt(h_x * h_x + h_y * h_y)) + 1j * np.zeros(h_x.shape, dtype = np.float64)

d_x             = gpuarray.to_gpu(h_x)
d_y             = gpuarray.to_gpu(h_y)
d_f             = gpuarray.to_gpu(h_f)

d_s             = gpuarray.to_gpu(h_s)
d_t             = gpuarray.to_gpu(h_t)
d_f             = gpuarray.to_gpu(h_f)
d_F             = gpuarray.zeros((1, M), dtype = np.complex128)

lib.NFFT3_2D_GPU.argtypes = [POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), c_double, c_int, c_int]
lib.NFFT3_2D_GPU(ctypes.cast(d_x.ptr, POINTER(c_double)), 
                ctypes.cast(d_y.ptr, POINTER(c_double)), 
                ctypes.cast(d_s.ptr, POINTER(c_double)), 
                ctypes.cast(d_t.ptr, POINTER(c_double)), 
                ctypes.cast(d_f.ptr, POINTER(c_double)), 
                ctypes.cast(d_F.ptr, POINTER(c_double)), 
                np.float64(eps),
                N, 
                M)

lib.NFFT3_2D_CPU.argtypes = [POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), c_double, c_int, c_int]
doublep = ctypes.POINTER(ctypes.c_double)

""" h_F_C           = np.zeros((1, M), dtype = np.complex128)
lib.NFFT3_2D_CPU(h_x.ctypes.data_as(doublep), 
                h_y.ctypes.data_as(doublep), 
                h_s.ctypes.data_as(doublep), 
                h_t.ctypes.data_as(doublep), 
                h_f.ctypes.data_as(doublep), 
                h_F_C.ctypes.data_as(doublep), 
                eps, 
                N,
                M) """

h_F             = NUDFT3_2D(h_x, h_y, h_s, h_t, h_f)

h_result_CUDA   = d_F.get()
h_result_CUDA   = np.reshape(h_result_CUDA, (M, 1))

errNorm         = 100 * np.sqrt(np.sum(np.float_power(np.abs(h_result_CUDA - h_F), 2)) / np.sum(np.float_power(np.abs(h_F), 2)))

print(errNorm)
#plt.plot(np.abs(h_F), color = 'r')
#print(h_result_CUDA.shape)
plt.plot(h_s / beta, 20. * np.log10(np.abs(h_result_CUDA) / np.max(np.abs(h_result_CUDA))), color = 'b')
plt.plot(h_s / beta, 20. * np.log10(np.abs(h_F) / np.max(np.abs(h_F))), color = 'r', marker = '*')
plt.xlabel(r'$u / \beta$')
plt.ylabel('Far field [dB]')
plt.show()