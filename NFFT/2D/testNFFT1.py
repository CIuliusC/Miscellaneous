import os
import sys
import numpy    as      np
import cmath
import ctypes
import matplotlib.pyplot  as plt    
from   ctypes   import  *

import pycuda.driver      as     drv
import pycuda.gpuarray    as     gpuarray
import pycuda.autoinit

FFTWFile    = 'D:\\FFTW64\\libfftw3-3.dll'
FFTW64      = cdll.LoadLibrary(FFTWFile)

lib = cdll.LoadLibrary('D:\\CEM\\ICECOM2019\\NUFFT_Library\\x64\Release\\NUFFT_Library.dll')

#############
# NUFFT2_2D #
#############
def NFFT1_2D(data, x1, x2):

    N1, N2 = data.shape
    M = x1.size

    # --- Algorithm parameters
    c = 2                                                    
    K = 6                                                    
    alfa = (2 - 1 / c) * np.pi - 0.01                                  

    ###################################
    # CONSTRUCTING THE SPATIAL WINDOW #
    ###################################
    kk1     = np.arange(-N1 / 2, N1 / 2, 1)
    kk2     = np.arange(-N2 / 2, N2 / 2, 1)
    xi1     = 2 * np.pi * kk1 / (c * N1)
    xi2     = 2 * np.pi * kk2 / (c * N2)
    phi1    = np.i0(K * np.sqrt(alfa * alfa - xi1 * xi1))
    phi2    = np.i0(K * np.sqrt(alfa * alfa - xi2 * xi2))
    PHI2, PHI1 = np.meshgrid(phi2, phi1)

    ####################################
    # CONSTRUCTING THE SPECTRAL WINDOW #
    ####################################
    mu1 = np.round(c * x1)                                          
    mu2 = np.round(c * x2)                                          

    KK1, MU1 = np.meshgrid(np.arange(-K, K + 1, 1), mu1)
    X1 = np.matmul(np.transpose(x1), np.ones((1, 2 * K + 1), dtype = np.complex128))
    P1 = np.sqrt(K * K - (c * X1 - (MU1 + KK1)) * (c * X1 - (MU1 + KK1)))
    spectrum_phi1 = (1 / np.pi) * np.sinh(alfa * P1) / P1
    spectrum_phi1[np.where(P1 == 0)] = alfa / np.pi

    KK2, MU2 = np.meshgrid(np.arange(-K, K + 1, 1), mu2)
    X2 = np.matmul(np.transpose(x2), np.ones((1, 2 * K + 1), dtype = np.complex128))
    P2 = np.sqrt(K * K - (c * X2 - (MU2 + KK2)) * (c * X2 - (MU2 + KK2)))
    spectrum_phi2 = (1 / np.pi) * np.sinh(alfa * P2) / P2
    spectrum_phi2[np.where(P2 == 0)] = alfa / np.pi

    ####################################
    # STEP 1: SCALING AND ZERO PADDING #
    ####################################
    u = np.zeros((c * N1, c * N2), dtype = np.complex128)
    u[np.int32((c - 1) * N1 / 2) : np.int32((c + 1) * N1 / 2), np.int32((c - 1) * N2 / 2): np.int32((c + 1) * N2 / 2)] = data / (PHI1 * PHI2)

    ##########
    # STEP 2 #
    ##########
    U = np.fft.fft2(np.fft.ifftshift(u))

    ##########
    # STEP 2 #
    ##########
    result = np.zeros((M, 1), dtype = np.complex128)
    for l in range(M):
        for m1 in range(2 * K + 1):
            result[l] = result[l] + np.sum(spectrum_phi1[l, m1] * spectrum_phi2[l, 0 : (2 * K + 1)] * U[np.int32(np.remainder(mu1[0, l] + m1 - K, c * N1)), np.int32(np.remainder(mu2[0, l] + np.arange(-K, K + 1), c * N2))])

    return result

#############
# NUDFT2_2D #
#############
def NUDFT1_2D(h_x, h_y, h_data, M, N):

    h_data   = np.reshape(np.transpose(h_data), (h_data.size, 1))
    
    h_u = np.arange(-N / 2, N / 2, 1)
    h_v = np.arange(-M / 2, M / 2, 1)

    h_U, h_V = np.meshgrid(h_u, h_v)
    h_U      = np.reshape(h_U, (N * M, 1))
    h_V      = np.reshape(h_V, (N * M, 1))

    h_X, h_U = np.meshgrid(h_x, h_U)
    h_Y, h_V = np.meshgrid(h_y, h_V)

    Kernel   = np.exp(-1j * 2. * np.pi * h_X * h_U / N) * np.exp(-1j * 2. * np.pi * h_Y * h_V / M)

    transf   = np.matmul(np.transpose(Kernel), h_data)
   
    return transf

######################
# PROBLEM PARAMETERS #
######################
# --- M_x * M_y output points
M_x             = 12
M_y             = 11

# --- N x M input points
N               = 8
M               = 6

lamb            = 1

h_x             = M_x * (lamb / 2.) * (np.random.rand(1, M_x * M_y) - 0.5)
h_y             = M_y * (lamb / 2.) * (np.random.rand(1, M_x * M_y) - 0.5)
#h_x             = np.ones((1, M_x * M_y), dtype = np.float64)
#h_x             = np.reshape(np.linspace(-1., 1., M_x * M_y), (1, M_x * M_y))
#h_y             = np.zeros((1, M_x * M_y), dtype = np.float64)
h_data          = (np.random.rand(N, M) - 0.5) + 1j * (np.random.rand(N, M) - 0.5)
#h_data          = np.ones((N, M), dtype = np.complex128)

#print(h_data)
d_x             = gpuarray.to_gpu(h_x)
d_y             = gpuarray.to_gpu(h_y)
d_data          = gpuarray.to_gpu(h_data)

d_result        = gpuarray.zeros((M_x * M_y, 1), dtype = np.complex128)
h_result_C      = np.zeros((M_x * M_y, 1), dtype = np.complex128)

lib.NFFT1_2D_GPU.argtypes = [POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), c_int, c_int, c_int]
lib.NFFT1_2D_GPU(ctypes.cast(d_result.ptr, POINTER(c_double)), 
                 ctypes.cast(d_data.ptr, POINTER(c_double)), 
                 ctypes.cast(d_x.ptr, POINTER(c_double)), 
                 ctypes.cast(d_y.ptr, POINTER(c_double)), 
                 N, 
                 M,
                 M_x * M_y)

lib.NFFT1_2D_CPU.argtypes = [POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), c_int, c_int, c_int]
doublep = ctypes.POINTER(ctypes.c_double)

lib.NFFT1_2D_CPU(h_result_C.ctypes.data_as(doublep), 
                 h_data.ctypes.data_as(doublep), 
                 h_x.ctypes.data_as(doublep), 
                 h_y.ctypes.data_as(doublep), 
                 N, 
                 M,
                 M_x * M_y)

h_result = NUDFT1_2D(h_x, h_y, h_data, M, N)

h_result_NFFT_Python = NFFT1_2D(h_data, h_x, h_y)

""" print('h_result')
print(h_result)
print('h_result_C')
print(h_result_C)
 """
h_result_CUDA = d_result.get()

""" print('h_result_CUDA')
print(h_result_CUDA)
 """
#print('h_result_C')
#print(h_result_C)

exponent = np.ones((h_result.size, 1), dtype = np.int32)
errNorm1 = 100 * np.sqrt(np.sum(np.float_power(np.abs(h_result - h_result_C), exponent)) / np.sum(np.float_power(np.abs(h_result), exponent)))
errNorm2 = 100 * np.sqrt(np.sum(np.float_power(np.abs(h_result - h_result_CUDA), exponent)) / np.sum(np.float_power(np.abs(h_result), exponent)))
errNorm3 = 100 * np.sqrt(np.sum(np.float_power(np.abs(h_result - h_result_NFFT_Python), exponent)) / np.sum(np.float_power(np.abs(h_result), exponent)))
print(errNorm1)
print(errNorm2)
print(errNorm3)

#print(h_result_NFFT_Python)

""" plt.plot(np.abs(h_result), color = 'k')
#plt.show()
plt.plot(np.abs(h_result_C), color = 'g')
plt.show()
 """

""" h_result_CUDA = d_result.get()
print(h_result_CUDA)
print(h_result_C)
print(h_result)
 """
""" plt.imshow(np.abs(h_result))                       #max(max(im)))=0.0143  
plt.show()

plt.imshow(np.abs(h_result_CUDA))                       #max(max(im)))=0.0143  
plt.show()

plt.imshow(np.abs(h_result_C))                       #max(max(im)))=0.0143  
plt.show() """
