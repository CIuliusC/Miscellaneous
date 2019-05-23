import os
import sys
import numpy    as      np
import cmath
import ctypes
from   ctypes   import  *

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
a               = 10. * lamb                                    # --- Radius of the cylinder

######################
# RANDOM CURRENT BOX #
######################
eps             = 1e-4
M               = 16
N               = M
h_x             = 2. * a * np.random.rand(1, N) - a 
h_y             = 2. * a * np.random.rand(1, N) - a
h_s             = beta * np.cos(np.linspace(0., 2. * np.pi, num = M)) 
h_t             = beta * np.sin(np.linspace(0., 2. * np.pi, num = M))
h_f             = np.random.rand(1, N) + 1j * np.random.rand(1, N) 

Nruns           = 10

d_x             = gpuarray.to_gpu(h_x)
d_y             = gpuarray.to_gpu(h_y)
d_s             = gpuarray.to_gpu(h_s)
d_t             = gpuarray.to_gpu(h_t)
d_f             = gpuarray.to_gpu(h_f)
d_F             = gpuarray.zeros((1, N), dtype = np.complex128)

#print(type(eps))
#dd.restype = c_double
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

h_F_C           = np.zeros((1, N), dtype = np.complex128)
lib.NFFT3_2D_CPU(h_x.ctypes.data_as(doublep), 
                 h_y.ctypes.data_as(doublep), 
                 h_s.ctypes.data_as(doublep), 
                 h_t.ctypes.data_as(doublep), 
                 h_f.ctypes.data_as(doublep), 
                 h_F_C.ctypes.data_as(doublep), 
                 eps, 
                 N,
                 M)


h_F = np.transpose(NUDFT3_2D(h_x, h_y, h_s, h_t, h_f))

h_F_CUDA = d_F.get()
exponent = np.ones((M, N), dtype = np.int32)
errNorm1 = 100 * np.sqrt(np.sum(np.float_power(np.abs(h_F - h_F_CUDA), exponent)) / np.sum(np.float_power(np.abs(h_F), exponent)))
errNorm2 = 100 * np.sqrt(np.sum(np.float_power(np.abs(h_F - h_F_C), exponent)) / np.sum(np.float_power(np.abs(h_F), exponent)))
print(errNorm1)
print(errNorm2)


print(h_F)
print(h_F_CUDA)
print(h_F_C)

""" timeNUDFT = 0;
for k = 1 : Nruns
    tic; 
%     F1 = NUDFT_3T_2D(x, y, f, s, t); 
%     F1 = NUDFT_3T_2D(Center_Plus(1, :), Center_Plus(2, :), 0.5 * EdgeLength .* RHO_Plus(1, :) .* f, s, t); 
%     F1 = F1 + NUDFT_3T_2D(Center_Minus(1, :), Center_Minus(2, :), 0.5 * EdgeLength .* RHO_Minus(1, :) .* f, s, t); 
    timeNUDFT = timeNUDFT + toc; 
end
fprintf('Time NUDFT Matlab = %1.8f\n', timeNUDFT / Nruns)  """

#timeNUFFTCUDA = 0;
#for k in range(Nruns):
    #tic; 
    #[F3r, F3i] = NUFFT3_2D_CUDA(x, y, s, t, complex(f), eps); F3 = F3r + 1i * F3i; 
    #timeNUFFTCUDA = timeNUFFTCUDA + toc; 
#fprintf('Time NUFFT CUDA = %1.8f\n', timeNUFFTCUDA / Nruns);

#lib.testDLL(3)