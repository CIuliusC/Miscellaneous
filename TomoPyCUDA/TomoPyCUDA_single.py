import numpy              as np 
import skcuda.linalg      as linalg
import skcuda.linalg      as culinalg
import skcuda.misc        as cumisc
import skcuda.cublas      as cublas
import pycuda.gpuarray    as gpuarray
import pycuda.cumath      as cumath
from   pycuda.elementwise import ElementwiseKernel
import pycuda.driver      as cuda
from   pycuda.compiler    import SourceModule 
import pycuda.autoinit
import pycuda.cumath as cumath
import time

import matplotlib.pyplot  as plt        
import cProfile

culinalg.init()

BLOCKSIZE  = 256
BLOCKSIZEX = 16
BLOCKSIZEY = 16

timing = 0

###################
# iDivUp FUNCTION #
###################
def iDivUp(a, b):
    return a // b + 1

#####################
# A KERNEL FUNCTION #
#####################
funca = ElementwiseKernel(
    "float *Xprime, float *X, float *Yprime, float *Y, float *Zprime, float zzero, int k, pycuda::complex<float> *a, float Deltaprime, float beta",
    """
    float R = sqrt((Xprime[i] - X[k]) * (Xprime[i] - X[k]) + (Yprime[i] - Y[k]) * (Yprime[i] - Y[k]) + (Zprime[i] + zzero) * (Zprime[i] + zzero));
    pycuda::complex<float> phase(0.f, -2.f * beta * R);
    pycuda::complex<float> module(Deltaprime / (R * R), 0.f);
    a[i] = module * exp(phase);
    //a[i] = Y[k];
    """,
    preamble="#include <pycuda-complex.hpp>")

funcAMod = SourceModule("""
  #include <pycuda-complex.hpp>
  __global__ void funcA(float *Xprime, float *X, float *Yprime, float *Y, float *Zprime, float zzero, int k, pycuda::complex<float> *A, float Deltaprime, float *beta, const int N)
  {
    const int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (tidx >= N) return;
    float R = sqrt((Xprime[tidx] - X[k]) * (Xprime[tidx] - X[k]) + (Yprime[tidx] - Y[k]) * (Yprime[tidx] - Y[k]) + (Zprime[tidx] + zzero) * (Zprime[tidx] + zzero));
    pycuda::complex<float> phase(0.f, -2.f * beta[k] * R);
    pycuda::complex<float> module(Deltaprime / (R * R), 0.f);
    A[tidx] = module * exp(phase);
   } 
  """)

funcAMod2D = SourceModule("""
  #include <pycuda-complex.hpp>
  __global__ void funcA2D(float *Xprime, float *X, float *Yprime, float *Y, float *Zprime, float zzero, int k, pycuda::complex<float> *A, float Deltaprime, float *beta, const int N, const int M)
  {
    const int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    const int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    if ((tidx >= N) || (tidy >= M)) return;
    float R = sqrt((Xprime[tidx] - X[tidy]) * (Xprime[tidx] - X[tidy]) + (Yprime[tidx] - Y[tidy]) * (Yprime[tidx] - Y[tidy]) + (Zprime[tidx] + zzero) * (Zprime[tidx] + zzero));
    pycuda::complex<float> phase(0.f, -2.f * beta[tidy] * R);
    pycuda::complex<float> module(Deltaprime / (R * R), 0.f);
    A[tidy * N + tidx] = module * exp(phase);
   } 
  """)

######################
# Ad KERNEL FUNCTION #
######################
funcAd = ElementwiseKernel(
    "float *Xprime, float *X, float *Yprime, float *Y, float *Zprime, float zzero, int k, pycuda::complex<float> *Ad, float Deltaprime, float *beta",
    """
    float R = sqrt((Xprime[k] - X[i]) * (Xprime[k] - X[i]) + (Yprime[k] - Y[i]) * (Yprime[k] - Y[i]) + (Zprime[k] + zzero) * (Zprime[k] + zzero));
    pycuda::complex<float> phase(0.f, 2.f * beta[i] * R);
    pycuda::complex<float> module(Deltaprime / (R * R), 0.f);
    Ad[i] = module * exp(phase);
    """,
    preamble="#include <pycuda-complex.hpp>")
    
funcAdMod = SourceModule("""
  #include <pycuda-complex.hpp>
  __global__ void funcAd(float *Xprime, float *X, float *Yprime, float *Y, float *Zprime, float zzero, int k, pycuda::complex<float> *Ad, float Deltaprime, float *beta, const int N)
  {
    const int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (tidx >= N) return;
    float R = sqrt((Xprime[k] - X[tidx]) * (Xprime[k] - X[tidx]) + (Yprime[k] - Y[tidx]) * (Yprime[k] - Y[tidx]) + (Zprime[k] + zzero) * (Zprime[k] + zzero));
    pycuda::complex<float> phase(0.f, 2.f * beta[tidx] * R);
    pycuda::complex<float> module(Deltaprime / (R * R), 0.f);
    Ad[tidx] = module * exp(phase);
  } 
  """)
  
funcAdMod2Dp = SourceModule("""
  #include <pycuda-complex.hpp>
  __global__ void funcAd2Dp(float *Xprime, float *X, float *Yprime, float *Y, float *Zprime, float zzero, int k, pycuda::complex<float> *Ad, float Deltaprime, float *beta, const int N, const int M)
  {
    const int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    const int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    if ((tidx >= N) || (tidy >= M)) return;
    for (int k = tidy; k < M; k = k + blockIdx.y * blockDim.y) {
        float R = sqrt((Xprime[k] - X[tidx]) * (Xprime[k] - X[tidx]) + (Yprime[k] - Y[tidx]) * (Yprime[k] - Y[tidx]) + (Zprime[k] + zzero) * (Zprime[k] + zzero));
        pycuda::complex<float> phase(0.f, 2.f * beta[tidx] * R);
        pycuda::complex<float> module(Deltaprime / (R * R), 0.f);
        Ad[k * N + tidx] = module * exp(phase);
    }
  } 
  """)

funcAdMod2D = SourceModule("""
  #include <pycuda-complex.hpp>
  __global__ void funcAd2D(float *Xprime, float *X, float *Yprime, float *Y, float *Zprime, float zzero, int k, pycuda::complex<float> *Ad, float Deltaprime, float *beta, const int N, const int M)
  {
    const int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    const int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    if ((tidx >= N) || (tidy >= M)) return;
    float R = sqrt((Xprime[tidy] - X[tidx]) * (Xprime[tidy] - X[tidx]) + (Yprime[tidy] - Y[tidx]) * (Yprime[tidy] - Y[tidx]) + (Zprime[tidy] + zzero) * (Zprime[tidy] + zzero));
    pycuda::complex<float> phase(0.f, 2.f * beta[tidx] * R);
    pycuda::complex<float> module(Deltaprime / (R * R), 0.f);
    Ad[tidy * N + tidx] = module * exp(phase);
  } 
  """)

#################################################
# FUNCTION TO COMPUTE RELEVANT HERMITIAN MATRIX #
#################################################
def computeAd(Xprime_gpu, XX_gpu, Yprime_gpu, YY_gpu, Zprime_gpu, zzero, FREQ_gpu, c, Deltaprime, sizePartition):

    funcAdModCall2D = funcAdMod2D.get_function("funcAd2D")
    blockDim  = (BLOCKSIZEX, BLOCKSIZEY, 1)
    gridDim   = (iDivUp(XX_gpu.size, BLOCKSIZEX), iDivUp(Xprime_gpu.size, BLOCKSIZEY), 1)
    #gridDim   = (iDivUp(XX_gpu.size, BLOCKSIZEX), 16384, 1)

    #print(blockDim)
    #print(gridDim)
    
    Ad_gpu              = gpuarray.zeros((Xprime_gpu.size, XX_gpu.size), dtype = np.complex64)
    k_gpu               = (2 * np.pi * FREQ_gpu) / c
    XX_gpu              = gpuarray.reshape(XX_gpu, (XX_gpu.size, 1))
    YY_gpu              = gpuarray.reshape(YY_gpu, (YY_gpu.size, 1))

    funcAdModCall2D(Xprime_gpu, XX_gpu, Yprime_gpu, YY_gpu, Zprime_gpu, np.float32(zzero), np.int32(i), Ad_gpu, np.float32(Deltaprime), k_gpu, np.int32(XX_gpu.size), np.int32(Xprime_gpu.size), block = blockDim, grid = gridDim)

    return Ad_gpu

####################################
# FUNCTION TO COMPUTE A_dagger * y #
####################################
def computeAdy(cublasHandle, y_gpu, Xprime_gpu, XX_gpu, Yprime_gpu, YY_gpu, Zprime_gpu, zzero, FREQ_gpu, c, Deltaprime, sizePartition, sizeOut):

    numPartitions = np.int32(np.ceil(XX_gpu.size / sizePartition))

    x_gpu = gpuarray.zeros((sizeOut, 1), dtype = np.complex64)

    p1 = 0
    p2 = sizePartition
    
    for k in range(numPartitions):
        currentAd_gpu   = computeAd(Xprime_gpu, XX_gpu[p1 : p2], Yprime_gpu, YY_gpu[p1 : p2], Zprime_gpu, zzero, FREQ_gpu[p1 : p2], c, Deltaprime, sizePartition)
        m, n            = currentAd_gpu.shape
        cublas.cublasCgemv(cublasHandle, 't', n, m, np.complex64(1), currentAd_gpu.gpudata, n, y_gpu[p1 : p2].gpudata, 1, np.complex64(1), x_gpu.gpudata, 1)
        #x_gpu       = x_gpu + culinalg.dot(currentAd_gpu, y_gpu[p1 : p2], 'N', 'N', cublasHandle)
        p1          = p2
        if (k == (numPartitions - 2)):
            p2 = XX_gpu.size
        else:
            p2 = p2 + sizePartition

    return x_gpu

#######################################
# FUNCTION TO COMPUTE RELEVANT MATRIX #
#######################################
def computeA(Xprime_gpu, X_gpu, XX_gpu, Yprime_gpu, Y_gpu, YY_gpu, Zprime_gpu, zzero, freq, FREQ_gpu, c, Deltaxprime, Deltayprime, Deltazprime):

    A_gpu        = gpuarray.empty((X_gpu.size * freq.size , Xprime_gpu.size), dtype = np.complex64)

    funcAModCall2D = funcAMod2D.get_function("funcA2D")
    blockDim  = (BLOCKSIZEX, BLOCKSIZEY, 1)
    gridDim   = (iDivUp(Xprime_gpu.size, BLOCKSIZEX), iDivUp(XX_gpu.size, BLOCKSIZEY), 1)

    k_gpu     = (2 * np.pi * FREQ_gpu) / c 

    funcAModCall2D(Xprime_gpu, X_gpu, Yprime_gpu, Y_gpu, Zprime_gpu, np.float32(zzero), np.int32(i), A_gpu, np.float32(Deltaxprime * Deltayprime * Deltazprime), k_gpu, np.int32(Xprime_gpu.size), np.int32(XX_gpu.size), block = blockDim, grid = gridDim)

    return A_gpu

#############################
# FUNCTION TO COMPUTE A * x #
#############################
def computeAx(x_gpu, Xprime_gpu, X_gpu, XX_gpu, Yprime_gpu, Y_gpu, YY_gpu, Zprime_gpu, zzero, freq, FREQ_gpu, c, Deltaxprime, Deltayprime, Deltazprime, sizePartition, sizeOut):

    numPartitions = np.int32(np.ceil(Xprime_gpu.size / sizePartition))

    y_gpu = gpuarray.zeros((sizeOut, 1), dtype = np.complex64)
    p1 = 0
    p2 = sizePartition

    for k in range(numPartitions):

        currentA_gpu = computeA(Xprime_gpu[p1 : p2], X_gpu, XX_gpu, Yprime_gpu[p1 : p2], Y_gpu, YY_gpu, Zprime_gpu[p1 : p2], zzero, freq, FREQ_gpu, c, Deltaxprime, Deltayprime, Deltazprime)
        #y_gpu   = y_gpu + culinalg.dot(currentA_gpu, x_gpu[p1 : p2], 'N', 'N')
        m, n            = currentA_gpu.shape
        cublas.cublasCgemv(cublasHandle, 't', n, m, np.complex64(1), currentA_gpu.gpudata, n, x_gpu[p1 : p2].gpudata, 1, np.complex64(1), y_gpu.gpudata, 1)
        p1      = p2
        if (k == (numPartitions - 2)):
            p2  = Xprime_gpu.size
        else:
            p2  = p2 + sizePartition

    return y_gpu

##################################################################################
# BICONJUGATE GRADIENT STABILIZED ON GPU - SINGLE PRECISION - NO PRECONDITIONING #
##################################################################################
def bicgstabMemory(cublasHandle, x_gpu, b_gpu, Xprime_gpu, X_gpu, XX_gpu, Yprime_gpu, Y_gpu, YY_gpu, Zprime_gpu, zzero, freq, FREQ_gpu, c, Deltaxprime, Deltayprime, Deltazprime, sizePartitionr, sizePartitionc, M, max_it, tol):
    
    # --- flag:     0 = solution found to tolerance
    #               1 = no convergence given max_it
    #              -1 = breakdown: rho = 0
    #              -2 = breakdown: omega = 0

    N       = xcg_gpu.size

    # --- Initializations
    iter    = np.float32(0)
    flag    = np.float32(0)
    alpha   = np.float32(0)
    rho_1   = np.float32(0)
    v_gpu   = gpuarray.zeros(N, dtype = np.float32)
    p_gpu   = gpuarray.zeros(N, dtype = np.float32)
#    d_p_hat = gpuarray.zeros(N, dtype = np.float32)
#    d_s_hat = gpuarray.zeros(N, dtype = np.float32)
#    d_t     = gpuarray.zeros(N, dtype = np.float32)

    #bnrm2   = np.sqrt((culinalg.dot(b_gpu, b_gpu.conj(), 'T', 'N').real).get())
    bnrm2   = cublas.cublasScnrm2(cublasHandle, N, b_gpu.gpudata, 1)
    if  bnrm2 == np.float32(0.0): 
        bnrm2 = np.float32(1.0)

    yprime_gpu  = computeAx(x_gpu, Xprime_gpu, X_gpu, XX_gpu, Yprime_gpu, Y_gpu, YY_gpu, Zprime_gpu, zzero, freq, FREQ_gpu, c, Deltaxprime, Deltayprime, Deltazprime, sizePartitionc, XX_gpu.size)
    xprime_gpu  = computeAdy(cublasHandle, yprime_gpu, Xprime_gpu, XX_gpu, Yprime_gpu, YY_gpu, Zprime_gpu, zzero, FREQ_gpu, c, Deltaxprime * Deltayprime * Deltazprime, sizePartitionr, b_gpu.size)
    r_gpu       = b_gpu - xprime_gpu
    error       = cublas.cublasScnrm2(cublasHandle, N, r_gpu.gpudata, 1) / bnrm2
    if (error < tol): 
        return x_gpu, error, iter, flag

    omega       = np.float32(1.0)
    r_tld_gpu   = r_gpu.copy()

    for iter in range(max_it):

        rho     = cublas.cublasCdotc(cublasHandle, N, r_tld_gpu.gpudata, 1, r_gpu.gpudata, 1)     # direction vector
        if (rho == np.float32(0.0)): 
            break

        if (iter > 0):
            beta = (rho / rho_1) * (alpha / omega)
            cublas.cublasCaxpy(cublasHandle, N, -omega,          v_gpu.gpudata, 1, p_gpu.gpudata, 1)
            cublas.cublasCscal(cublasHandle, N, beta,            p_gpu.gpudata, 1)
            cublas.cublasCaxpy(cublasHandle, N, np.float32(1.0), r_gpu.gpudata, 1, p_gpu.gpudata, 1)
        else:
            p_gpu = r_gpu.copy()

        p_hat_gpu = p_gpu.copy()
        yprime_gpu  = computeAx(p_hat_gpu, Xprime_gpu, X_gpu, XX_gpu, Yprime_gpu, Y_gpu, YY_gpu, Zprime_gpu, zzero, freq, FREQ_gpu, c, Deltaxprime, Deltayprime, Deltazprime, sizePartitionc, XX_gpu.size)
        v_gpu       = computeAdy(cublasHandle, yprime_gpu, Xprime_gpu, XX_gpu, Yprime_gpu, YY_gpu, Zprime_gpu, zzero, FREQ_gpu, c, Deltaxprime * Deltayprime * Deltazprime, sizePartitionr, b_gpu.size)
 
        alpha       = rho / cublas.cublasCdotc(cublasHandle, N, r_tld_gpu.gpudata, 1, v_gpu.gpudata, 1)
        s_gpu       = r_gpu.copy()
        cublas.cublasCaxpy(cublasHandle, N, -alpha, v_gpu.gpudata, 1, s_gpu.gpudata, 1)
        norms       = cublas.cublasScnrm2(cublasHandle, N, s_gpu.gpudata, 1)
        if (norms < tol):                          # --- early convergence check
            cublas.cublasCaxpy(cublasHandle, N, np.float32(alpha), p_hat_gpu.gpudata, 1, x_gpu.gpudata, 1)
            break

        # --- stabilizer
        s_hat_gpu   = s_gpu.copy()
        yprime_gpu  = computeAx(s_hat_gpu, Xprime_gpu, X_gpu, XX_gpu, Yprime_gpu, Y_gpu, YY_gpu, Zprime_gpu, zzero, freq, FREQ_gpu, c, Deltaxprime, Deltayprime, Deltazprime, sizePartitionc, XX_gpu.size)
        t_gpu       = computeAdy(cublasHandle, yprime_gpu, Xprime_gpu, XX_gpu, Yprime_gpu, YY_gpu, Zprime_gpu, zzero, FREQ_gpu, c, Deltaxprime * Deltayprime * Deltazprime, sizePartitionr, b_gpu.size)
        omega       = cublas.cublasCdotc(cublasHandle, N, t_gpu.gpudata, 1, s_gpu.gpudata, 1) / cublas.cublasCdotc(cublasHandle, N, t_gpu.gpudata, 1, t_gpu.gpudata, 1)

        # --- update approximation
        cublas.cublasCaxpy(cublasHandle, N, alpha, p_hat_gpu.gpudata, 1, x_gpu.gpudata, 1)
        cublas.cublasCaxpy(cublasHandle, N, omega, s_hat_gpu.gpudata, 1, x_gpu.gpudata, 1)

        r_gpu     = s_gpu.copy()
        cublas.cublasCaxpy(cublasHandle, N, -omega, t_gpu.gpudata, 1, r_gpu.gpudata, 1)

        error = cublas.cublasScnrm2(cublasHandle, N, r_gpu.gpudata, 1) / bnrm2;                     # --- check convergence
        if (error <= tol):
            break

        if (omega == np.float32(0.0)):
            break
        
        rho_1 = rho

        print("iteration")

    temp  = np.sqrt(gpuarray.max(s_gpu.real * s_gpu.real + s_gpu.imag * s_gpu.imag).get())
    if ((error <= np.float32(tol)) or temp <= tol):                  # --- converged
        if (temp <= tol):
            error = cublas.cublasScnrm2(cublasHandle, N, s_gpu.gpudata, 1) / bnrm2
        flag = 0
    elif (omega == np.float32(0.0)):                # --- breakdown
        flag = -2
    elif (rho == np.float32(0.0)):
        flag = -1
    else:                                           # --- no convergence
        flag = 1

    p_hat_gpu.gpudata.free() 
    s_hat_gpu.gpudata.free() 
    v_gpu.gpudata.free() 
    t_gpu.gpudata.free() 

    return xcg_gpu, 0, 0, 0

########
# MAIN #
########
culinalg.init()
cuda.init()
cublas_handle = cublas.cublasCreate()

start = cuda.Event()
end   = cuda.Event()

# --- Wave propagation
c            = np.float32(3e8)

# --- Frequency definitions
fmin         = np.float32(1e9)
fmax         = np.float32(2.e9)
freq         = np.arange(fmin, fmax + .5e9, .5e9, dtype = np.float32)           

lambdamin    = np.float32(c / fmax)
lambdamax    = np.float32(c / fmin)

# --- Measurement domain
Deltax       = np.float32(lambdamin / 2)
Deltay       = np.float32(lambdamin / 2)

xM           = np.float32(5 * lambdamax)
yM           = np.float32(5 * lambdamax)

x            = np.arange(-xM, xM + Deltax, Deltax, dtype = np.float32) 
y            = np.arange(-yM, yM + Deltay, Deltay, dtype = np.float32)
[Y, X]       = np.meshgrid(y, x)  
X_gpu        = gpuarray.to_gpu(X)
Y_gpu        = gpuarray.to_gpu(Y)

zzero        = np.float32(6.666666666666667 * lambdamax)

# --- Investigation domain
Deltaxprime  =  np.float32(lambdamin / 4)
Deltayprime  =  np.float32(lambdamin / 4)
Deltazprime  =  np.float32(lambdamin / 4)
Deltaprime   =  np.float32(Deltaxprime * Deltayprime * Deltazprime)

xprimeM      =  np.float32(5 * lambdamax)
yprimeM      =  np.float32(5 * lambdamax)
zprimeM      =  np.float32(5 * lambdamax)

xprime       = np.arange(-xprimeM, xprimeM, Deltaxprime, dtype = np.float32)
yprime       = np.arange(-yprimeM, yprimeM, Deltayprime, dtype = np.float32)
zprime       = np.arange(-zprimeM, zprimeM, Deltazprime, dtype = np.float32)

Nxprime      = len(xprime)     
Nyprime      = len(yprime)
Nzprime      = len(zprime)
Nprime       = Nxprime * Nyprime * Nzprime

[Xprime, Zprime, Yprime]      =   np.meshgrid(xprime, zprime, yprime) 
Xprime_gpu   = gpuarray.to_gpu(Xprime)
Yprime_gpu   = gpuarray.to_gpu(Yprime)
Zprime_gpu   = gpuarray.to_gpu(Zprime)
Xprime_gpu   = Xprime_gpu.ravel()
Yprime_gpu   = Yprime_gpu.ravel()
Zprime_gpu   = Zprime_gpu.ravel()

# --- Data generation
Es         = (np.zeros ((X.size * freq.size, 1), dtype = np.complex64))  

for i in range(X.size):
    Rprime =  np.sqrt ((0 - X.item(i))**2 + (0 - Y.item(i))**2 + (0 + zzero)**2)
    for j in range(freq.size):
      k   =  np.float32((2 *np.pi * freq[j]) / c) 
      Es[i + j * X.size,:] = np.reshape((np.exp(-1j * 2 * Rprime * k) / Rprime**2) * Deltaprime, (1, Rprime.size))
Es_gpu          = gpuarray.to_gpu(Es)

#[FREQ, XX]      = np.meshgrid(freq, np.reshape(X, (np.size(X), 1)))
#[FREQ, XX]      = np.meshgrid(freq, np.reshape(X, (np.size(X), 1)))
[XX, FREQ]      = np.meshgrid(np.reshape(X, (np.size(X), 1)), freq)
[YY, FREQ]      = np.meshgrid(np.reshape(Y, (np.size(Y), 1)), freq)
FREQ_gpu        = gpuarray.to_gpu(FREQ)
FREQ_gpu        = FREQ_gpu.ravel()
XX_gpu          = gpuarray.to_gpu(XX)
XX_gpu          = XX_gpu.ravel()
YY_gpu          = gpuarray.to_gpu(YY)
YY_gpu          = YY_gpu.ravel()

sizePartitionr  = 20
sizePartitionc  = 20
#plt.plot(np.abs(Es)) 
#plt.show()
cublasHandle    = cublas.cublasCreate()

start.record()
Esprime_gpu     = computeAdy(cublasHandle, Es_gpu, Xprime_gpu, XX_gpu, Yprime_gpu, YY_gpu, Zprime_gpu, zzero, FREQ_gpu, c, Deltaprime, sizePartitionr, Xprime_gpu.size)
end.record() 
end.synchronize()
secs = start.time_till(end) * 1e-3
print("Processing time = %fs" % (secs))
print("Processing time = %fs" % (timing))

#pycuda.driver.stop_profiler()
""" 
Esprime         = Esprime_gpu.get()
plt.plot(np.abs(Esprime))
plt.show()
"""
maxNumIter      = 1
xcg_gpu         = gpuarray.zeros((Xprime.size, 1), dtype = np.complex64)



xcg_gpu, error, iter, flag = bicgstabMemory(cublasHandle, xcg_gpu, Esprime_gpu, Xprime_gpu, X_gpu, XX_gpu, Yprime_gpu, Y_gpu, YY_gpu, Zprime_gpu, zzero, freq, FREQ_gpu, c, Deltaxprime, Deltayprime, Deltazprime, sizePartitionr, sizePartitionc, 1, maxNumIter, 1e-13)
#BicGStabGPU(cublasHandle, d_A, d_x, d_b, max_it = 10, tol = 0.01)

Pcampo_gpu                          = gpuarray.reshape(xcg_gpu, (Nzprime, Nxprime, Nyprime))
Pcampo_cpu = Pcampo_gpu.get()                 
m, n, k    = Pcampo_cpu.shape
mr = np.int32(np.round((m - 1) / 2 + 1))
nr = np.int32(np.round((n - 1) / 2 + 1))
kr = np.int32(np.round((k - 1) / 2 + 1))
im = (np.abs(Pcampo_cpu[:, :, kr]))              #image 
plt.imshow(im / np.max(im), interpolation = 'bicubic', extent = [-zprimeM / lambdamax, zprimeM / lambdamax, -xprimeM / lambdamax, xprimeM / lambdamax])                       #max(max(im)))=0.0143  
plt.ylabel('x [$\lambda_{max}$]')
plt.xlabel('z [$\lambda_{max}$]')
plt.show()

im = (np.abs(Pcampo_cpu[:, nr, :]))              #image 
plt.imshow(im / np.max(im), interpolation = 'bicubic', extent = [-zprimeM / lambdamax, zprimeM / lambdamax, -yprimeM / lambdamax, yprimeM / lambdamax])                       #max(max(im)))=0.0143  
plt.xlabel('z [$\lambda_{max}$]')
plt.ylabel('y [$\lambda_{max}$]')
plt.show()

im=(np.abs(Pcampo_cpu[mr, :, :]))              #image 
plt.imshow(im / np.max(im), interpolation = 'bicubic', extent = [-xprimeM / lambdamax, xprimeM / lambdamax, -yprimeM / lambdamax, yprimeM / lambdamax])                       #max(max(im)))=0.0143  
plt.xlabel('x [$\lambda_{max}$]')
plt.ylabel('y [$\lambda_{max}$]')
plt.show()   
