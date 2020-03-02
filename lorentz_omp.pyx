
cimport cython
import numpy as np


cimport openmp
from cython.parallel cimport prange
from cython.parallel cimport parallel


@cython.boundscheck(False)

@cython.wraparound(False)

cpdef parallel_2ddot(const double[::1] A, const double[::1] B, 
                const double[::1] C, const double[::1] D , 
                 int chunksize, int schedule ):
    
    cdef int n = A.shape[0]
    cdef int i
    cdef double R[2] 
    if schedule == 1:
        with nogil:
            for i in prange(n, schedule='static', chunksize=chunksize):
                R[0] += A[i] * B[i]
                R[1] += C[i] * D[i]
    elif schedule == 2:
        with nogil :
            for i in prange(n, schedule='dynamic', chunksize=chunksize):
                R[0] += A[i] * B[i]
                R[1] += C[i] * D[i]
    else:
        with nogil :
            for i in prange(n):
                R[0] += A[i] * B[i]
                R[1] += C[i] * D[i]
                
    return R





cpdef lorentz_resolution_omp( R, sigma, b, x_0, y_0, z_0, N, T_c, step, cython.int chunksize=32, cython.int schedule=0):
    '''
    Inputs :  R, sigma, b the paramaters of the lorentz system 
              x0, y0, z0 the initial conditions
              N order of the taylor series development
              step the step size between two stages
    Outputs : vectors x,y,z containing all the trajectory of the particle
    
    '''
    # initialisation :  create empty array
    n_steps = int(T_c / step )+1
    x = np.zeros(n_steps + 1 )
    y = np.zeros(n_steps + 1)
    z = np.zeros(n_steps + 1)
    
    x[0] = x_0
    y[0] = y_0
    z[0] = z_0
    
    tau = np.ascontiguousarray( step**np.arange(1,N+1), dtype = np.float64 )


    
    for t in range(1, (n_steps +1 )) : #loop over time
        
        #initialisation : store alpha1, beta1, gamma1
        
        alpha = np.ascontiguousarray(np.zeros(N + 1) , dtype=np.float64)
        beta = np.ascontiguousarray(np.zeros(N + 1) , dtype=np.float64)
        gamma = np.ascontiguousarray(np.zeros(N + 1) , dtype=np.float64)
        
        alpha[0] = x[t-1]
        beta[0] = y[t-1]
        gamma[0] = z[t-1]
        
        alpha[1] = sigma * (-x[t-1] + y[t-1])
        beta[1] = R * x[t-1] - y[t-1] -x[t-1] * z[t-1]
        gamma[1] = x[t-1] * y[t-1] - b * z[t-1]
        
        for i in range(1, N ) : #loop over derivatives, [::-1] to invert the order of the elements of the vector
            
            
            s =  parallel_2ddot( np.ascontiguousarray(alpha[:(i+1)][::-1], dtype=np.float64) 
                                    , np.ascontiguousarray(gamma[:(i+1)], dtype = np.float64)
                                    , np.ascontiguousarray(alpha[:(i+1)][::-1], dtype=np.float64)
                                    , np.ascontiguousarray(beta[ :(i+1)] , dtype=np.float64)
                                    , chunksize, schedule) 
            
            div = 1/(i+1)
            alpha[i+1] = div * sigma * ( beta[i]  - alpha[i] )  
            beta[i+1] = div * ( R * alpha[i] - beta[i] - s[0] )     
            gamma[i+1] = div * ( s[1] - b * gamma[i]  )
        
        x[t] = x[t-1] + np.sum( alpha[1:] * tau )
        y[t] = y[t-1] + np.sum( beta[1:] * tau )
        z[t] = z[t-1] + np.sum( gamma[1:] * tau )
        
    return(x,y,z)
