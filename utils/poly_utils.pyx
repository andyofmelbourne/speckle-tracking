

import numpy as np
cimport numpy as np

from libc.math cimport sin, cos, acos, exp, sqrt, fabs, M_PI, pow

FLOAT = np.float
INT   = np.int

ctypedef np.float_t FLOAT_t
ctypedef np.int_t INT_t

cimport cython

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def polymul2d(np.ndarray[FLOAT_t, ndim=2] a, np.ndarray[FLOAT_t, ndim=2] b, np.ndarray[FLOAT_t, ndim=2] c):
    """
    """
    cdef int n, m, l, l2, k, k2

    for n in range(c.shape[0]):
        for m in range(c.shape[1]):
            c[n, m] = 0.
            for l in range(max(n-a.shape[0]+1, 0), min(n+1, b.shape[0])):
                for l2 in range(max(m-a.shape[1]+1, 0), min(m+1, b.shape[1])):
                    k  = n - l
                    k2 = m - l2
                    c[n, m] += a[k, k2] * b[l, l2]
    return c

def polyint2d(np.ndarray[FLOAT_t, ndim=2] a, np.ndarray[FLOAT_t, ndim=1] b, float xmin, float xmax, float ymin, float ymax):
    cdef int l, k
    cdef float out = 0.
    for l in range(a.shape[1]):
        b[l] = 0.
        for k in range(a.shape[0]):
            b[l] += (xmax**(k+1) - xmin**(k+1))/float(k+1) * a[k, l]
    
    for l in range(a.shape[1]):
        out += (ymax**(l+1) - ymin**(l+1))/float(l+1) * b[l]

    return out

def polyder2d(np.ndarray[FLOAT_t, ndim=2] a, np.ndarray[FLOAT_t, ndim=2] a_out, int axis):
    cdef int i, j
    if axis == 0 :
        for i in range(a.shape[0]-1):
            for j in range(a.shape[1]):
                a_out[i, j] = (i+1) * a[i+1,j]

        for j in range(a.shape[1]):
            a_out[a.shape[0]-1, j] = 0
    
    if axis == 1 :
        for i in range(a.shape[0]):
            for j in range(a.shape[1]-1):
                a_out[i, j] = (j+1) * a[i,j+1]

        for i in range(a.shape[0]):
            a_out[i, a.shape[1]-1] = 0
    return a_out
