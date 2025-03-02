# Description
# Gives first-order finite differences matrices with specified boundary conditions. 
# The boundary conditions are w.r.t. to the signal, e.g., reflect, and defined according to scipy.ndimage boundary conditions.
# Except 'constant' which has default value zero in scipy. This BC is obtained by BC==zero in this script.
# Gives differences x_i+1 - x_i where i goes from left to right and top to bottom

import numpy as np
from scipy import sparse as sps
import sys

# ravel matrix by stacking columns
rav = lambda x: np.ravel(x, order='F')

# reshape vector to matrix column-wise
res = lambda x, shape: np.reshape(x, shape, order='F')

# 2D finite differences matrices, image given as column-stacked vector, differences given in column-stacked order
def DvDh_mat(M, N, BC):
# M: rows
# N: cols
    
    if BC == 'reflect' or BC == 'nearest' : 
        # Dv, vertical differences
        d = np.vstack( ( np.append(-1*np.ones(M-1), 0), np.ones(M) ) )
        Dv_1D = sps.dia_array( (d, (0, 1)), shape=(M, M) )

        # Dh, horizontal differences
        d = np.vstack( ( np.append(-1*np.ones(N-1), 0), np.ones(N) ) )
        Dh_1D = sps.dia_array( (d, (0, 1)), shape=(N, N) )

    elif BC == 'zero':
        # Dv, vertical differences
        d = np.vstack( (-1*np.ones(M), np.ones(M)) )
        Dv_1D = sps.dia_array( (d, (0, 1)), shape=(M, M) )

        # Dh, horizontal differences
        d = np.vstack( (-1*np.ones(N), np.ones(N)) )
        Dh_1D = sps.dia_array( (d, (0, 1)), shape=(N, N) )

    else: sys.exit('BC not implemented!')

    I = sps.eye_array(N, N)
    Dv = sps.csr_array( sps.kron(I, Dv_1D) ) 
    I = sps.eye_array(M, M)
    Dh = sps.csr_array( sps.kron(Dh_1D, I) ) 
            
    return Dv, Dh

# 2D finite differences operators, image given as column-stacked vector, differences given in column-stacked order

def Dv_op(x, M, N, BC):
# M: rows
# N: cols
    # if BC == 'reflect' or BC == 'nearest' : 
    if BC == 'zero':
        y =rav(np.diff(res(x, (M,N)), axis=0, append=0 ))
    else: sys.exit('BC not implemented!')
    return y

def Dh_op(x, M, N, BC):
# M: rows
# N: cols
    # if BC == 'reflect' or BC == 'nearest' : 
    if BC == 'zero':
        y =rav(np.diff(res(x, (M,N)), axis=1, append=0 ))
    else: sys.exit('BC not implemented!')
    return y

def DvT_op(y, M, N, BC): # adjoint
# M: rows
# N: cols
    # if BC == 'reflect' or BC == 'nearest' : 
    if BC == 'zero':
        x = -rav(np.diff(res(y, (M,N)), axis=0, prepend=0 ))
    else: sys.exit('BC not implemented!')
    return x

def DhT_op(y, M, N, BC): # adjoint
# M: rows
# N: cols
    # if BC == 'reflect' or BC == 'nearest' : 
    if BC == 'zero':
        x = -rav(np.diff(res(y, (M,N)), axis=1, prepend=0 ))
    else: sys.exit('BC not implemented!')
    return x