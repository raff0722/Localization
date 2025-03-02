from scipy import ndimage, sparse
import numpy as np
import sys
import pickle
from pathlib import Path
from math import tan, pi

def load(file):
    file = Path(file)
    file = open(file, 'rb')
    variable = pickle.load(file)
    file.close()
    return variable

def save(file, variable):
    file = Path(file)
    file = open(file, 'wb')
    pickle.dump(variable, file)
    file.close()

def motion_kernel(radius, theta):
    kernel = np.zeros((2*radius+1, 2*radius+1))
    x_ind = np.arange(-radius, radius+1)
    y_ind = np.int32( np.rint(tan(theta/180*pi) * x_ind) )
    kernel[np.flip(y_ind+radius), x_ind+radius] = 1
    kernel = kernel/(2*radius+1)
    return kernel

# ravel matrix by stacking columns
rav = lambda x: np.ravel(x, order='F')

# reshape vector to matrix column-wise
res = lambda x, shape: np.reshape(x, shape, order='F')

# vertical 2D finite differences operator, 
# image given as column-stacked vector, differences given in column-stacked order
def Dv_op(x, M, N, BC):
# M: rows
# N: cols
    # if BC == 'reflect' or BC == 'nearest' : 
    if BC == 'zero':
        y = rav(np.diff( res(x, (M,N)), axis=0, append=0 ))
    else: sys.exit('BC not implemented!')
    return y

# horizontal 2D finite differences operator, 
# image given as column-stacked vector, differences given in column-stacked order
def Dh_op(x, M, N, BC):
# M: rows
# N: cols
    # if BC == 'reflect' or BC == 'nearest' : 
    if BC == 'zero':
        y = rav(np.diff( res(x, (M,N)), axis=1, append=0 ))
    else: sys.exit('BC not implemented!')
    return y

# adjoint vertical 2D finite differences operator, 
# image given as column-stacked vector, differences given in column-stacked order
def DvT_op(y, M, N, BC):
# M: rows
# N: cols
    # if BC == 'reflect' or BC == 'nearest' : 
    if BC == 'zero':
        x = -rav(np.diff( res(y, (M,N)), axis=0, prepend=0 ))
    else: sys.exit('BC not implemented!')
    return x

# adjoint horizontal 2D finite differences operator, 
# image given as column-stacked vector, differences given in column-stacked order
def DhT_op(y, M, N, BC):
# M: rows
# N: cols
    # if BC == 'reflect' or BC == 'nearest' : 
    if BC == 'zero':
        x = -rav(np.diff( res(y, (M,N)), axis=1, prepend=0 ))
    else: sys.exit('BC not implemented!')
    return x

def apply_noise(true_data, seed, noise_spec, noise_val):

    np.random.seed(seed)

    # if noise_sepc == 'level':
        
    #     e = np.random.randn(true_data.size)
    #     e = e/np.norm(e) * np.norm(true_data.flatten()) * noise[1]
    #     e = np.reshape(e, true_data.shape)
    #     std = noise[1] * np.linalg.norm(true_data) * 1/np.sqrt( true_data.size )

    #     return true_data + e, std
    
    if noise_spec == 'BSNR':
        std = np.sqrt( np.var( true_data, ddof=1 ) / 10**(noise_val/10) )

    elif noise_spec == 'std':
        std = noise_val

    e = np.random.normal(loc=0, scale=std, size=true_data.shape)
    noise_level = std / np.linalg.norm(true_data) * np.sqrt( true_data.size )

    return true_data + e, std, noise_level

def PSNR(I, K, maxI):
    # I: original image, K: approximation
    MSE = 1/I.size**2 * np.linalg.norm( I - K )**2
    return 10 * np.log10( maxI**2 / MSE )

# def constr_R_mat()
    
#     except:
#         R_im = aux.constr_blurr_op(blur=par['blur'], sigma=None, mode=par['ext_mode_blur'], radius=par['radius'])
#         R_vec = lambda x: f.rav( R_im( f.res( x, N ) ))
#         R_mat = matrixTools.matrix_from_linOp(R_vec, d, d, sparse_flag=True)
#         pickle_routines.save(par['problem_dir'] + '\R_mat', R_mat)


    # print('autocorrelation')
    # if cluster:
    #     logpdf_ac.main(par, sam, sam['save_dir_cluster'])
    # else:
    #     logpdf_ac.main(par, sam, sam['save_dir'])