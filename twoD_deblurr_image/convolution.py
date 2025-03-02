import numpy as np
import sys
from scipy import ndimage 
from math import tan, pi

# ravel matrix by stacking columns
rav = lambda x: np.ravel(x, order='F')

# reshape vector to matrix column-wise
res = lambda x, shape: np.reshape(x, shape, order='F')

def motion_kernel(radius, theta):
    kernel = np.zeros((2*radius+1, 2*radius+1))
    x_ind = np.arange(-radius, radius+1)
    y_ind = np.int32( np.rint(tan(theta/180*pi) * x_ind) )
    kernel[np.flip(y_ind+radius), x_ind+radius] = 1
    kernel = kernel/(2*radius+1)
    return kernel

def conv(x, shape, type, radius=None, std=None, kernel=None, BC='zero'):
# convolves images which are given as column stacked vectors

    if BC=='zero': BC_ = 'constant'
    else: sys.exit('BC not defined!')

    if type == 'Gaussian':
        y = rav( ndimage.gaussian_filter( res(x, shape), sigma=std, mode=BC_, radius=radius) )

    elif type == 'motion':
        y = rav( ndimage.convolve( res(x, shape), kernel, mode=BC_))

    else: sys.exit('Type not defined!')
        
    return y


def conv_T(y, shape, type, radius=None, std=None, kernel=None, BC='zero'):
# adjoint of convolution of images which are given as column stacked vectors

    if BC=='zero': BC_ = 'constant'
    else: sys.exit('BC not defined!')

    if type == 'Gaussian':
        x = conv(y, shape, type, radius, std=std, BC=BC)
    
    elif type == 'motion':
        x = conv(y, shape, type, kernel=kernel, BC=BC)

    else: sys.exit('Type not defined!')
        
    return x