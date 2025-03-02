# Generates the problem for image deblurring. I.e., sets up convolution operator and data. 
# The rate parameter, i.e., prior information is not defined.

import numpy as np
import imageio
from pathlib import Path

from Localization.twoD_deblurr_image import functions, convolution

def main(par):

    # original image
    x_im = np.asarray(imageio.imread( Path( par['image_file'] )))
    if x_im.ndim == 3: x_im = x_im[:,:,0]

    # section
    x_im = x_im[par['section'][0][0]: par['section'][0][1], par['section'][1][0]: par['section'][1][1]]

    # downsample
    x_im = x_im[:, ::par['down_sam']]
    x_im = x_im[::par['down_sam'], :]
    N = x_im.shape[0]
    d = functions.rav(x_im).size

    # normalize
    if par['max_int'] == 1: x_im = x_im/255
    elif par['max_int'] == 255: pass

    # convolution
    if par['blur'] == 'motion': kernel = functions.motion_kernel(par['radius'], par['theta'])
    else: kernel = None
    y_im_true = functions.res(convolution.conv(functions.rav(x_im), x_im.shape, par['blur'], par['radius'], std=par['blur_std'], kernel=kernel, BC=par['ext_mode_blur']), x_im.shape)
    
    # noise
    y_im, std_noise, noise_level = functions.apply_noise(y_im_true, par['seed'], noise_spec=par['noise_spec'], noise_val=par['noise_val'])
    lam = 1/std_noise**2

    functions.save( Path( par['problem_dir'] / 'problem' ), [x_im, y_im_true, y_im, lam, N, d] )

