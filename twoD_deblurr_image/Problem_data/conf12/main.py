# 512 x 512 section

from pathlib import Path, PureWindowsPath

from Localization.twoD_deblurr_image import problem, convolution, functions, maj_min_TV_deblurring

print(functions.load(r'C:\Users\flock\Python\Localization\twoD_deblurr_image\Problem_data\conf12\delta_ad_TV'))
delta_TV = ((r'C:\Users\flock\Python\Localization\twoD_deblurr_image\Problem_data\conf12\delta_ad_TV'))
# problem configuration
conf_nr = '12'

# parameters
par = {
    'image_file' : PureWindowsPath( r'twoD_deblurr_image\cameraman_512.tif' ),
    'max_int' : 1,
    'section': ((0*64,8*64), (0*64,8*64)),
    'down_sam' : 1,
    'seed' : 0,
    'noise_spec' : 'std', #'BSNR', 
    'noise_val' : 1e-2, # 40,
    
    'blur' : 'motion',
    'blur_std' : None, 
    'radius' : 8, # 4 in FISTA paper
    'theta' : 45,
    'ext_mode_blur' : 'zero', # reflect: cba|abcd|dcb  mirror: dcb|abcd|cba

    'problem_dir' : PureWindowsPath( r'twoD_deblurr_image\Problem_data', 'conf'+conf_nr ),

    'delta' : 34.83
}
Path.mkdir(Path(par['problem_dir']), parents=True, exist_ok=True)
functions.save(par['problem_dir'] / 'par', par)

# create problem
problem.main(par)

# adaptive TV to find delta and the MAP (eps=0)
x_im, y_im_true, y_im, lam, N, d = functions.load(par['problem_dir'] / 'problem')
kernel = functions.motion_kernel(par['radius'], par['theta'])
conv_op = lambda x: convolution.conv(x, x_im.shape, par['blur'], par['radius'], std=par['blur_std'], kernel=kernel, BC=par['ext_mode_blur'])
conv_op_T = lambda y: convolution.conv_T(y, x_im.shape, par['blur'], par['radius'], std=par['blur_std'], kernel=kernel, BC=par['ext_mode_blur'])
delta_TV, x_MAP_TV = maj_min_TV_deblurring.adaptive_debl(functions.rav(y_im), conv_op, conv_op_T, par['ext_mode_blur'], 1/lam, mode='TV', max_int=par['max_int'], beta=1, alpha=1, theta='default')
functions.save(par['problem_dir'] / 'map_ad_TV', x_MAP_TV)
functions.save(par['problem_dir'] / 'delta_ad_TV', delta_TV)