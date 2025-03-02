# 384 x 384 section

from pathlib import Path, PureWindowsPath

from Localization.twoD_deblurr_image import problem, convolution, functions, maj_min_TV_deblurring

# problem configuration
conf_nr = '7'

# parameters
par = {
    'image_file' : PureWindowsPath( r'twoD_deblurr_image\cameraman_512.tif' ),
    'max_int' : 1,
    'section': ((1*64,7*64), (1*64,7*64)),
    'down_sam' : 1,
    'seed' : 0,
    'noise_spec' : 'std', #'BSNR', 
    'noise_val' : 1e-2, # 40,
    
    'blur' : 'Gaussian',
    'blur_std' : 8, # 4 in FISTA paper 
    'radius' : 8, # 4 in FISTA paper
    'ext_mode_blur' : 'zero', # reflect: cba|abcd|dcb  mirror: dcb|abcd|cba

    'problem_dir' : PureWindowsPath( r'twoD_deblurr_image\Problem_data', 'conf'+conf_nr ),

    'delta' : 35.80 
}
Path.mkdir(Path(par['problem_dir']), parents=True, exist_ok=True)
functions.save(par['problem_dir'] / 'par', par)

# create problem
problem.main(par)

# MAP (eps=0)
x_im, y_im_true, y_im, lam, N, d = functions.load(par['problem_dir'] / 'problem')
conv_op = lambda x: convolution.conv(x, x_im.shape, par['blur'], par['radius'], std=par['blur_std'], BC=par['ext_mode_blur'])
conv_op_T = lambda y: convolution.conv_T(y, x_im.shape, par['blur'], par['radius'], std=par['blur_std'], BC=par['ext_mode_blur'])
MAP_TV = maj_min_TV_deblurring.debl(functions.rav(y_im), conv_op, conv_op_T, par['ext_mode_blur'], 1/lam, mode='TV', max_int=par['max_int'], delta=par['delta'], it_MM_max=1_000_000, rel_diff_x_max=1e-5, it_CG_max=200, eps=0)
functions.save(par['problem_dir'] / 'map_TV', MAP_TV)