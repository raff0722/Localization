# sampling via local MALA and eps=1e-3

import numpy as np
from pathlib import Path, PurePath, PureWindowsPath

from Localization.twoD_deblurr_image import functions, convolution, maj_min_TV_deblurring
from Localization import eval_samples

# problem and sampling configuration
conf_nr = '8'
sam_nr = 'lM_eps3'

# load configuration parameters
par = functions.load( PureWindowsPath(r'twoD_deblurr_image\Problem_data', 'conf'+conf_nr, 'par' ) )

# sampling parameters
sam = {
    'acc_int_disp' : 1000,

    'n_ch' : 5, # number of chains
    'n_proc' : 16, # number of processes; required CPUs = number of processes + 1

    'N_po' : 2_000,
    'N_save' : 200,
    'th' : 200,
    'N_b' : 31_250,
    'wG' : 1, # within-Gibbs iterations   

    'h0' : 7.441456582772584e-06,
    'M' : 10,
    'adapt_step' : True,
    'tar_acc' : .547,

    'q' : 64,

    'eps' : 1e-3,

    'seed' : 0,

    'sampling_dir' : PurePath( par['problem_dir'], 'sam_'+sam_nr )
}
sam['x0'] = functions.load(PureWindowsPath(r'twoD_deblurr_image\Problem_data', 'conf'+conf_nr, 'map_ad_TV'))
functions.save(PurePath(sam['sampling_dir'], 'par'), sam)

# max number of parallel updates
x_im, y_im_true, y_im, lam, N, d = functions.load( Path( par['problem_dir'] / 'problem' ) )
print(f'Update max {np.ceil((x_im.shape[0]/(2*sam["q"]))**2)} blocks at once.')

# MAP (eps \neq 0)
conv_op = lambda x: convolution.conv(x, x_im.shape, par['blur'], par['radius'], std=par['blur_std'], BC=par['ext_mode_blur'])
conv_op_T = lambda y: convolution.conv_T(y, x_im.shape, par['blur'], par['radius'], std=par['blur_std'], BC=par['ext_mode_blur'])
map_eps = maj_min_TV_deblurring.debl(np.ravel(y_im, order='F'), conv_op, conv_op_T, par['ext_mode_blur'], 1/lam, 'TV', 1, par['delta'], it_MM_max=1_000_000, rel_diff_x_max=1e-8, it_CG_max=200, CG_rtol=1e-5, eps=sam['eps'])
functions.save(PurePath(sam['sampling_dir'], 'map_eps'), map_eps)            

# # eval samples -- run after sampling by calling "main_loc_MALA_parallel_TV.py" 
# print('Evaluate samples...')
# flags = {'mean':1, 'HDI':0, 'CI':1, 'ESS':1, 'rhat':1}
# options = {'CI':0.90}
# eval_samples.main(PurePath(r'/work3/raff/Localization', sam['sampling_dir']), sam['n_ch'], sam['N_po']//sam['N_save'], flags, options, remove_samples=False)  