# sampling via "full" (global) MALA with smoothed TV prior

# adaptive step according to 
# Marshall, Tristan, and Gareth Roberts. 
# “An Adaptive Approach to Langevin MCMC.” 
# Statistics and Computing 22, no. 5 (2012): 1041–57. https://doi.org/10.1007/s11222-011-9276-6.

import numpy as np
from time import time
from scipy.optimize import check_grad
from pathlib import Path

# import warnings
# warnings.filterwarnings("error")

from Localization.twoD_deblurr_image import functions, convolution, finite_differences

def main(par, sam, ch_nr):

    # problem data
    [x_im, y_im_true, y_im, lam, N, d] = functions.load( Path( par['problem_dir'] / 'problem' ) )
    shape = x_im.shape
    d = x_im.size

    # convolution (and adjoint) for images represented as column-stacked vectors
    if par['blur'] == 'motion': kernel = functions.motion_kernel(par['radius'], par['theta'])
    else: kernel = None
    conv = lambda x: convolution.conv(x, shape, type=par['blur'], radius=par['radius'], std=par['blur_std'], kernel=kernel, BC=par['ext_mode_blur'])
    conv_T = lambda x: convolution.conv_T(x, shape, type=par['blur'], radius=par['radius'], std=par['blur_std'], kernel=kernel, BC=par['ext_mode_blur'])

    # finite difference operators and adjoints
    Dh = lambda x: finite_differences.Dh_op(x, M=shape[0], N=shape[1], BC=par['ext_mode_blur'])
    Dv = lambda x: finite_differences.Dv_op(x, M=shape[0], N=shape[1], BC=par['ext_mode_blur'])
    DhT = lambda x: finite_differences.DhT_op(x, M=shape[0], N=shape[1], BC=par['ext_mode_blur'])
    DvT = lambda x: finite_differences.DvT_op(x, M=shape[0], N=shape[1], BC=par['ext_mode_blur'])

    # twice an approximation of the square-root function
    psi = lambda x: 2*np.sqrt( x + sam['eps'] )
    d_psi = lambda x: 1/np.sqrt( x + sam['eps'] )

    # adaptive step
    if sam['adapt_step'] and sam['N_b']>0:
        dh = 0.001*sam['h0']
        b = 1
        r = - np.log( dh/b ) / ( np.log(sam['N_b']) ) 
        c = lambda n: b * n**(-r)
    else: c=None
    
    # log like and its gradient
    y = functions.rav(y_im)
    def loglike_grad(x):
        ymAx = y - conv(x)
        ll = -lam/2 * np.linalg.norm( ymAx )**2 
        gl = lam * conv_T(ymAx)
        return ll, gl

    # log prior and its gradient    
    def logprior_grad(x):
        Dvx = Dv(x)
        Dvx2 = Dvx**2
        Dhx = Dh(x)
        Dhx2 = Dhx**2
        Dvx2_p_Dhx2 = Dvx2 + Dhx2
        lp = - par['delta']/2 * np.sum(psi( Dvx2_p_Dhx2 ))
        d_psi_Dvx2_p_Dhx2 = d_psi(Dvx2_p_Dhx2)
        glp = - par['delta'] * ( DvT(d_psi_Dvx2_p_Dhx2 * Dvx) + DhT(d_psi_Dvx2_p_Dhx2 * Dhx) )
        return lp, glp

    def logpdf_grad(x):
        lp, glp = logprior_grad(x)
        ll, gl = loglike_grad(x)
        return ll+lp, glp+gl

    # # check gradient
    # fun = lambda x: logpdf_grad(x)[0]
    # grad = lambda x: logpdf_grad(x)[1]
    # err = check_grad(fun, grad, sam['x0'])
    # rel_err = err/ np.linalg.norm( grad(sam['x0']) )
    # print(f'gradient check: {rel_err}')

    # x0
    np.random.seed(ch_nr)
    x0 = sam['x0'] * np.random.uniform(0.5, 1.5, sam['x0'].size)
    
    # save directory
    save_dir = sam['sampling_dir'] / ('ch'+str(ch_nr))
    Path.mkdir(Path(save_dir), parents=True, exist_ok=True)

    # sample array and storage
    n_parts = sam['N_po']//sam['N_save']
    sam_per_part = [sam['N_save']] * n_parts
    if sam['N_po']%sam['N_save']:
        sam_per_part.append(sam['N_po']%sam['N_save'])
        n_parts +=1 
    sam_per_part.append(0) # add dummy to avoid 'list index out of range' 

    # accepted samples
    acc = np.zeros(sam['N_po'] * sam['th'] + sam['N_b'] + 1, dtype=bool)
    acc[0] = True # initial state is accepted

    # adaptive step size
    h = sam['h0']
    M = 10
    tar_acc = 0.574

    # initialize state 
    x = np.zeros((d, sam_per_part[0]))
    x_state = x0
    log_tar_state, log_tar_state_grad = logpdf_grad(x0)
    mean_state = x_state + h * log_tar_state_grad
    
    # sampling
    kk = 0 # counter for saving to physical storage
    jj = 0 # counter for N samples
    t0 = time()
    for ii in range(1, sam['N_po'] * sam['th'] + sam['N_b'] + 1):

        if ii == sam['N_b']+1: 
            sam['adapt_step'] = False
            t1 = time()

        # monitoring of sampling
        if ii%sam['acc_int_disp'] == 0:
            sum_acc_int = np.sum( acc[ ii-sam['acc_int_disp']: ii ])
            t_int = (time() - t0)/60
            t_proj = t_int/ii * (sam['N_po'] * sam['th'] + sam['N_b'])
            print('sample {}/{} -- acc {:>4}/{} -- step size {:.2e} -- time {:.2f}/{:.2f} minutes'.format(ii, sam['N_po']*sam['th']+sam['N_b'], sum_acc_int, sam['acc_int_disp'], h, t_int, t_proj), end='\r')

        # propose
        mean_state = x_state + h * log_tar_state_grad
        x_prop = mean_state + np.sqrt(2*h) * np.random.normal(size=d)
        log_tar_prop, log_tar_prop_grad = logpdf_grad(x_prop)
        mean_prop = x_prop + h * log_tar_prop_grad
        q_forw = -1/(4*h) *np.linalg.norm( x_prop - mean_state )**2
        q_back = -1/(4*h) *np.linalg.norm( x_state - mean_prop )**2
    
        # acceptance prob
        log_alpha = min( 0, (log_tar_prop + q_back) - (log_tar_state + q_forw) )

        # accept/reject
        if log_alpha > np.log(np.random.random()): # accept
            x_state = x_prop
            log_tar_state = log_tar_prop
            log_tar_state_grad = log_tar_prop_grad
            mean_state = mean_prop
            acc[ii] = True
        else: # reject
            pass

        # adapt step size
        if sam['adapt_step'] and ii <= sam['N_b'] and ii >= M:
            h_star = min( 0.001*h, c(ii) )
            acc_rate_M = np.sum( acc[(ii - M + 1) : (ii + 1)] ) / M
            if acc_rate_M < tar_acc:
                h -= h_star
            else: # acc_rate_M >= tar_acc:
                h += h_star

        # save sample
        if ii > sam['N_b'] and (ii-sam['N_b'])%sam['th']==0:
            x[:, jj] = x_state
            jj += 1

            # new samples array if needed
            if jj == sam_per_part[kk]:    
                np.save( Path( save_dir / ('p_' + str(kk)) ), x, allow_pickle=False )
                out = {'h':h, 'rand_state':np.random.get_state()}
                functions.save( Path( save_dir / ('out_'+ str(kk)) ), out)
                kk += 1
                x = np.zeros((d, sam_per_part[kk]))
                jj = 0
    
    out = {'acc': acc[sam['N_b']:], 'time':[t1-t0,time()-t1] , 'h':h}
    functions.save( Path( save_dir / 'out' ), out)
        
    print('\nSampling finished')