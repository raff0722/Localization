#%% description

# Local MALA with parallel updating of the blocks using shared memory.
# Samples one chain. 
# The prior is TV.
# Uses linear operators.

# The step adaptation is for each block and according to 
# Marshall, Tristan, and Gareth Roberts. 
# “An Adaptive Approach to Langevin MCMC.” 
# Statistics and Computing 22, no. 5 (2012): 1041–57. https://doi.org/10.1007/s11222-011-9276-6.

#%% Python modules

from multiprocessing import RawArray, Pool
from time import time
import sys
import numpy as np
import os
from pathlib import Path, PurePosixPath, PureWindowsPath, PurePath
from scipy.optimize import check_grad
from scipy import ndimage

import warnings
warnings.filterwarnings("error")

from Localization.twoD_deblurr_image import aux_loc_MALA_TV, functions, convolution, finite_differences

#%% global variables

def initializer(in1, in2, in3, in_shared, seed):

    global s, p, lam, kernel
    global y_hat, n_blocks, b_ad, r_ad, N_total
    global I, I_hathat, shape_hathat, I_hathat__hat, I_hathat__, I_hathat__c, I_hat, I_til, shape_til, I_til__, I_til__c
    global X_state_raw, h_raw, acc_hist_raw
    
    s, p, lam, kernel = in1
    y_hat, n_blocks, b_ad, r_ad, N_total = in2
    I, I_hathat, shape_hathat, I_hathat__hat, I_hathat__, I_hathat__c, I_hat, I_til, shape_til, I_til__, I_til__c = in3
    X_state_raw, h_raw, acc_hist_raw = in_shared

    np.random.seed(seed*os.getpid())

#%% local update
    
def loc_upd(task):

    block, ii = task
    
    # get the shared arrays
    X_state = np.frombuffer( X_state_raw, dtype=np.double )
    h_state = np.frombuffer( h_raw, dtype=np.double )
    acc_hist = np.frombuffer( acc_hist_raw, dtype=np.bool_ ).reshape((n_blocks, (N_total+1)))

    # block index and tag
    i_b, tag = block

    # step size
    h = h_state[i_b]

    # crop from state (capital letter means complete image)
    x_state = X_state[I[i_b]]
    x_hathat = X_state[I_hathat[i_b]] # for convolution
    x_til = X_state[I_til[i_b]] # for TV
    
    # get modified data for likelihood
    y_mod = y_hat[i_b] - sel(conv(proj(x_hathat, I_hathat__c[tag]), shape_hathat[tag]), I_hathat__hat[tag])

    # constants in finite differences
    bv = Dv(proj(x_til, I_til__c[tag]), shape_til[tag])
    bh = Dh(proj(x_til, I_til__c[tag]), shape_til[tag])
    
    # log pdf and gradient for current within-Gibbs iteratäions
    logpdf_grad_it = lambda x: logpdf_grad(x, y_mod, bv, bh, tag)

    # # check gradient
    # fun = lambda z: logpdf_grad_it(z)[0]
    # grad = lambda z: logpdf_grad_it(z)[1]
    # abs_err = check_grad(fun, grad, x_state)
    # print(f'rel error: {abs_err/np.linalg.norm(grad(x_state))}')

    # init state
    lpdf_state, grad_state = logpdf_grad_it(x_state)
    mean_state = x_state + h * grad_state
    
    # within-Gibbs iterations
    for _ in range(s['wG']):

        # generate proposal
        x_prop = mean_state + np.sqrt(2*h) * np.random.randn(s['q']**2)
        
        # evaluate forward and backward äproposal density
        lpdf_prop, grad_prop = logpdf_grad_it(x_prop)
        mean_prop = x_prop + h * grad_prop
        q_forw = -1/(4*h) * np.linalg.norm( x_prop - mean_state )**2
        q_back = -1/(4*h) * np.linalg.norm( x_state - mean_prop )**2
        
        # acceptance prob
        log_alpha = min( 0, lpdf_prop + q_back - (lpdf_state + q_forw) )
        
        # accept/reject
        if log_alpha > np.log(np.random.random()): # accept
            x_state = x_prop
            mean_state = mean_prop
            lpdf_state = lpdf_prop
            acc_hist[i_b, ii] = 1
        else: # reject
            pass

    # save results
    X_state[I[i_b]] = x_state

    # adapt stepsize                
    if s['adapt_step'] and ii <= s['N_b'] and ii >= s['M']:
        h_star = min( 0.001*h, c(ii) )
        acc_rate_M = np.mean( acc_hist[i_b, ii-s['M']+1:ii+1] )
        if acc_rate_M < s['tar_acc']:
            h -= h_star
        elif acc_rate_M >= s['tar_acc']:
            h += h_star
    h_state[i_b] = h

#%% functions for local update

# convolution (and adjoint) for images represented as column-stacked vectors
conv = lambda x, shape: convolution.conv(x, shape, type=p['blur'], radius=p['radius'], std=p['blur_std'], kernel=kernel, BC=p['ext_mode_blur'])
conv_T = lambda x, shape: convolution.conv_T(x, shape, type=p['blur'], radius=p['radius'], std=p['blur_std'], kernel=kernel, BC=p['ext_mode_blur'])

# finite difference operators (and adjoints)
Dh = lambda x, shape: finite_differences.Dh_op(x, M=shape[0], N=shape[1], BC=p['ext_mode_blur'])
Dv = lambda x, shape: finite_differences.Dv_op(x, M=shape[0], N=shape[1], BC=p['ext_mode_blur'])
DhT = lambda x, shape: finite_differences.DhT_op(x, M=shape[0], N=shape[1], BC=p['ext_mode_blur'])
DvT = lambda x, shape: finite_differences.DvT_op(x, M=shape[0], N=shape[1], BC=p['ext_mode_blur'])

# projection onto euclidian basis vectors
def proj(x, I):
    y = np.zeros_like(x)
    y[I] = x[I]
    return y
 
# selection of coordinates (= U^T x in notes)
def sel(x, I):
    return x[I]

# transform selected coordinates back to full space (= U x in notes)
def sel_back(x, I, d): 
    y = np.zeros(d)
    y[I] = x
    return y

#%% step adaption

def c(ii):
    # r = - np.log( dh_b/b ) / ( np.log(N_b) ) 
    return b_ad * ii**(-r_ad)

#%% twice an approximation of the square-root function

def psi(x):
    return 2*np.sqrt( x + s['eps'] )

def d_psi(x):
    return 1/np.sqrt( x + s['eps'] )

#%% log posterior pdf and gradient of blocks

def logpdf_grad(x, y_mod, bv, bh, tag):
    # x: block x without boundary values
    
    # log likelihood and gradient
    d_hathat = shape_hathat[tag][0]*shape_hathat[tag][1]
    bmAx = y_mod - sel(conv(sel_back(x, I_hathat__[tag], d_hathat), shape_hathat[tag]), I_hathat__hat[tag])
    ll = -lam/2 * np.linalg.norm( bmAx )**2 
    gll = lam * sel(conv_T(sel_back(bmAx, I_hathat__hat[tag], d_hathat), shape_hathat[tag]), I_hathat__[tag])

    # log prior and gradient
    d_til = shape_til[tag][0] * shape_til[tag][1]
    Dvx = Dv(sel_back(x, I_til__[tag], d_til), shape_til[tag]) + bv
    Dvx2 = Dvx**2
    Dhx = Dh(sel_back(x, I_til__[tag], d_til), shape_til[tag]) + bh
    Dhx2 = Dhx**2
    Dvx2_p_Dhx2 = Dvx2 + Dhx2
    lp = - p['delta']/2 * np.sum(psi(Dvx2_p_Dhx2))
    d_psi_Dvx2_p_Dhx2 = d_psi(Dvx2_p_Dhx2)
    glp = - p['delta'] * ( sel(DvT(d_psi_Dvx2_p_Dhx2 * Dvx, shape_til[tag]), I_til__[tag])
                          +sel(DhT(d_psi_Dvx2_p_Dhx2 * Dhx, shape_til[tag]), I_til__[tag]))

    # log pdf and gradient
    lpdf = ll + lp
    grad_lpdf = gll + glp

    # return ll, gll
    return lpdf, grad_lpdf
    
if __name__ == '__main__':

    #%% check input arguments
    
    if len(sys.argv) > 1: 
    
        if sys.argv[1] == 'cluster':
            cluster = 1
            # sys.path.append( PurePosixPath(r'/zhome/00/d/170891/Python'))
            os.chdir( PurePosixPath(r'/zhome/00/d/170891/Python/Localization') )
            conf = sys.argv[2]
            sam = sys.argv[3]
            ch_nr = sys.argv[4]

        else: # run from local terminal
            cluster = 0
            # sys.path.append( PureWindowsPath(r'C:\Users\raff\Python' ))
            os.chdir( PureWindowsPath( r'C:\Users\raff\Python\Localization' ) )
            conf = sys.argv[1]
            sam = sys.argv[2]
            ch_nr = sys.argv[3]

    else: # run from IDE 
        cluster = 0
        conf = '8' 
        sam = 'lM_eps3'
        ch_nr = '0' 
    
    #%% load problem

    path_conf = PurePath( r'twoD_deblurr_image/Problem_data/conf'+conf  )
    path_sam = path_conf / ('sam_'+sam)
    p = functions.load( Path( path_conf / 'par' ) )
    s = functions.load( Path( path_sam / 'par' ) )
    
    if cluster: save_dir = PurePosixPath( r'/work3/raff/Localization', s['sampling_dir'], 'ch'+ch_nr )
    else: save_dir = PureWindowsPath( s['sampling_dir'], 'ch'+ch_nr )
    Path.mkdir( Path( save_dir ), parents=True, exist_ok=True )

    [x_im, y_im_true, y_im, lam, N, d] = functions.load( Path( path_conf / 'problem' ) )
    
    if p['blur'] == 'motion': kernel = functions.motion_kernel(p['radius'], p['theta'])
    else: kernel = None

    #%% general sampling parameters

    # total number of samples
    N_total = s['N_po'] * s['th'] + s['N_b']

    # x0
    np.random.seed(int(ch_nr))
    s['x0'] = np.random.uniform(0.5, 1.5, s['x0'].size) * s['x0']

    # adaptive step
    b_ad, r_ad = None, None
    if s['adapt_step'] and s['N_b']>0:
        b_ad = 1
        r_ad = - np.log( 0.001*s['h0']/b_ad ) / ( np.log(s['N_b']) ) 
        print(f'last adaptation: {c(s["N_b"])}')

    #%% set up parallel sampling

    N = x_im.shape[0]                  # N: side length (pixels) of square image 
    if N%s['q'] != 0: sys.exit('mod(N,q)!=0')
    else: n = N//s['q']                 # number of blocks along side
    n_blocks = n**2                     # number of blocks

    # indices for selection of blocks and pixel areas withhin blocks
    par_blocks, I, I_hathat, shape_hathat, I_hathat__hat, I_hathat__, I_hathat__c, I_hat, I_til, shape_til, I_til__, I_til__c = aux_loc_MALA_TV.indices(p, s)
    
    # crop from data
    y = functions.rav(y_im)
    y_hat = []
    for i_b in range(n_blocks):
        y_hat.append( y[I_hat[i_b]] )
    
    # prepare global input
    in1 = s, p, lam, kernel
    in2 = y_hat, n_blocks, b_ad, r_ad, N_total
    in3 = I, I_hathat, shape_hathat, I_hathat__hat, I_hathat__, I_hathat__c, I_hat, I_til, shape_til, I_til__, I_til__c
    
    #%% initialize

    # sample array and storage
    n_parts = s['N_po']//s['N_save']
    sam_per_part = [s['N_save']] * n_parts
    if s['N_po']%s['N_save']:
        sam_per_part.append(s['N_po']%s['N_save'])
        n_parts +=1 
    sam_per_part.append(0) # add dummy to avoid 'list index out of range' 
    X = np.zeros((N**2, sam_per_part[0]))

    # initialize state 
    X_state_raw = RawArray('d', d)
    X_state_np = np.frombuffer(X_state_raw, dtype=np.double)
    np.copyto(X_state_np, s['x0'])

    # accepted samples
    acc_hist_raw = RawArray('b', n_blocks*(N_total+1))

    # initial step size
    h_raw = RawArray('d', n_blocks)
    h_np = np.frombuffer(h_raw, dtype=np.double)
    np.copyto(h_np, s['h0'] * np.ones(n_blocks))

    # initializer input
    in_shared = X_state_raw, h_raw, acc_hist_raw

    # counters
    kk = 0 # for saving samples in array
    ll = 0 # for saving sample arrays into physical storage

    # Pool
    if s['n_proc']>1: pool = Pool(processes=s['n_proc'], initializer=initializer, initargs=(in1, in2, in3, in_shared, int(ch_nr)))
    else: initializer(in1, in2, in3, in_shared, int(ch_nr))

    # start sampling        
    t0 = time()

    for ii in range(1, N_total + 1):

        # monitoring of sampling
        if ii%s['acc_int_disp'] == 0:
            last_acc = np.frombuffer(acc_hist_raw, dtype=np.bool_).reshape(n_blocks, (N_total+1))[:, ii-s['acc_int_disp']+1:ii+1]
            mean_sum_acc = np.mean( np.sum( last_acc, axis=1))
            mean_h = np.mean( np.frombuffer(h_raw, dtype=np.double) )
            t_int = (time() - t0)/60
            t_proj = t_int/ii * N_total
            print('sample {}/{} -- mean acc {:>4.2f}/{} -- mean step size {:.2e} -- time {:.2f}/{:.2f} minutes'.format(ii, N_total, mean_sum_acc, s['acc_int_disp'], mean_h, t_int, t_proj), end='\r')
        
        for jj in range(4):
            # map tasks to pool
            tasks = [(block, ii) for block in par_blocks[jj]]
            if s['n_proc']>1:
                res = pool.map_async(loc_upd, tasks)
                res.wait()
            else:
                for task_ii in tasks: loc_upd(task_ii)

        # save sample
        if ii > s['N_b'] and (ii-s['N_b'])%s['th']==0:
            X[:, kk] = np.frombuffer(X_state_raw, dtype=np.double)
            kk += 1

            # new samples array if needed
            if kk==sam_per_part[ll]:    
                np.save( Path( save_dir / ('p_' + str(ll)) ), X, allow_pickle=False )
                out = {'h':np.frombuffer(h_raw, dtype=np.double), 'rand_state':np.random.get_state()}
                functions.save( Path( save_dir / ('out_'+ str(ll)) ), out)
                ll += 1
                X = np.zeros((N**2, sam_per_part[ll]))
                kk = 0
    
    if s['n_proc']>1:
        pool.close()
        pool.join()

    t1 = time() - t0
    acc_rate = np.mean( np.frombuffer(acc_hist_raw, dtype=np.bool_).reshape(n_blocks, (N_total+1))[:, (s['N_b']+1):], axis=1 )
    functions.save( Path( save_dir / 'out' ), {'h':np.frombuffer(h_raw, dtype=np.double), 'time':t1, 'sam_per_part':sam_per_part[:-1], 'acc':acc_rate})

    print('\nSampling finished')