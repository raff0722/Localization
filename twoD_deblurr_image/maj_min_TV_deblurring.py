# Deblurring with isotropic TV prior.
#
# Function "debl" according to  
# Figueiredo, Mario A.T., José M.B. Dias, João P. Oliveira, and Robert D. Nowak. 
# “On Total Variation Denoising: A New Majorization-Minimization Algorithm and an Experimental Comparisonwith Wavalet Denoising.” 
# In 2006 International Conference on Image Processing, 2633–36. IEEE, 2006.
# 
# Function "adaptive_debl" according to  
# Oliveira, João P., José M. Bioucas-Dias, and Mário A.T. Figueiredo. 
# “Adaptive Total Variation Image Deblurring: A Majorization–Minimization Approach.” 
# Signal Processing 89, no. 9 (2009): 1683–93. https://doi.org/10.1016/j.sigpro.2009.03.018.

import sys
import numpy as np
from scipy.sparse.linalg import LinearOperator, cg

from Localization.twoD_deblurr_image import finite_differences
rav = lambda x: np.ravel(x, order='F')

def adaptive_debl(y, H, HT, BC, sigma2, mode, max_int, beta=1, alpha=1, theta='default', it_lam_max = 10, rel_diff_lam_max=1e-2, it_MM_max=5, rel_diff_x_max=1e-5, it_CG_max=100, eps=0):
    """Image has to be square.
    Smoothed TV possible by providing eps>0.

    Args:
        y (_type_): Noisy column-stacked image
        H (_type_): forward operator, operates on column-stacked images
        HT (_type_): transpose of forward operator, operates on column-stacked images
        BC (string): boundary condition for finite difference operators
        sigma2 (_type_): variance of noise
        mode (_type_): 'TV' or 'aTV' (anistropic TV, however no good choice for theta known yet)
        max_int (_type_): maximal intensity (1 or 255)
        beta (_type_): parameter of gamma hyperprior
        alpha (_type_): parameter of gamma hyperprior
        theta (str, optional): Tuning parameter of integral approximation. See paper. Defaults to 'default'.
        it_lam_max (int, optional): Max number of iterations for lambda. Defaults to 10.
        rel_diff_lam_max (_type_, optional): Relative difference stopping criterion for lambda. Defaults to 1e-2.
        it_MM_max (int, optional): Max number of maj-min iterations. Defaults to 5.
        rel_diff_x_max (_type_, optional): Relative difference stopping criterion for maj-min iterations. Defaults to 1e-5.
        it_CG_max (int, optional): Max number of iterations for CG. Defaults to 100.
        eps (int, optional): Small number for smoothing of TV. Defaults to 0.

    Returns:
        tuple: delta, MAP
    """    
  
    N = int( np.sqrt( y.size ) ) # image side length
    d = y.size # dimensionality of complete image vector

    Dv = lambda x: finite_differences.Dv_op(x, N, N, BC)
    Dh = lambda x: finite_differences.Dh_op(x, N, N, BC)
    DvT = lambda x: finite_differences.DvT_op(x, N, N, BC)
    DhT = lambda x: finite_differences.DhT_op(x, N, N, BC)

    D = lambda x: np.concatenate( (Dv(x), Dh(x)) ) # 2d x d
    DT = lambda x: DvT(x[:d]) + DhT(x[d:]) # d x 2d

    if mode == 'TV':
        if theta == 'default': theta = 0.4
        else: theta = theta
        TV = lambda x: np.sum(np.sqrt( Dv(x)**2 + Dh(x)**2 + eps))
        def W(x_t, x):
            div = np.sqrt( Dv(x_t)**2 + Dh(x_t)**2 + eps )
            w = np.zeros(d)
            ind = (div!=0)
            w[ind] = 1/div[ind]
            return np.tile(w, 2)*x
                
    else: sys.exit('Mode not implemented!')
    # elif mode == 'aTV': ## not implemented in operator form yet !!
        # if theta == 'default': theta = 0.4
        # else: theta = theta
    #     TV = lambda x: np.sum(np.abs(Dv(x)) + np.abs(Dh(x)) + eps)
    #     def W(x):
    #         div = np.abs(Dv@x+eps)
    #         w_v = np.zeros_like(x)
    #         ind = (div!=0)
    #         w_v[ind] = 1/div[ind]
    #         div = np.abs(Dh@x+eps)
    #         w_h = np.zeros_like(x)
    #         ind = (div!=0)
    #         w_h[ind] = 1/div[ind]
    #         return diags( np.concatenate( (w_v, w_h), 0 ), shape=(2*M_D, 2*M_D) )
                
    y_pr = HT(y)
    rho = 2 * ( alpha +  theta * N**2 )
    x_t = rav( np.random.randn(N, N) * 128/255 * max_int )
    
    # linear operator attributes for matrix-free CG
    dtype = np.double
    shape = (N**2, N**2)

    it_lam = 1
    rel_diff_lam = 1
    lam_t = rho * sigma2 / ( TV(x_t) + beta )

    while it_lam <= it_lam_max and rel_diff_lam > rel_diff_lam_max:
        
        print(f'Lambda iteration {it_lam}')

        it_MM = 1
        rel_diff_x = 1

        while it_MM <= it_MM_max and rel_diff_x > rel_diff_x_max:
            
            print(f'MM iteration {it_MM}')
        
            matvec = lambda x: HT(H(x)) + lam_t * DT( W(x_t, D(x) ) )
            rmatvec = lambda x: H(HT(x)) + lam_t * DT( W(x_t, D(x) ) )
            A = LinearOperator(shape=shape, matvec=matvec, rmatvec=rmatvec, dtype=dtype)

            x_tt, info = cg(A, y_pr, x0=x_t, maxiter=it_CG_max) # default atol, rtol = 0, 1e-5
            
            if info < 0:
                sys.exit(f'Warning: Illegal input or breakdown of CG.')
            else: 
                if info > 0: print(f'Warning: CG not converged.')
                rel_diff_x = np.linalg.norm( x_tt - x_t ) / np.linalg.norm( x_t ) 
                x_t = x_tt
            
            it_MM += 1
        
        lam_tt = rho * sigma2 / ( TV(x_t) + beta )
        rel_diff_lam = np.abs( (lam_tt - lam_t) / lam_t )
        lam_t = lam_tt

        it_lam += 1

    delta = lam_t/2/sigma2

    return delta, x_t


def debl(y, H, HT, BC, sigma2, mode, max_int, delta, it_MM_max=5, rel_diff_x_max=1e-5, it_CG_max=100, CG_rtol=1e-5, eps=0):
    """Image has to be square.
    Smoothed TV possible by providing eps>0.

    Args:
        y (_type_): Noisy column-stacked image
        H (_type_): forward operator, operates on column-stacked images
        HT (_type_): transpose of forward operator, operates on column-stacked images
        BC (string): boundary condition for finite difference operators
        sigma2 (_type_): variance of noise
        mode (_type_): 'TV' or 'aTV' (anistropic TV, however no good choice for theta known yet)
        max_int (_type_): maximal intensity (1 or 255)
        delta (_type_): regularization parameter
        it_MM_max (int, optional): Max number of maj-min iterations. Defaults to 5.
        rel_diff_x_max (_type_, optional): Relative difference stopping criterion for maj-min iterations. Defaults to 1e-5.
        it_CG_max (int, optional): Max number of iterations for CG. Defaults to 100.
        CG_rtol (_type_, optional): Relative difference stopping criterion for CG. Defaults to 1e-5.
        eps (int, optional): Small number for smoothing of TV. Defaults to 0.

    Returns:
        _type_: MAP
    """    

    N = int( np.sqrt( y.size ) ) # image side length
    d = y.size # dimensionality of complete image vector

    Dv = lambda x: finite_differences.Dv_op(x, N, N, BC)
    Dh = lambda x: finite_differences.Dh_op(x, N, N, BC)
    DvT = lambda x: finite_differences.DvT_op(x, N, N, BC)
    DhT = lambda x: finite_differences.DhT_op(x, N, N, BC)

    D = lambda x: np.concatenate( (Dv(x), Dh(x)) ) # 2d x d
    DT = lambda x: DvT(x[:d]) + DhT(x[d:]) # d x 2d
    
    if mode == 'TV':
        def W(x_t, x):
            div = np.sqrt( Dv(x_t)**2 + Dh(x_t)**2 + eps )
            w = np.zeros(d)
            ind = (div!=0)
            w[ind] = 1/div[ind]
            return np.tile(w, 2)*x
        
    else: sys.exit('Mode not implemented!')
    # elif mode == 'aTV': ## not implemented in operator form yet !!
    #     def W(x):
    #         div = np.abs(Dv@x+eps)
    #         w_v = np.zeros_like(x)
    #         ind = (div!=0)
    #         w_v[ind] = 1/div[ind]
    #         div = np.abs(Dh@x+eps)
    #         w_h = np.zeros_like(x)
    #         ind = (div!=0)
    #         w_h[ind] = 1/div[ind]
    #         return diags( np.concatenate( (w_v, w_h), 0 ), shape=(2*M_D, 2*M_D) )
        
    y_pr = HT(y)
    x_t = rav( np.random.randn(N, N) * 128/255 * max_int )
    lam = 2*sigma2*delta

    # linear operator attributes for matrix-free CG
    dtype = np.double
    shape = (N**2, N**2)

    # CG settings
    rel_diff_cg_tol = 1e-5 # from paper

    # mm
    it_MM = 1
    rel_diff_x = 1
    
    while it_MM <= it_MM_max and rel_diff_x > rel_diff_x_max:
        
        print(f'MM iteration {it_MM}')

        matvec = lambda x: HT(H(x)) + lam * DT( W(x_t, D(x) ) )
        rmatvec = lambda x: H(HT(x)) + lam * DT( W(x_t, D(x) ) )
        A = LinearOperator(shape=shape, matvec=matvec, rmatvec=rmatvec, dtype=dtype)

        x_tt, info = cg(A, y_pr, x0=x_t, maxiter=it_CG_max, rtol=CG_rtol) # default atol, rtol = 0, 1e-5
        
        if info < 0:
            sys.exit(f'Warning: Illegal input or breakdown of CG.')
        else: 
            if info > 0: print(f'Warning: CG not converged.')
            rel_diff_x = np.linalg.norm( x_tt - x_t ) / np.linalg.norm( x_t ) 
            x_t = x_tt
        
        it_MM += 1
    
    return x_t





