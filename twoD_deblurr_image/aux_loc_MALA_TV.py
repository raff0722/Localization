import numpy as np

from Localization.twoD_deblurr_image import functions

#%% description

# Computes the indices and shapes of all blocks which are needed for the 
# convolution, finite difference, selection, and projection operators for local MALA.
# Also computes the ranges for selecting the required image parts from the whole image.
# Everything is w.r.t. to column-stacked images, i.e., vectors.

# block indices of image
# | 0 | 1   | ... | q-1  |
# |----------------------|
# | q | q+1 | ... | 2q-1 |
# |----------------------|
# |...                   |
# |----------------------|
# | q | q+1 | ... | q^2-1|

# x + r -> x hat
# x + 2r -> x hat hat

#%% 

def indices(par, sam):

    _, _, _, _, N, _ = functions.load(  par['problem_dir'] / 'problem'  )

    q = sam['q'] # side length of block
    n = N // sam['q'] # number of blocks at one side
    ra = par['radius'] # convolution radius

    # tags for specification of position of block in image
    # t:top, b:bottom, l:left, r:right, c:center
    p = ['tl', 't', 'tr', 'l', 'c', 'r', 'bl', 'b', 'br']

    # # for checks
    # q = 2 # side length of block in pixels
    # n = 4 # number of blocks
    # ra = 1 # radius
    # N = q*n # side lenght in pixels
    # d = N**2
    # x_im = np.random.randint(10, size=(N, N))
    # x_rav = matrix_tools.rav(x_im)

    # parallel blocks with tags:
    # computes 4 lists, each containing tuples with two entries.
    # The first entries are the indices of blocks such that the blocks in each list don't overlap
    # The second entries are tags specifying the positions of the blocks in the whole image:
    # ['tl', 't', 'tr', 'l', 'c', 'r', 'bl', 'b', 'br']
    par_blocks = [ [] for i in range(4) ]
    kk = -1
    
    for ii in range(n):
        for jj in range(n):

            kk += 1

            if ii%2 == 0:
                if jj%2 == 0:
                    li = 0 # list index
                    par_blocks[li].append([kk, None])
                else:
                    li = 1 # list index
                    par_blocks[li].append([kk, None])
            else:
                if jj%2 == 0:
                    li = 2 # list index
                    par_blocks[li].append([kk, None])
                else:
                    li = 3 # list index
                    par_blocks[li].append([kk, None])

            if ii == 0:
                if jj == 0:
                    par_blocks[li][-1][1] = 'tl'
                elif jj == n-1:
                    par_blocks[li][-1][1] = 'tr'
                else:
                    par_blocks[li][-1][1] = 't'
            elif ii == n-1:
                if jj == 0:
                    par_blocks[li][-1][1] = 'bl'
                elif jj == n-1:
                    par_blocks[li][-1][1] = 'br'
                else:
                    par_blocks[li][-1][1] = 'b'
            else:
                if jj == 0:
                    par_blocks[li][-1][1] = 'l'
                elif jj == n-1:
                    par_blocks[li][-1][1] = 'r'
                else:
                    par_blocks[li][-1][1] = 'c'
        
    ## indices to extract areas from image vectors
    # i.e. reshape( vector of complete image [ r[0] ] ) = image of block 0
    # i.e. reshape( vector of complete image [ r_hat[0] ] ) = image of block 0 + radius 

    # indices for x_j within x_im
    I = []
    for ii in range(n):
        for jj in range(n):
            xv, yv = np.meshgrid(np.arange(jj*q, (jj+1)*q), np.arange(ii*q, (ii+1)*q)) # x = cols, y = rows, cartesian indexing by default!
            xv = np.ravel(xv, order='F')
            yv = np.ravel(yv, order='F')
            indices = np.ravel_multi_index((yv, xv), dims=(N, N), order='F')
            I.append( indices )
            
    # indices for hat x_j within x_im (to preselect data)
    I_hat = []
    i0 = np.insert(np.arange(q-ra, n*q-ra, q), 0, 0)
    i1 = np.append(np.arange(q+ra, n*q+ra, q), n*q)
    for ii in range(n): # rows
        for jj in range(n): # columns
            xv, yv = np.meshgrid(np.arange(i0[jj], i1[jj]), np.arange(i0[ii], i1[ii])) # x = cols, y = rows, cartesian indexing by default!
            xv = np.ravel(xv, order='F')
            yv = np.ravel(yv, order='F')
            indices = np.ravel_multi_index((yv, xv), dims=(N, N), order='F')
            I_hat.append( indices )
            
    # indices for hat hat x_j within x_im 
    I_hathat = []
    i0 = np.insert(np.arange(q-2*ra, n*q-2*ra, q), 0, 0)
    i1 = np.append(np.arange(q+2*ra, n*q+2*ra, q), n*q)
    for ii in range(n): # rows
        for jj in range(n): # columns
            xv, yv = np.meshgrid(np.arange(i0[jj], i1[jj]), np.arange(i0[ii], i1[ii])) # x = cols, y = rows, cartesian indexing by default!
            xv = np.ravel(xv, order='F')
            yv = np.ravel(yv, order='F')
            indices = np.ravel_multi_index((yv, xv), dims=(N, N), order='F')
            I_hathat.append( indices )

    # indices for tilde x_j within x_im 
    I_til = []
    i0 = np.insert(np.arange(q-1, n*q-1, q), 0, 0) 
    i1 = np.append(np.arange(q+1, n*q+1, q), n*q)
    for ii in range(n): # rows
        for jj in range(n): # columns
            xv, yv = np.meshgrid(np.arange(i0[jj], i1[jj]), np.arange(i0[ii], i1[ii])) # x = cols, y = rows, cartesian indexing by default!
            xv = np.ravel(xv, order='F')
            yv = np.ravel(yv, order='F')
            indices = np.ravel_multi_index((yv, xv), dims=(N, N), order='F')
            I_til.append( indices )

    # indices within hat hat x_j to obtain x_j
    I_hathat__ = {}
    I_hathat__c = {}
    shape_hathat = {}
    length_hathat = [q+2*ra, q+4*ra, q+2*ra]
    r_K = [np.arange(q), np.arange(2*ra,q+2*ra), np.arange(2*ra,q+2*ra)] 
    kk = 0
    for ii in range(3): # rows
        for jj in range(3): # columns
            xv, yv = np.meshgrid(r_K[jj], r_K[ii]) # x = cols, y = rows, cartesian indexing by default!
            xv = np.ravel(xv, order='F')
            yv = np.ravel(yv, order='F')
            shape = (length_hathat[ii], length_hathat[jj])
            shape_hathat[p[kk]] = shape
            I_hathat__[p[kk]] = np.ravel_multi_index((yv, xv), dims=shape, order='F')
            I_hathat__c[p[kk]] = np.setdiff1d(np.arange(shape[0]*shape[1]), I_hathat__[p[kk]])
            kk += 1
    
    # check
    # print(x_im)
    # print(res(x_rav[r_hathat[5]], 6,6))
    # print(res(x_rav[r_hathat[5]][K['c']], 2,2))

    # indices within hat hat x_j to obtain hat x_j
    I_hathat__hat = {}
    length_hathat = [q+2*ra, q+4*ra, q+2*ra]
    r_K_hat = [np.arange(q+ra), np.arange(ra,q+3*ra), np.arange(ra,q+2*ra)] 
    kk = 0
    for ii in range(3): # rows
        for jj in range(3): # columns
            xv, yv = np.meshgrid(r_K_hat[jj], r_K_hat[ii]) # x = cols, y = rows, cartesian indexing by default!
            xv = np.ravel(xv, order='F')
            yv = np.ravel(yv, order='F')
            shape = (length_hathat[ii], length_hathat[jj])
            I_hathat__hat[p[kk]] = np.ravel_multi_index((yv, xv), dims=shape, order='F')
            kk += 1

    # indices within tilde x_j to obtain x_j
    I_til__ = {}
    I_til__c = {}
    shape_til = {}
    length_til = [q+1, q+2, q+1]
    r_L = [np.arange(q), np.arange(1,q+1), np.arange(1,q+1)] # indices within x tilde to obtain x
    kk = 0
    for ii in range(3): # rows
        for jj in range(3): # columns
            xv, yv = np.meshgrid(r_L[jj], r_L[ii]) # x = cols, y = rows, cartesian indexing by default!
            xv = np.ravel(xv, order='F')
            yv = np.ravel(yv, order='F')
            shape = (length_til[ii], length_til[jj])
            shape_til[p[kk]] = shape
            I_til__[p[kk]] = np.ravel_multi_index((yv, xv), dims=shape, order='F')
            I_til__c[p[kk]] = np.setdiff1d(np.arange(shape[0]*shape[1]), I_til__[p[kk]])
            kk += 1

    return par_blocks, I, I_hathat, shape_hathat, I_hathat__hat, I_hathat__, I_hathat__c, I_hat, I_til, shape_til, I_til__, I_til__c


if __name__ == '__main__':

    indices(None, None)