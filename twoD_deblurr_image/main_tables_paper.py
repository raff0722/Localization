# prints the rows of the tables in 
#
# Flock, Rafael, Shuigen Liu, Yiqiu Dong, and Xin T. Tong. 
# “Local MALA-within-Gibbs for Bayesian Image Deblurring with Total Variation Prior.” 
# arXiv, 2024. http://arxiv.org/abs/2409.09810.
# submitted to SISC
#
# in LaTeX format

from My_modules import pickle_routines

from pathlib import PurePath
import numpy as np

data_dir = PurePath( r'twoD_deblurr_image\Problem_data' )

#%% Table 1: study of different eps using MLwG with adaptive step size

print('#'*30)
print('Table 1')
print('#'*30)

# columns: nESS [%] - tau [10-6] - alpha [%] - max PSRF - median PSRF
# rows: epsilon = 1e-3, 1e-5, 1e-7

# p_... (print...) 
p_eps = ['$10^{-3}$', '$10^{-5}$', '$10^{-7}$']

for kk, ii in enumerate([3,5,7]):
    
    stats = pickle_routines.load(PurePath(r'twoD_deblurr_image\Problem_data\conf8', f'sam_lM_eps{ii}', 'stats'))
    sam = pickle_routines.load(PurePath(r'twoD_deblurr_image\Problem_data\conf8', f'sam_lM_eps{ii}', 'par'))
    
    p_eps[kk] += f'&{np.min( np.mean(stats["ESS"], axis=1) / sam["N_po"] ) *100 :.1f}'
    # p_eps[kk] += f'&{np.mean( stats["ESS"] / sam["N_po"] ) *100 :.1f}'
    p_eps[kk] += f'&{np.mean( np.array([stats["out"][ii]["h"] for ii in range(5)]) ) *1e6 :.1f}'
    p_eps[kk] += f'&{np.mean( np.array([stats["out"][ii]["acc"] for ii in range(5)]) ) *100 :.1f}'
    p_eps[kk] += f'&{np.max(stats["rhat"]) :.2f}'
    p_eps[kk] += f'&{np.median(stats["rhat"]) :.2f}'

    print(p_eps[kk]+'\\\\')

#%% Table 2: MALA with adaptive step size vs MLwG with fixed step size

print('#'*30)
print('Table 2')
print('#'*30)

# columns: problem sizes -- 128×128 - 256×256 - 384×384 - 512×512
# rows: nESS - tau - acc rate - burn-in - rhat -- each for MLwG / MALA

# p_... (print...) 

p_nESS_MLWG = '&MLwG'
p_nESS_MALA = '&MALA'

p_tau_MLWG = '&MLwG'
p_tau_MALA = '&MALA'

p_alpha_MLWG = '&MLwG'
p_alpha_MALA = '&MALA'

p_burn_MLWG = '&MLwG'
p_burn_MALA = '&MALA'

p_rhat_MLWG = '&MLwG'
p_rhat_MALA = '&MALA'

for ii in range(4):
    
    # MLwG

    stats_MLwG = pickle_routines.load(PurePath(r'twoD_deblurr_image\Problem_data', 'conf'+str(ii+5), 'sam_lM_fix\stats'))
    sam_MLwG = pickle_routines.load(PurePath(r'twoD_deblurr_image\Problem_data', 'conf'+str(ii+5), 'sam_lM_fix\par'))
    
    p_nESS_MLWG += f'&{np.min(np.mean(stats_MLwG["ESS"] / sam_MLwG["N_po"], axis=1)) *100 :.1f}'
    # p_nESS_MLWG += f'&{np.mean( stats_MLwG["ESS"] / sam_MLwG["N_po"] ) *100 :.1f}'
    p_tau_MLWG += f'&{np.mean( np.array([stats_MLwG["out"][ii]["h"] for ii in range(5)]) ) *1e6 :.1f}'
    p_alpha_MLWG += f'&{np.mean( np.array([stats_MLwG["out"][ii]["acc"] for ii in range(5)]) ) *100 :.1f}'
    p_burn_MLWG += f'&{sam_MLwG["N_b"] *1e-3 :.3f}'
    p_rhat_MLWG += f'&{np.max(stats_MLwG["rhat"]) :.2f}'
    

    # MALA

    stats_MALA = pickle_routines.load(PurePath(r'twoD_deblurr_image\Problem_data', 'conf'+str(ii+5), 'sam_fM\stats'))
    sam_MALA = pickle_routines.load(PurePath(r'twoD_deblurr_image\Problem_data', 'conf'+str(ii+5), 'sam_fM\par'))

    p_nESS_MALA += f'&{np.min(np.mean(stats_MALA["ESS"] / sam_MALA["N_po"], axis=1)) *100 :.1f}'
    # p_nESS_MALA += f'&{np.mean( stats_MALA["ESS"] / sam_MALA["N_po"] ) *100 :.1f}'
    p_tau_MALA += f'&{np.mean( np.array([stats_MALA["out"][ii]["h"] for ii in range(5)]) ) *1e6 :.1f}'
    p_alpha_MALA += f'&{np.mean( np.array([stats_MALA["out"][ii]["acc"] for ii in range(5)]) ) *100 :.1f}'
    p_burn_MALA += f'&{sam_MALA["N_b"] *1e-3 :.3f}'
    p_rhat_MALA += f'&{np.max(stats_MALA["rhat"]) :.2f}'

print(p_nESS_MLWG+'\\\\')
print(p_nESS_MALA+'\\\\')
print('-'*30)
print(p_tau_MLWG+'\\\\')
print(p_tau_MALA+'\\\\')
print('-'*30)
print(p_alpha_MLWG+'\\\\')
print(p_alpha_MALA+'\\\\')
print('-'*30)
print(p_burn_MLWG+'\\\\')
print(p_burn_MALA+'\\\\')
print('-'*30)
print(p_rhat_MLWG+'\\\\')
print(p_rhat_MALA+'\\\\')

#%% Table 3: House example: MALA vs MLwG, both with adaptive step size

print('#'*30)
print('Table 3')
print('#'*30)

# columns: nESS [%] - tau [10-6] - alpha [%] - burn-in - max PSRF
# rows: MLwG - MALA

# p_... (print...) 
p = ['MLwG', 'MALA']
sam_folder = ['sam_lM', 'sam_fM']

		# MLwG &32.2 &7.4 &54.3 &31.250 &1.02 \\
		# MALA &7.8 &1.4 &55.9 &2000.000 &1.09 \\

for kk, method in enumerate(p):
    
    stats = pickle_routines.load(PurePath(r'twoD_deblurr_image\Problem_data\conf12', sam_folder[kk], 'stats'))
    sam = pickle_routines.load(PurePath(r'twoD_deblurr_image\Problem_data\conf12', sam_folder[kk], 'par'))
    
    p[kk] += f'&{np.min( np.mean(stats["ESS"], axis=1) / sam["N_po"] ) *100 :.1f}'
    # p[kk] += f'&{np.mean( stats["ESS"] / sam["N_po"] ) *100 :.1f}'
    p[kk] += f'&{np.mean( np.array([stats["out"][ii]["h"] for ii in range(5)]) ) *1e6 :.1f}'
    p[kk] += f'&{np.mean( np.array([stats["out"][ii]["acc"] for ii in range(5)]) ) *100 :.1f}'
    p[kk] += f'&{sam["N_b"] *1e-3 :.3f}'
    p[kk] += f'&{np.max(stats["rhat"]) :.2f}'

    print(p[kk]+'\\\\')