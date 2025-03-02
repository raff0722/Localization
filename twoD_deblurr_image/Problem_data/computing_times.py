import numpy as np
from My_modules import pickle_routines
from pathlib import PureWindowsPath

# take MALA python wallclock times = approx. CPU time (dont take burn-in time into account)
# take MLwG wallclock and CPU times from HPC notification emails (no burn-in since fixed step size)

def main():

    # from HPC notification emails
    MWG_CPU = np.array([                            # conf 
        [125135, 122372, 126435, 123964, 121017],   # 8
        [61755, 67552, 66138, 67027, 59985],        # 7
        [32359, 36088, 31604, 31406, 30959],        # 6
        [12072, 11798, 11718, 11551, 11436]         # 5
    ], dtype=np.float64)

    # from HPC notification emails
    MWG_WALL = np.array([                           # conf
        [13820 , 13500 , 14593 , 13825 , 13194 ],   # 8
        [9347 , 13403 , 13234 , 10258 , 9151 ],     # 7
        [15424 , 13528 , 10662 , 10564 , 10107 ],   # 6
        [12077 , 11800 , 11723 , 11553 , 11439 ],   # 5
    ], dtype=np.float64)

    MALA_WALL = np.zeros((4, 5)) # without burn in!!
    kk = 0
    for ii in range(8,4,-1):
        
        # load MALA stats
        stats_path = PureWindowsPath(r'twoD_deblurr_image\Problem_data', 'conf'+str(ii), 'sam_fM\stats')
        stats = pickle_routines.load(stats_path)

        # load MALA sampling parameters
        sam_path = PureWindowsPath(r'twoD_deblurr_image\Problem_data', 'conf'+str(ii), 'sam_fM\par')
        sam = pickle_routines.load(sam_path)

        # compute sampling time per sample (without burn-in)
        for jj in range(5):
            MALA_WALL[kk, jj] = stats['out'][jj]['time'][1] / (sam['N_po']*sam['th'])

        # load MLWG sampling parameters
        sam_path = PureWindowsPath(r'twoD_deblurr_image\Problem_data', 'conf'+str(ii), 'sam_lM\par')
        sam = pickle_routines.load(sam_path)

        # compute sampling time per sample (no burn-in since fixed step size)
        for jj in range(5):
            MWG_WALL[kk, jj] = MWG_WALL[kk, jj] / (sam['N_po']*sam['th']+sam['N_b'])
            MWG_CPU[kk, jj] = MWG_CPU[kk, jj] / (sam['N_po']*sam['th']+sam['N_b'])

        kk += 1

    return MALA_WALL, MWG_WALL, MWG_CPU


    # problem = ['512', '384', '256', '128']
    # for ii in range(4):
    #     print(f'---- problem {problem[ii]} ----')
    #     print(f'mean wall-clock time')
    #     print(f'MALA: {np.mean(MALA_WALL[ii,:]*fac):.2f} pm {np.std(MALA_WALL[ii,:]*fac):.2f}')
    #     print(f'MWG:  {np.mean(MWG_WALL[ii,:]*fac):.2f} pm {np.std(MWG_WALL[ii,:]*fac):.2f}')
    #     print(f'mean CPU time')
    #     print(f'MALA: {np.mean(MALA_CPU[ii,:]*fac):.2f} pm {np.std(MALA_CPU[ii,:]*fac):.2f}')
    #     print(f'MWG:  {np.mean(MWG_CPU[ii,:]*fac):.2f} pm {np.std(MWG_CPU[ii,:]*fac):.2f}')