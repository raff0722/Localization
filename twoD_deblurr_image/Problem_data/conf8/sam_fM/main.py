# sampling via "full" (global) MALA and eps=1e-5

from pathlib import PureWindowsPath, Path
from multiprocessing import Pool

from Localization.twoD_deblurr_image.functions import load, save, rav
from Localization import eval_samples
from Localization.twoD_deblurr_image import sample_full_MALA_TV

def paral_work_full_MCMC(input):
    par, sa, ch_nr = input
    sample_full_MALA_TV.main(par, sa, ch_nr)

def main():

    # problem and sampling configuration
    conf_nr = '8'
    sam_nr = 'fM' # "full MALA"

    # parameters
    sam = {

        'acc_int_disp' : 100,

        'N_po' : 2_000,
        'N_save' : 200,
        'th' : 200,
        'N_b' : 2_000_000,
        'n_ch' : 5, # number of chains

        'h0' : 1e-06,
        'M' : 10,
        'adapt_step' : 1,
        'tar_acc' : .547,

        'eps' : 1e-5,

        'sampling_dir' : PureWindowsPath( r'twoD_deblurr_image\Problem_data', 'conf'+conf_nr, 'sam_'+sam_nr ),
    }
    sam['x0'] = rav( load(PureWindowsPath(r'twoD_deblurr_image\Problem_data', 'conf'+conf_nr, 'map_TV')) )
    save( Path( sam['sampling_dir'] / 'par'), sam)
    
    # tasks for CPUS
    TASKS = []
    par = load(PureWindowsPath(r'twoD_deblurr_image\Problem_data', 'conf'+conf_nr, 'par'))
    for ii in range(sam['n_ch']):
        TASKS.append((par, sam, ii))

    # map to CPUs
    # for task in TASKS:
    #     paral_work_full_MCMC(task)
    with Pool(processes=sam['n_ch']) as pool:
        pool.map(paral_work_full_MCMC, TASKS)
    pool.close()
    pool.join()
        
    # evaluate samples
    print('Eval samples...')
    flags = {'mean':1, 'HDI':0, 'CI':1, 'ESS':1, 'rhat':1}
    options = {'CI':0.90}
    eval_samples.main(sam['sampling_dir'], sam['n_ch'], sam['N_po']//sam['N_save'], flags, options)

if __name__ == '__main__':
    main()