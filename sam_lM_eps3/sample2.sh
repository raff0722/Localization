#!/bin/sh 

### General options 

###BSUB -R select[model==XeonE5_2650v4]

### -- specify queue -- 
#BSUB -q hpc

### -- set the job Name --
#BSUB -J 8_sam_lM_eps3_2

### -- ask for number of cores (default: 1) -- 
#BSUB -n 17

### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"

### -- specify that we need x GB of memory per core/slot -- 
#BSUB -R "rusage[mem=2GB]"

### -- specify that we want the job to get killed if it exceeds x GB per core/slot -- 
#BSUB -M 2GB

### -- set walltime limit: hh:mm -- 
#BSUB -W 24:00

### -- set the email address -- 
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u your_email_address

### -- send notification at start -- 
#BSUB -B

### -- send notification at completion -- 
#BSUB -N

### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o Output_%J.out 
#BSUB -e Error_%J.err 

# module load python3/3.11.4 #(3.9.12)
# module load matplotlib/3.7.1-numpy-1.24.3-python-3.11.4
# module load scipy/1.10.1-python-3.11.4

### python3 -m pip install --user imageio
### python3 -m pip install --user PyWavelets

source /zhome/00/d/170891/Python/Localization/env/bin/activate
python3 ../../../main_loc_MALA_parallel_TV.py cluster 8 lM_eps3 2 > 8_sam_lM_eps3_2.txt
### deactivate