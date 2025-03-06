# Localization
Code of the preprint

Flock, Rafael, Shuigen Liu, Yiqiu Dong, and Xin T. Tong. 2024. 
“Local MALA-within-Gibbs for Bayesian Image Deblurring with Total Variation Prior.” 
arXiv. http://arxiv.org/abs/2409.09810.

The manuscript is submitted to SIAM Scientific Computing and currently under review.

## Instructions
### Setting up the problems
The used examples and sampling configurations of the paper are in the folders "conf5", "conf6", "conf7", "conf8", and "conf12" which can be found in "twoD_deblurr_image/Problem_data". 
To set up the problems, first run "main.py" in the corresponding "conf" folders. 
These scripts also contain the specific parameters of the problems. 

### Sampling via the local and parallel MALA-within-Gibbs algorithm
To sample via the method presented in the paper, first run the "main.py" script in one of the "sam_lM" folders in the respective "conf" folder. 
This script sets and saves the sampling parameters. 
Then, execute "twoD_deblurr_image/main_loc_MALA_parallel_TV.py" to run the local and parallel MALA-within-Gibbs algorithm on the specified number of cores.
The samples are not stored in this repo to save storage. 
However, the sample statistics (such as mean or 90% CI bounds) which are required to make the plots and tables of the paper are uploaded to this repo. 
The statistics are in the "stats" files in the respective "sam" folders.

### Creating the plots and tables of the paper
The plots and tables of the paper can be recreated with "twoD_deblurr_image/plots_paper.py" and "twoD_deblurr_image/tables_paper.py", respectively.

