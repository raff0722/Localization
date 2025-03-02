# Localizations
Code of the paper “Local MALA-within-Gibbs for Bayesian Image Deblurring with Total Variation Prior” (http://arxiv.org/abs/2409.09810) which is submitted to SIAM Scientific Computing and currently under review.

## Instructions
The used examples and sampling configurations of the paper are in the folders "conf5", "conf6", "conf7", "conf8", and "conf12" which can be found in "twoD_deblurr_image/Problem_data". 
To set up the problems, first run "main.py" in the corresponding "conf" folders. 
This script also contains the specific parameters of the problems. 
To sample via the method presented in the paper, first run the "main.py" script in one of the "sam_lM" folders in the respective "conf" folder. 
This script sets and saves the sampling parameters. 
Then, execute "twoD_deblurr_image/main_loc_MALA_parallel_TV.py" to run the local and parallel MALA-within-Gibbs algorithm.
The samples are not stored in this repo to save storage. 
However, the sample statistics (such as mean or 90% CI bounds) are available. 
The plots and tables of the paper can be recreated with "plots_paper.py" and "tables_paper.py", respectively.
