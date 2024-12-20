# ParameterInference

Use Markov Chain Monte Carlo (MCMC) sampling to estimate posterior distrubution of cosmological parameters, by confronting emulator prediction of the projected Two-point correlation function (TPCF) to mock data of the projected TPCF.


## inference.py
Performs the MCMC analysis, varying all emulator params, i.e. cosmological and HOD parameters. 

Currently missing the following data to work:
 - Trained emulator
 - Covariance matrix
 - The corrfunc computed from the fiducial C+G parameters. Used to define the distance variable, r. 
 - The proj. corrfunc computed from the fiducial C+G parameters in s_z.
 - Yaml file containing the parameter priors. 
     - Yaml file generated from following script `HOD/HaloModel/HOD_and_cosmo_emulation/parameter_samples/make_param_prioirs_yaml_file.py` 


### Remaining scripts.
 - `X_varied_fixed_Y.py`: Same as `inference.py`, but with either C or G held fixed, varying only one subset. 
 - `plot_posteriors.py`: Makes triangle plots of posterior distr. 
 - `vary_cosmosubset.py`: Same as `inference.py`, with fixed HOD parameters. Allows for a chosen set of the cosmological parameters to be held fixed. Used to study the effect of a few parameters only.    