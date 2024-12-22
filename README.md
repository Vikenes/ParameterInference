# ParameterInference

Use Markov Chain Monte Carlo (MCMC) sampling to estimate posterior distrubution of cosmological parameters, by confronting emulator prediction of the projected Two-point correlation function (TPCF) to mock data of the projected TPCF.

## prerequisites
Before these scripts can be run, certain lost data needs to be regenerated. To get everything in place, the following has to be done:
 1. Go through the necessary steps outlined in the various READMEs of the HOD repo. Particularly, the README of `HOD/HaloModel/HOD_and_cosmo_emulation`. 
 2. Train a new emulator using the `TPCFEmulationCosmoHOD` repository. *Note:* The code in this repository requires scripts from the `EmulationUtilities` repo in order to work. Remaining details is outlined in the READE of `TPCFEmulationCosmoHOD`



## inference.py
Performs the MCMC analysis, varying all emulator params, i.e. cosmological and HOD parameters. 

Currently missing the following data to work:
 - Trained emulator
 - Covariance matrix
 - The corrfunc, xi(r) computed from the fiducial C+G parameters. Used to define the distance variable, r. This r is used to both make emulator predictions from which wp is computed, and to get the correct r_para and r_perp arrays to compute wp(r_perp).   
 - The proj. corrfunc computed from the fiducial C+G parameters in s_z, i.e. the "true value". 
 - Yaml file containing the parameter priors. 
     - Yaml file generated from following script `HOD/HaloModel/HOD_and_cosmo_emulation/parameter_samples/make_param_prioirs_yaml_file.py` 


### Remaining scripts.
 - `X_varied_fixed_Y.py`: Same as `inference.py`, but with either C or G held fixed, varying only one subset. 
 - `plot_posteriors.py`: Makes triangle plots of posterior distr. 
 - `vary_cosmosubset.py`: Same as `inference.py`, with fixed HOD parameters. Allows for a chosen set of the cosmological parameters to be held fixed. Used to study the effect of a few parameters only.    