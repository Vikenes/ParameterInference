import numpy as np 
import h5py
import yaml 
import os 

covpath = "/mn/stornext/d5/data/vetleav/HOD_AbacusData/covariance_data_fiducial/cov_wp_fiducial.npy"
cov     = np.load(covpath)
icov    = np.linalg.inv(cov)

def compute_likelihood(
        data:       dict,
        theory:     dict,
        cov_inv:    np.ndarray,
):
    lnprob = 0. 

    # delta  = (data - theory).flatten()
    delta  = data - theory
    lnprob = -0.5 * np.einsum('i,ij,j', delta, cov_inv, delta) 
    return lnprob 


def lnprob(p, ):
    pass 

