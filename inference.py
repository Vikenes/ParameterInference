import numpy as np 
import h5py
import yaml 
import os 


def compute_likelihood(
        data:       dict,
        theory:     dict,
        cov_inv:    np.ndarray,
):
    lnprob = 0. 

    for key in data.keys():
        delta = (data[key] - theory[key]).flatten()
        # lnprob += np.dot(delta, np.dot(cov_inv, delta))
        lnprob += np.einsum('i,ij,j', delta, cov_inv, delta)
    lnprob *= -0.5

    return lnprob 


def lnprob(p, ):
    pass 

