import numpy as np 
import h5py
import yaml 
import os 

LRG_params = {
    "Mcut": 13.1,
    "M1": 14.5
    }

HOD_params = {
    "LRG_params": LRG_params,
}

tracer_flags = {"LRG": True, "KUK": False}

tracers = {}
for key in tracer_flags.keys():
    if tracer_flags[key]:
        tracers[key] = HOD_params[key+"_params"]

print(tracers)


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

