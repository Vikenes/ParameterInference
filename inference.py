import numpy as np 
import sys 
sys.path.append("/uio/hume/student-u74/vetleav/Documents/thesis/emulation/emul_utils")
from _predict import Predictor 



class xi_emulator_class:
    def __init__(
            self, 
            LIGHTING_LOGS_PATH  = "emulator_data/emulators/compare_scaling",
            version             =   6,
            ):
        self.predictor = Predictor.from_path(f"{LIGHTING_LOGS_PATH}/version_{version}")

    def __call__(
        self,
        params,
    ):
        return self.predictor(np.array(params)).reshape(-1)
    




emulator = xi_emulator_class()

covpath = "/mn/stornext/d5/data/vetleav/HOD_AbacusData/covariance_data_fiducial/cov_wp_fiducial.npy"
cov     = np.load(covpath)
icov    = np.linalg.inv(cov)


def compute_likelihood(
        data:       dict,
        theta:      dict,
        cov_inv:    np.ndarray,
):
    lnprob = 0. 
    # xi = emulator(theta)
    # wp = xi_to_wp(xi)
    # theory_C = emulator(theta_c, G_fid)
    # delta  = (data - theory).flatten()
    theory = 0. 
    delta  = data - theory

    # xiR -> wpR 

    lnprob = -0.5 * np.einsum('i,ij,j', delta, cov_inv, delta) 
    return lnprob 


def lnprob(p, ):
    pass 

