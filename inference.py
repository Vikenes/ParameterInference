import numpy as np 
import h5py 
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
from scipy.integrate import simps
import pandas as pd 
import yaml 

import sys 
sys.path.append("/uio/hume/student-u74/vetleav/Documents/thesis/emulation/emul_utils")
from _predict import Predictor 

sys.path.append("/uio/hume/student-u74/vetleav/Documents/thesis/HOD/HaloModel/HOD_and_cosmo_emulation/parameter_samples_plot")

D13_PATH = "/mn/stornext/d13/euclid_nobackup/halo/AbacusSummit/emulation_files/"
D5_PATH = "emulator_data/vary_r/"

class xi_emulator_class:
    def __init__(
            self, 
            LIGHTING_LOGS_PATH  = "emulator_data/vary_r/emulators/compare_scaling",
            version             =   6,
            ):
        path            = Path(f"{LIGHTING_LOGS_PATH}/version_{version}")
        self.predictor  = Predictor.from_path(path)
        self.config     = self.predictor.load_config(path)


    def __call__(
        self,
        params,
    ):
        return self.predictor(np.array(params)).reshape(-1)
    



class Likelihood:
    def __init__(
        self,
        data_path = "/mn/stornext/d5/data/vetleav/HOD_AbacusData/covariance_data_fiducial",
        emulator_path = "emulator_data/vary_r/emulators/compare_scaling",
        emulator_version = 6
    ):
        self.data_path = Path(data_path)
        self.load_wp_data()
        self.load_fiducial_xi()
        
        self.load_covariance_data()

        self.emulator = xi_emulator_class(emulator_path, emulator_version)

        self.r_para = np.linspace(0, 100, int(1000))
        self.r_from_rp_rpi   = np.sqrt(self.r_perp.reshape(-1,1)**2 + self.r_para.reshape(1,-1)**2)

        emulator_config = self.emulator.config
        self.emulator_param_names = emulator_config["data"]["feature_columns"][:-1]


    def load_covariance_data(self):
        """
        Load covariance matrix and its inverse
        Computed from the wp data loaded in "load_wp_data()"
        """
        self.cov_matrix        = np.load(self.data_path / "cov_wp_fiducial.npy")
        self.cov_matrix_inv    = np.linalg.inv(self.cov_matrix)
        
    def load_wp_data(self):
        """
        Load fiducial wp data
        computed from fiducial AbacusSummit simulation: c000_ph000-c000_ph024
        """
        WP = h5py.File(self.data_path / "wp_from_sz_fiducial_ng_fixed.hdf5", "r")
        self.r_perp = WP["rp_mean"][:]
        self.w_p_data = WP["wp_mean"][:]
        WP.close()


    def load_fiducial_xi(self):
        """
        Load fiducial xi data
        computed from fiducial AbacusSummit simulation: c000_ph000-c000_ph024
        """
        XI = h5py.File(self.data_path / "tpcf_r_fiducial_ng_fixed.hdf5", "r")
        r_lst = []
        xi_lst = []
        for ph in range(25):
            XI_sim = XI[f"AbacusSummit_base_c000_ph{str(ph).zfill(3)}"]
            r_lst.append(XI_sim["r"][:])
            xi_lst.append(XI_sim["xi"][:])

        XI.close()

        self.r_data     = np.array(r_lst)
        self.xi_data    = np.array(xi_lst)

    def get_parameter_priors(self):

        config = yaml.safe_load(open(f"{self.data_path}/priors_config.yaml"))
        N_params = len(config) 
        self.param_priors = np.zeros((N_params, 2))

        for i, param_name in enumerate(self.emulator_param_names):
            self.param_priors[i] = config[param_name]

        mean_param_values = np.mean(self.param_priors, axis=1)
        ff = np.ones_like(mean_param_values) * 0.3
        nwalkers = N_params * 4
        self.initial_guess = mean_param_values + np.random.normal(0, 1, size=(nwalkers, N_params)) * ff[None, :]
        print(f"{self.initial_guess.shape=}")



    def __call__(
        self,
        params,
    ):
        r = self.r_data[0]
        Nr = len(r)
        emul_input = np.column_stack((
            np.vstack(
                [params] * Nr
                )
            ,r
            ))
        
        xi = self.emulator(emul_input)
        xiR_func = IUS(
            r, xi
        )


        w_p_theory = 2.0 * simps(
            xiR_func(
                self.r_from_rp_rpi
            ),
            self.r_para,
            axis=-1
        )        

        delta = self.w_p_data - w_p_theory
        # lnprob = -0.5 * delta.T @ self.cov_matrix_inv @ delta
        lnprob = -0.5 * np.einsum('i,ij,j', delta, self.cov_matrix_inv, delta) 
        return lnprob


L = Likelihood()
# L.load_fiducial_xi()
L.get_parameter_priors()

exit()
HOD_param_names = ['sigma_logM', 'alpha', 'kappa', 'log10M1', 'log10Mmin']
TPCF_data = h5py.File(D5_PATH+"TPCF_test_ng_fixed.hdf5", "r")
c0 = TPCF_data["AbacusSummit_base_c000_ph000"]["node0"]
HOD_params = pd.read_csv(f"{D13_PATH}/AbacusSummit_base_c000_ph000/HOD_parameters/HOD_parameters_fiducial_ng_fixed.csv")
# print(HOD_params)
# print(c0.attrs.keys())
np.random.seed(123)
params = []
for param in L.emulator_param_names:
    if param in HOD_param_names:
        pval = HOD_params[param].values[0]
    else:
        pval = c0.attrs[param]
    params.append(pval + np.random.normal(0, 1) * 1e-4)


# lnprob = L(params)

# print(lnprob)
    
