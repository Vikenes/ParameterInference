import numpy as np 
import h5py 
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
from scipy.integrate import simps
import pandas as pd 
import yaml 
import emcee 

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
        data_path           = "/mn/stornext/d5/data/vetleav/HOD_AbacusData/covariance_data_fiducial",
        emulator_path       = "emulator_data/vary_r/emulators/compare_scaling",
        emulator_version    = 6,
        walkers_per_param   = 4,
    ):
        self.data_path = Path(data_path)
        self.r_perp, self.w_p_data = self.load_wp_data()

        self.cov_matrix_inv = self.load_covariance_data()

        self.emulator       = xi_emulator_class(emulator_path, emulator_version)

        self.r              = self.get_r_from_fiducial_xi()
        self.r_emul_input   = self.r.reshape(-1,1)
        self.r_para         = np.linspace(0, int(np.max(self.r)), int(1000))
        self.r_from_rp_rpi  = np.sqrt(self.r_perp.reshape(-1,1)**2 + self.r_para.reshape(1,-1)**2)

        self.emulator_param_names   = self.emulator.config["data"]["feature_columns"][:-1]
        self.nparams               = len(self.emulator_param_names)
        self.nwalkers               = self.nparams * walkers_per_param
        self.param_priors           = self.get_parameter_priors()


    def load_covariance_data(self):
        """
        Load covariance matrix and its inverse
        Computed from the wp data loaded in "load_wp_data()"
        """
        cov_matrix             = np.load(self.data_path / "cov_wp_fiducial.npy")
        cov_matrix_inv    = np.linalg.inv(cov_matrix)
        return cov_matrix_inv
        
    def load_wp_data(self):
        """
        Load fiducial wp data
        computed from fiducial AbacusSummit simulation: c000_ph000-c000_ph024
        """
        WP = h5py.File(self.data_path / "wp_from_sz_fiducial_ng_fixed.hdf5", "r")
        r_perp = WP["rp_mean"][:]
        w_p_data = WP["wp_mean"][:]
        WP.close()
        return r_perp, w_p_data


    def get_r_from_fiducial_xi(self):
        """
        Load fiducial xi data
        computed from fiducial AbacusSummit simulation: c000_ph000-c000_ph024
        """
        XI = h5py.File(self.data_path / "tpcf_r_fiducial_ng_fixed.hdf5", "r")
        r = XI["r_mean"][:]
        XI.close()
        return r 

    def get_parameter_priors(self):

        config          = yaml.safe_load(open(f"{self.data_path}/priors_config.yaml"))
        param_priors    = np.zeros((self.nparams, 2))

        for i, param_name in enumerate(self.emulator_param_names):
            param_priors[i] = config[param_name]

        return param_priors


    def inrange(self, params):
        """
        Check if the parameters are within the prior range
        """
        return np.all((params >= self.param_priors[:,0]) & (params <= self.param_priors[:,1]))
    
    def log_likelihood(self, params):
        
        wp_theory   = self.get_wp_theory(params)
        delta       = self.w_p_data - wp_theory
        lnprob      = -0.5 * np.einsum('i,ij,j', delta, self.cov_matrix_inv, delta) 
        return lnprob
    
    def get_wp_theory(self, params):

        emul_input = np.hstack((
            params * np.ones_like(self.r_emul_input), 
            self.r_emul_input
        ))
        xi_theory = self.emulator(emul_input)

        xiR_func = IUS(
            self.r, xi_theory
        )

        w_p_theory = 2.0 * simps(
            xiR_func(
                self.r_from_rp_rpi
            ),
            self.r_para,
            axis=-1
        )
        return w_p_theory
    

    def log_prob(self, params):
        if self.inrange(params):
            lnprob = self.log_likelihood(params)
        else:
            lnprob = -np.inf
        return lnprob


    def test_log_prob(self):
        FIDUCIAL_HOD_params     = pd.read_csv(f"{D13_PATH}/fiducial_data/HOD_parameters_fiducial_ng_fixed.csv")
        FIDUCIAL_cosmo_params   = pd.read_csv(f"{D13_PATH}/fiducial_data/cosmological_parameters.dat", sep=" ")
        FIDUCIAL_params         = pd.concat([FIDUCIAL_HOD_params, FIDUCIAL_cosmo_params], axis=1)
        FIDUCIAL_params         = FIDUCIAL_params.iloc[0].to_dict()

        test_params = [FIDUCIAL_params[param] for param in self.emulator_param_names]
        test_params += np.random.normal(0, 1e-3, size=len(test_params))

        lnprob = self.log_prob(test_params)
        print(lnprob)
            
    def run_emcee(self, nsteps=200):
        nwalkers = self.nwalkers
        nparams = self.nparams
        mean_param_values = np.mean(self.param_priors, axis=1)
        self.initial_guess = mean_param_values + np.random.normal(0, 1e-3, size=(self.nwalkers, self.nparams))

        sampler = emcee.EnsembleSampler(
            nwalkers, 
            nparams, 
            self.log_prob,
        )
        sampler.run_mcmc(
            self.initial_guess, 
            nsteps, 
            progress=True)

        return sampler



L = Likelihood()
# L.test_log_prob()
L.run_emcee()
# L.load_fiducial_xi()
# L.get_parameter_priors()

# exit()


# Combine the fiducial HOD and cosmological parameters
# print(FIDUCIAL_params)
# exit()

# for key, val in FIDUCIAL_cosmo_params.items():
# for key, val in FIDUCIAL_HOD_params.items():


# lnprob = L(params)

# print(lnprob)
    



    
