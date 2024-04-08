import time
import numpy as np 
import h5py 
from pathlib import Path
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
from scipy.integrate import simps
import pandas as pd 
import yaml 
import emcee 
import sys 


D13_PATH = "/mn/stornext/d13/euclid_nobackup/halo/AbacusSummit/emulation_files/"
D5_PATH = "emulator_data/vary_r/"

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams.update({'font.size': 12})
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{physics}'
params = {'xtick.top': True, 'ytick.right': True, 'xtick.direction': 'in', 'ytick.direction': 'in'}
plt.rcParams.update(params)

sys.path.append("/uio/hume/student-u74/vetleav/Documents/thesis/emulation/emul_utils")
from _predict import Predictor 


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
        data_path           = "/mn/stornext/d5/data/vetleav/HOD_AbacusData/inference_data",
        emulator_path       = "emulator_data/vary_r/emulators/compare_scaling",
        emulator_version    = 6,
        walkers_per_param   = 4,
    ):
        self.data_path              = Path(data_path)
        self.emulator_path          = Path(f"{emulator_path}/version_{emulator_version}")

        emul_path_suffix            = "_".join(self.emulator_path.parts[-2:])
        self.outpath                = Path(self.data_path / "chains" / emul_path_suffix)
        self.outpath.mkdir(parents=True, exist_ok=True)
         

        self.r_perp, self.w_p_data  = self.load_wp_data()
        self.cov_matrix_inv         = self.load_covariance_data()

        self.emulator       = xi_emulator_class(emulator_path, emulator_version)

        self.r              = self.get_r_from_fiducial_xi()
        self.r = self.r[self.r <= 60]
        self.r_emul_input   = self.r.reshape(-1,1)
        self.r_para         = np.linspace(0, int(np.max(self.r)), int(1000))
        self.r_from_rp_rpi  = np.sqrt(self.r_perp.reshape(-1,1)**2 + self.r_para.reshape(1,-1)**2)

        self.emulator_param_names   = self.emulator.config["data"]["feature_columns"][:-1]
        self.HOD_param_names        = ["log10Mmin", "log10M1", "sigma_logM", "kappa", "alpha"]
        self.cosmo_param_names      = ["N_eff", "alpha_s", "ns", "sigma8", "w0", "wa", "wb", "wc"]
        self.nparams                = len(self.emulator_param_names)
        self.nwalkers               = int(self.nparams * walkers_per_param)
        self.param_priors           = self.get_parameter_priors()

    def get_param_names_latex(self):
        HOD_param_labels = {
            "log10Mmin"     : r"$\log M_\mathrm{min}$",
            "log10M1"       : r"$\log M_1$",
            "sigma_logM"    : r"$\sigma_{\log M}$",
            "kappa"         : r"$\kappa$",
            "alpha"         : r"$\alpha$",
        }
        cosmo_param_labels = {
            "N_eff"     : r"$N_\mathrm{eff}$",
            "alpha_s"   : r"$\mathrm{d} n_s / \mathrm{d} \ln k$",
            "ns"        : r"$n_s$",
            "sigma8"    : r"$\sigma_8$",
            "w0"        : r"$w_0$",
            "wa"        : r"$w_a$",
            "wb"        : r"$\omega_b$",
            "wc"        : r"$\omega_\mathrm{cdm}$",
        }

        # Combine the two dictionaries
        param_names_latex_dict = {**HOD_param_labels, **cosmo_param_labels}
        return param_names_latex_dict
    
    def get_fiducial_params(self):
        FIDUCIAL_HOD_params     = pd.read_csv(f"{D13_PATH}/fiducial_data/HOD_parameters_fiducial_ng_fixed.csv")
        FIDUCIAL_cosmo_params   = pd.read_csv(f"{D13_PATH}/fiducial_data/cosmological_parameters.dat", sep=" ")
        FIDUCIAL_params         = pd.concat([FIDUCIAL_HOD_params, FIDUCIAL_cosmo_params], axis=1)
        FIDUCIAL_params         = FIDUCIAL_params.iloc[0].to_dict()
        
        Fiducial_params = [FIDUCIAL_params[param] for param in self.emulator_param_names]
        # print(Fiducial_params)
        return Fiducial_params

    def load_covariance_data(self):
        """
        Load covariance matrix and its inverse
        Computed from the wp data loaded in "load_wp_data()"
        """
        cov_matrix        = np.load(self.data_path / "cov_wp_fiducial.npy")
        # cov_matrix        = np.load(self.data_path / "corrcoef_wp_fiducial.npy")

        # cov_matrix = np.where(cov_matrix < 0, 0, cov_matrix)
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
        delta       = wp_theory - self.w_p_data 
        # return -0.5 * np.einsum("i,ij,j", delta, self.cov_matrix_inv, delta)
        return -0.5 * delta @ self.cov_matrix_inv @ delta, delta 
    
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

    def plot_wp_likelihood(
            self,
            filename:           str,
            stddev_factor:      float,
            max_n:              int     = int(1e5),
            moves = emcee.moves.StretchMove()
            ):
        

        

        # Initial chain 
        init_param_values = self.get_fiducial_params()
        np.random.seed(32)
        initial_step   = init_param_values + stddev_factor * np.random.normal(0, 1, size=(self.nwalkers, self.nparams))

        r_perp = self.r_perp 
        # fig, ax = plt.subplots(ncols=2, figsize=(12,8))
        fig, ax = plt.subplots(2, 2, figsize=(14,10))


        ax[0][0].plot(r_perp, r_perp * self.w_p_data, "o-", color="red", lw=1, ms=2, label="mean", zorder=100)
        ax[0][1].plot(r_perp, r_perp * self.w_p_data, "o-", color="red", lw=1, ms=2, label="mean", zorder=100)

        ax[0][0].plot([], "--", lw=0.7, alpha=0.7, c='gray', label="theory")
        ax[0][1].plot([], "--", lw=0.7, alpha=0.7, c='gray', label="theory")


        errors   = np.zeros((initial_step.shape[0], len(r_perp)))
        log_like = np.zeros(initial_step.shape[0])

        ll_neg = []
        ll_pos = []

        for i, params in enumerate(initial_step):
            wp_theory = self.get_wp_theory(params)
            err = np.abs(wp_theory / self.w_p_data - 1)
            errors[i] = err 
            # log_like[i] = self.log_likelihood(params)
            ll, delta = self.log_likelihood(params)
            log_like[i] = ll
            if ll < 0:
                ax[0][0].plot(r_perp, r_perp * wp_theory, "--", lw=0.7, alpha=0.7)
                # ax[0][0].plot(r_perp, delta, "o-", lw=0.5, alpha=0.7, ms=2)
                ax[1][0].plot(*[i, ll], "o", ms=2, lw=0.7)#, label="log_like")
            else:
                ax[0][1].plot(r_perp, r_perp * wp_theory, "--", lw=0.7, alpha=0.7)
                # ax[0][1].plot(r_perp, delta, "o-", lw=0.5, alpha=0.7, ms=2)
                ax[1][1].plot(*[i, ll], "o", ms=2, lw=0.7)



            # ax.plot(r_perp, wp_theory, "--", lw=0.5, alpha=0.5)

        # print(log_like)
        # exit()
        # ax.plot(np.arange(len(log_like)), log_like, "o", ms=2, lw=0.7, label="log_like")
        ax[0][0].set_title(r"$\log \mathcal{L} < 0$")
        ax[1][0].set_title(r"$\log \mathcal{L} < 0$")

        ax[0][1].set_title(r"$\log \mathcal{L} > 0$")
        ax[1][1].set_title(r"$\log \mathcal{L} > 0$")



        ax[0][0].set_xscale("log")
        ax[0][0].set_yscale("log")
        ax[0][1].set_xscale("log")
        ax[0][1].set_yscale("log")

        ax[1][0].set_xscale("linear")
        ax[1][0].set_yscale("linear")
        ax[1][1].set_xscale("linear")
        ax[1][1].set_yscale("linear")

        ax[0][0].set_xlabel(r"$r_\perp \: [h^{-1} \mathrm{Mpc}]$", fontsize=10)
        ax[0][1].set_xlabel(r"$r_\perp \: [h^{-1} \mathrm{Mpc}]$", fontsize=10)

        ax[1][0].set_xlabel("idx")
        ax[1][1].set_xlabel("idx")

        ax[0][0].set_ylabel(r"$w_p(r_\perp) \: [h^{-2} \mathrm{Mpc}^{2}]$", fontsize=10)
        ax[0][1].set_ylabel(r"$w_p(r_\perp) \: [h^{-2} \mathrm{Mpc}^{2}]$", fontsize=10)
        ax[1][0].set_ylabel(r"$\log \mathcal{L}$")
        ax[1][1].set_ylabel(r"$\log \mathcal{L}$")

        ax[0][0].legend()
        ax[0][1].legend()
        # ax[1][0].legend()

        # Increase spacing between top and bottom plots
        plt.subplots_adjust(hspace=0.3)

        plt.show()
        # fig.savefig("figures/likelihood_tests/delta_L.png", dpi=200)
        # fig.clf()


        exit()

        sampler = emcee.EnsembleSampler(
            self.nwalkers, 
            self.nparams, 
            self.log_prob,
            moves = moves,
        )

        for ii, (pos, prob, state) in enumerate(sampler.sample(initial_step, iterations=max_n, progress=True)):
            exit()


        return None 
 
    def plot_delta_likelihood(
            self,
            filename:           str,
            stddev_factor:      float,
            max_n:              int     = int(1e5),
            moves = emcee.moves.StretchMove()
            ):
        

        

        # Initial chain 
        init_param_values = self.get_fiducial_params()
        np.random.seed(32)
        initial_step   = init_param_values + stddev_factor * np.random.normal(0, 1, size=(self.nwalkers, self.nparams))

        r_perp = self.r_perp 
        # fig, ax = plt.subplots(ncols=2, figsize=(12,8))
        fig, ax = plt.subplots(2, 2, figsize=(14,10))


        # ax[0][0].plot(r_perp, r_perp * self.w_p_data, "o-", color="red", lw=1, ms=2, label="mean", zorder=100)
        # ax[0][1].plot(r_perp, r_perp * self.w_p_data, "o-", color="red", lw=1, ms=2, label="mean", zorder=100)

        # ax[0][0].plot([], "--", lw=0.7, alpha=0.7, c='gray', label="theory")
        # ax[0][1].plot([], "--", lw=0.7, alpha=0.7, c='gray', label="theory")


        errors   = np.zeros((initial_step.shape[0], len(r_perp)))
        log_like = np.zeros(initial_step.shape[0])

        ll_neg = []
        ll_pos = []

        for i, params in enumerate(initial_step):
            wp_theory = self.get_wp_theory(params)
            err = np.abs(wp_theory / self.w_p_data - 1)
            errors[i] = err 
            # log_like[i] = self.log_likelihood(params)
            ll, delta = self.log_likelihood(params)
            log_like[i] = ll
            if ll < 0:
                # ax[0][0].plot(r_perp, r_perp * wp_theory, "--", lw=0.7, alpha=0.7)
                ax[0][0].plot(r_perp, delta, "o-", lw=0.5, alpha=0.7, ms=2)
                ax[1][0].plot(*[i, ll], "o", ms=2, lw=0.7)#, label="log_like")
            else:
                # ax[0][1].plot(r_perp, r_perp * wp_theory, "--", lw=0.7, alpha=0.7)
                ax[0][1].plot(r_perp, delta, "o-", lw=0.5, alpha=0.7, ms=2)
                ax[1][1].plot(*[i, ll], "o", ms=2, lw=0.7)



            # ax.plot(r_perp, wp_theory, "--", lw=0.5, alpha=0.5)

        # print(log_like)
        # exit()
        # ax.plot(np.arange(len(log_like)), log_like, "o", ms=2, lw=0.7, label="log_like")
        ax[0][0].set_title(r"$\log \mathcal{L} < 0$")
        ax[1][0].set_title(r"$\log \mathcal{L} < 0$")

        ax[0][1].set_title(r"$\log \mathcal{L} > 0$")
        ax[1][1].set_title(r"$\log \mathcal{L} > 0$")



        ax[0][0].set_xscale("log")
        # ax[0][0].set_yscale("log")
        ax[0][1].set_xscale("log")
        # ax[0][1].set_yscale("log")

        ax[1][0].set_xscale("linear")
        ax[1][0].set_yscale("linear")
        ax[1][1].set_xscale("linear")
        ax[1][1].set_yscale("linear")

        ax[0][0].set_xlabel(r"$r_\perp \: [h^{-1} \mathrm{Mpc}]$", fontsize=10)
        ax[0][1].set_xlabel(r"$r_\perp \: [h^{-1} \mathrm{Mpc}]$", fontsize=10)

        ax[1][0].set_xlabel("idx")
        ax[1][1].set_xlabel("idx")

        ax[0][0].set_ylabel(r"$\Delta = (w_p^\mathrm{emul} - \overline{w_p})$", fontsize=15) # \: [h^{-2} \mathrm{Mpc}^{2}]$", fontsize=10)
        ax[0][1].set_ylabel(r"$\Delta = (w_p^\mathrm{emul} - \overline{w_p})$", fontsize=15) # \: [h^{-2} \mathrm{Mpc}^{2}]$", fontsize=10)

        ax[1][0].set_ylabel(r"$\log \mathcal{L}$")
        ax[1][1].set_ylabel(r"$\log \mathcal{L}$")

        # ax[0][0].legend()
        # ax[0][1].legend()
        # ax[1][0].legend()

        # Increase spacing between top and bottom plots
        plt.subplots_adjust(hspace=0.3)

        # plt.show()
        fig.savefig("figures/likelihood_tests/delta_L.png", dpi=200)
        fig.clf()


        exit()



"""
TODO:

 - Implement method for keeping certain parameters fixed
 - Make som plots of wp_theory to ensure that it's reasonable. 

"""

L4 = Likelihood(walkers_per_param=4)
L4.plot_delta_likelihood("test.hdf5", stddev_factor=1e-3, max_n=int(10))
# L4.plot_wp_likelihood("test.hdf5", stddev_factor=1e-3, max_n=int(10))

# L8 = Likelihood(walkers_per_param=8)
# L12 = Likelihood(walkers_per_param=12)

# L12.continue_chain("DEMove_fidu_std1e-3_12w.hdf5", check_convergence=False, max_new_iterations=int(2e5))

