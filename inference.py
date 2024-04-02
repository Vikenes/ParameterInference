import numpy as np 
import h5py 
from pathlib import Path
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
from scipy.integrate import simps
import pandas as pd 
import yaml 
import emcee 
import corner 
import sys 

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams.update({'font.size': 12})
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
# matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{physics}'
params = {'xtick.top': True, 'ytick.right': True, 'xtick.direction': 'in', 'ytick.direction': 'in'}
plt.rcParams.update(params)

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
        self.r_emul_input   = self.r.reshape(-1,1)
        self.r_para         = np.linspace(0, int(np.max(self.r)), int(1000))
        self.r_from_rp_rpi  = np.sqrt(self.r_perp.reshape(-1,1)**2 + self.r_para.reshape(1,-1)**2)

        self.emulator_param_names   = self.emulator.config["data"]["feature_columns"][:-1]
        self.HOD_param_names        = ["log10Mmin", "log10M1", "sigma_logM", "kappa", "alpha"]
        self.cosmo_param_names      = ["N_eff", "alpha_s", "ns", "sigma8", "w0", "wa", "wb", "wc"]
        self.nparams               = len(self.emulator_param_names)
        self.nwalkers               = self.nparams * walkers_per_param
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
        fiduci_params = self.get_fiducial_params()
        test_params += np.random.normal(0, 1e-3, size=len(test_params))

        lnprob = self.log_prob(test_params)
        print(lnprob)
            
    def load_sampler(self, 
                     nsteps=200,
                     backend=None 
                     ):
        nwalkers            = self.nwalkers
        nparams             = self.nparams
        

        sampler = emcee.EnsembleSampler(
            nwalkers, 
            nparams, 
            self.log_prob,
            backend=backend,
        )

        return sampler
    
    def run_chain(
            self,
            filename = "test2.h5",
            max_n = 100000,
            ):
        outfile = Path(self.outpath / filename)
        backend = emcee.backends.HDFBackend(outfile)
        if outfile.exists():
            rerun = input("File already exists. Overwrite? (y/n): ")
            if rerun == "y":
                print("Resetting backend, rerunning...")
                backend.reset(self.nwalkers, self.nparams)
            else:
                proceed = input("Continue from last iteration? (y/n): ")
                if proceed != "y":
                    print("Exiting...")
                    return
                else:
                    print("Continuing from last iteration...")
                
        sampler = self.load_sampler(backend=backend)

        mean_param_values   = np.mean(self.param_priors, axis=1)
        initial_guess  = mean_param_values + np.random.normal(0, 1e-3, size=(self.nwalkers, self.nparams))

        old_tau = np.inf

        for sample in sampler.sample(initial_guess, iterations=max_n, progress=True):
            if sampler.iteration % 100:
                continue

            tau = sampler.get_autocorr_time(tol=0)
            converged = np.all(tau * 100 < sampler.iteration)
            converged &= np.all(np.abs(old_tau - tau) / tau < 0.11)

            if converged:
                break

            old_tau = tau


    def plot_chain(self, filename = "test2.h5"):
        outfile     = Path(self.outpath / filename)
        if not outfile.exists():
            raise FileNotFoundError(f"File {outfile} not found. Run chain first.")
        reader      = emcee.backends.HDFBackend(outfile)
        sampler     = self.load_sampler(backend=reader)
        # tau         = reader.get_autocorr_time()
        
        # burnin      = int(2 * np.max(tau))
        # thin        = int(0.5 * np.min(tau))
        burnin = 0
        thin = 1
        samples     = reader.get_chain(discard=burnin, thin=thin, flat=True)

        ltex_labels = self.get_param_names_latex() 
        labels      = [ltex_labels[i] for i in self.emulator_param_names]
        fiducial_params = self.get_fiducial_params()
        print(f"{samples.shape=}")
        corner.corner(
            samples, 
            labels=labels,
            truths=fiducial_params,
            )
        plt.show()


    def plot_cosmo(
            self,
            filename="test.h5",
        ):

        outfile     = Path(self.outpath / filename)
        if not outfile.exists():
            raise FileNotFoundError(f"File {outfile} not found. Run chain first.")
        reader      = emcee.backends.HDFBackend(outfile)
        sampler     = self.load_sampler(backend=reader)
        
        samples     = reader.get_chain(discard=0, flat=True)

        # Get indices where self.cosmo_param_names are found in self.emulator_param_names
        cosmo_indices = [self.emulator_param_names.index(param) for param in self.cosmo_param_names]
        cosmo_samples = samples[:, cosmo_indices]
        cosmo_labels = [self.get_param_names_latex()[param] for param in self.cosmo_param_names]
        fiducial_params = self.get_fiducial_params()
        cosmo_fiducial_params = [fiducial_params[i] for i in cosmo_indices]
        fig = corner.corner(
            cosmo_samples, 
            labels=cosmo_labels,
            truths=cosmo_fiducial_params,
            )
        fig.savefig("figures/cosmo_corner.png", dpi=200)
        fig.clf()


    def plot_HOD(
            self,
            filename="test.h5",
        ):

        outfile     = Path(self.outpath / filename)
        if not outfile.exists():
            raise FileNotFoundError(f"File {outfile} not found. Run chain first.")
        reader      = emcee.backends.HDFBackend(outfile)
        sampler     = self.load_sampler(backend=reader)

        samples     = reader.get_chain(discard=0, flat=True)

        # Get indices where self.HOD_param_names are found in self.emulator_param_names
        HOD_indices = [self.emulator_param_names.index(param) for param in self.HOD_param_names]
        HOD_samples = samples[:, HOD_indices]
        HOD_labels = [self.get_param_names_latex()[param] for param in self.HOD_param_names]
        fiducial_params = self.get_fiducial_params()
        HOD_fiducial_params = [fiducial_params[i] for i in HOD_indices]
        fig = corner.corner(
            HOD_samples, 
            labels=HOD_labels,
            truths=HOD_fiducial_params,
            )
        fig.savefig("figures/HOD_corner.png", dpi=200)
        fig.clf()
        

        

L = Likelihood()
# L.run_chain()
L.plot_cosmo()
L.plot_HOD()
# L.test_log_prob()
# L.store_chain()
# L.plot_chain()
# print(L.emulator_param_names)