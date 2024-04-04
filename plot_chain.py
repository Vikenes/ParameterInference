import time
import numpy as np 
import h5py 
from pathlib import Path
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


# sys.path.append("/uio/hume/student-u74/vetleav/Documents/thesis/HOD/HaloModel/HOD_and_cosmo_emulation/parameter_samples_plot")

D13_PATH = "/mn/stornext/d13/euclid_nobackup/halo/AbacusSummit/emulation_files/"
D5_PATH = "emulator_data/vary_r/"


class Plot_MCMC:
    def __init__(
        self,
        data_path           = "/mn/stornext/d5/data/vetleav/HOD_AbacusData/inference_data",
        emulator_path       = "emulator_data/vary_r/emulators/compare_scaling",
        emulator_version    = 6,
    ):
        self.data_path              = Path(data_path)
        self.emulator_path          = Path(f"{emulator_path}/version_{emulator_version}")

        emul_path_suffix            = "_".join(self.emulator_path.parts[-2:])
        self.chain_path             = Path(self.data_path / "chains" / emul_path_suffix)
        
        # Load parameter names, labels, priors, and fiducial values from emulator config 
        emulator_config             = self.load_config(self.emulator_path)
        self.emulator_param_names   = emulator_config["data"]["feature_columns"][:-1]
        self.HOD_param_names        = ["log10Mmin", "log10M1", "sigma_logM", "kappa", "alpha"]
        self.cosmo_param_names      = ["N_eff", "alpha_s", "ns", "sigma8", "w0", "wa", "wb", "wc"]
        self.nparams                = len(self.emulator_param_names)
        self.param_priors           = self.get_parameter_priors()
        self.param_labels_latex     = self.get_param_names_latex()

    def load_config(self, path):
        with open(path / "config.yaml", "r") as fp:
            config = yaml.safe_load(fp)
        return config

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
        return Fiducial_params


    def get_parameter_priors(self):

        config          = yaml.safe_load(open(f"{self.data_path}/priors_config.yaml"))
        param_priors    = np.zeros((self.nparams, 2))

        for i, param_name in enumerate(self.emulator_param_names):
            param_priors[i] = config[param_name]

        return param_priors



    def store_autocorr_time(
            self,
            chainfile: Path,
        ):
        """
        Estimate the autocorrelation time of all walkers in the chain 
        store it in the chainfile
        """

        file = h5py.File(chainfile, "r+")
        chain = file["chain"][:]
        tau = emcee.autocorr.integrated_time(chain, c=1, quiet=True)

        file.create_dataset("tau", data=tau)
        file.close()
        return None 


    def plot_cosmo(
            self,
            filename:   str,
            figname:    str = "test_cosmo_corner.png",
        ):

        chainfile     = Path(self.chain_path / filename)
        if not chainfile.exists():
            raise FileNotFoundError(f"File {chainfile} not found. Run chain first.")

        fff = h5py.File(chainfile, "r")
        if not "tau" in fff.keys():
            # Store autocorrelation time in chainfile if not already stored
            # Computing the autocorrelation time can take a while (30+ sec)
            fff.close()
            self.store_autocorr_time(chainfile)
            fff = h5py.File(chainfile, "r")


        # prob  = fff["lnprob"][:] # (nsteps, nwalkers) 
        tau   = fff["tau"][:]    # (nparams,)

        burnin      = int(10 * np.max(tau))
        thin        = int(0.5 * np.min(tau))
        # burnin      = 0
        # thin        = 100


        # Get samples from chain array
        chain = fff["chain"][:]                     # (nsteps, nwalkers, nparams). Same as get_chain(discard=0, thin=1, flat=False)
        samples = chain.reshape(-1, self.nparams)   # (nsteps * nwalkers, nparams)
        samples = samples[burnin::thin, ...]        # ((nsteps-burnin)//thin * nwalkers, nparams)

        # Get indices where self.cosmo_param_names are found in self.emulator_param_names
        cosmo_indices           = [self.emulator_param_names.index(param) for param in self.cosmo_param_names]
        cosmo_samples           = samples[:, cosmo_indices] # MCMC samples for cosmological parameters
        cosmo_labels            = [self.param_labels_latex[param] for param in self.cosmo_param_names] # Latex labels for cosmological parameters
        fiducial_params         = self.get_fiducial_params() # Fiducial parameter values
        cosmo_fiducial_params   = [fiducial_params[i] for i in cosmo_indices] # Fiducial parameter values for cosmological parameters
        cosmo_param_ranges      = [tuple(self.param_priors[i]) for i in cosmo_indices]

        fig = corner.corner(
            cosmo_samples, 
            labels=cosmo_labels,
            truths=cosmo_fiducial_params,
            range=cosmo_param_ranges,
            max_n_ticks=3,
            quiet=True,
            )
        # fig.savefig(f"figures/{figname}", dpi=200)
        # fig.clf()
        plt.show()


    def plot_HOD(
            self,
            filename: str,
            figname:  str = "test_HOD_corner.png",
        ):

        chainfile     = Path(self.chain_path / filename)
        if not chainfile.exists():
            raise FileNotFoundError(f"File {chainfile} not found. Run chain first.")
        
        fff = h5py.File(chainfile, "r")
        if not "tau" in fff.keys():
            # Store autocorrelation time in chainfile if not already stored
            # Computing the autocorrelation time can take a while (30+ sec)
            fff.close()
            self.store_autocorr_time(chainfile)
            fff = h5py.File(chainfile, "r")


        # prob  = fff["lnprob"][:] # (nsteps, nwalkers)
        tau   = fff["tau"][:]    # (nparams,)

        burnin      = int(10 * np.max(tau))
        thin        = int(0.5 * np.min(tau))
        # burnin      = 0
        # thin        = 100


        # Get samples from chain array
        chain = fff["chain"][:]                     # (nsteps, nwalkers, nparams). Same as get_chain(discard=0, thin=1, flat=False)
        samples = chain.reshape(-1, self.nparams)   # (nsteps * nwalkers, nparams)
        samples = samples[burnin::thin, ...]        # ((nsteps-burnin)//thin * nwalkers, nparams)
        # samples = samples[:burnin, ...]        # ((nsteps-burnin)//thin * nwalkers, nparams)


        # Get indices where self.HOD_param_names are found in self.emulator_param_names
        HOD_indices         = [self.emulator_param_names.index(param) for param in self.HOD_param_names]
        HOD_samples         = samples[:, HOD_indices]
        HOD_labels          = [self.param_labels_latex[param] for param in self.HOD_param_names]
        fiducial_params     = self.get_fiducial_params()
        HOD_fiducial_params = [fiducial_params[i] for i in HOD_indices]
        HOD_param_ranges    = [tuple(self.param_priors[i]) for i in HOD_indices]

        fig = corner.corner(
            HOD_samples, 
            labels=HOD_labels,
            truths=HOD_fiducial_params,
            range=HOD_param_ranges,
            max_n_ticks=3,
            use_math_text=True,
            quiet=True,
            )
        # fig.savefig(f"figures/{figname}", dpi=200)
        # fig.clf()
        plt.show()



L = Plot_MCMC()
# L.plot_cosmo("test.hdf5")
L.plot_cosmo("test_mean_1e-3_std1.hdf5", figname="test_mean_1e-3_std1.png")
# L.plot_cosmo("test_fidu_1_std1e-3.hdf5", figname="test_fidu_1_std1e-3.png")
# L.plot_cosmo("test_fidu_1e-3_std1.hdf5", figname="test_fidu_1e-3_std1.png")

# L.plot_HOD("test2.h5")
# L.store_chain()
# L.plot_chain()
# print(L.emulator_param_names)
