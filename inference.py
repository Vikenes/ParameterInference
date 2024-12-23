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

sys.path.append("/Path/to/EmulationUtilities")
from _predict import Predictor 

D13_PATH = "/mn/stornext/d13/euclid_nobackup/halo/AbacusSummit/emulation_files/"

"""
Script for running MCMC chains using emcee

The xi_emulator_class is used to load the trained emulator 
It is called by the Likelihood class, yielding xi(r,C,G), from which wp(rp) is computed.

"""


class xi_emulator_class:
    def __init__(
            self, 
            LIGHTING_LOGS_PATH  = "data/emulator_data/sliced_r/emulators/batch_size_3040",
            version             =   2,
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
        data_path           = "data/inference_data",
        emulator_path       = "data/emulator_data/sliced_r/emulators/batch_size_3040",
        emulator_version    = 2,
        walkers_per_param   = 4,
        r_min               = 0.0,
        r_max               = 105.0,
    ):

        self.data_path              = Path(data_path)

        self.emulator_path          = Path(f"{emulator_path}/version_{emulator_version}")
        self.emulator               = xi_emulator_class(emulator_path, emulator_version)

        emul_path_suffix            = "_".join(self.emulator_path.parts[-3:])
        self.outpath                = Path(self.data_path / "chains" / emul_path_suffix)
        self.outpath.mkdir(parents=True, exist_ok=True)

        self.cov_matrix_inv         = self.load_covariance_matrix()

        # Use wp data from fiducial AbacusSummit simulation, NOT MGGLAM
        self.r_perp, self.w_p_data  = self.load_wp_data()

        r_xi           = self.get_r_from_fiducial_xi()
        self.r_xi     = r_xi[(r_xi >= r_min) & (r_xi <= r_max)]
        self.r_emul_input   = self.r_xi.reshape(-1,1)
        self.r_para         = np.linspace(0, r_max, int(1000))
        self.r_from_rp_rpi  = np.sqrt(self.r_perp.reshape(-1,1)**2 + self.r_para.reshape(1,-1)**2)

        self.emulator_param_names   = self.emulator.config["data"]["feature_columns"][:-1]
        self.HOD_param_names        = ["log10_ng", "log10M1", "sigma_logM", "kappa", "alpha"]
        self.cosmo_param_names      = ["N_eff", "alpha_s", "ns", "sigma8", "w0", "wa", "wb", "wc"]
        self.nparams                = len(self.emulator_param_names)
        self.nwalkers               = int(self.nparams * walkers_per_param)
        self.param_priors           = self.get_parameter_priors()

    def get_fiducial_params(self):
        FIDUCIAL_HOD_params     = pd.read_csv(f"{D13_PATH}/fiducial_data/HOD_parameters_fiducial.csv")
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
        
    def load_covariance_matrix(self):
        """
        Load covariance matrix and its inverse
        Computed from the wp data loaded in "load_wp_data()"
        """
        # Load covariance matrix
        # Scale by 1/4^3, since cov matrix is computed from simulations with 1/4^3 times the volume
        cov_matrix          = np.load(self.data_path / "cov_wp_small.npy") / 64.0 
        return np.linalg.inv(cov_matrix)

        
    def load_wp_data(self):
        """
        Load fiducial wp data
        computed from fiducial AbacusSummit simulation: c000_ph000-c000_ph024
        """
        WP = h5py.File(self.data_path / "wp_from_sz_fiducial.hdf5", "r")
        r_perp = WP["rp_mean"][:]
        w_p_data = WP["wp_mean"][:]
        WP.close()
        return r_perp, w_p_data

    def get_r_from_fiducial_xi(self):
        """
        Load fiducial xi data
        computed from fiducial AbacusSummit simulation: c000_ph000-c000_ph024
        """
        XI = h5py.File(self.data_path / "tpcf_r_fiducial.hdf5", "r")
        r  = XI["r_mean"][:]
        XI.close()
        return r 
    

    def inrange(self, params):
        """
        Check if the parameters are within the prior range
        """
        return np.all((params >= self.param_priors[:,0]) & (params <= self.param_priors[:,1]))
    
    def get_wp_theory(self, params):

        emul_input = np.hstack((
            params * np.ones_like(self.r_emul_input), 
            self.r_emul_input
        ))
        xi_theory = self.emulator(emul_input)

        xiR_func = IUS(
            self.r_xi, xi_theory, ext=1,
        )

        w_p_theory = 2.0 * simps(
            xiR_func(
                self.r_from_rp_rpi
            ),
            self.r_para,
            axis=-1
        )
        return w_p_theory
    
    def log_likelihood(self, params):
        
        wp_theory   = self.get_wp_theory(params)
        delta       = wp_theory - self.w_p_data 
        return -0.5 * delta @ self.cov_matrix_inv @ delta 
    
    def log_prob(self, params):
        if self.inrange(params):
            lnprob = self.log_likelihood(params)
        else:
            lnprob = -np.inf
        return lnprob


    def run_chain(
            self,
            filename:      str,
            stddev_factor: float,
            max_n:         int      = int(1e5),
            moves                   = emcee.moves.DEMove()
            ):
        

        outfile = Path(self.outpath / filename)
        if outfile.exists():
            msg = f"File {outfile} already exists. Choose another filename.\n"
            msg += f"  Run 'continue_chain('{outfile.name}')' to continue from last iteration."
            raise FileExistsError(msg)
        else:
            print(f"Running chain, storing in {outfile}...")

        # Initial chain 
        init_param_values = self.get_fiducial_params()
        np.random.seed(4200)
        initial_step   = init_param_values + stddev_factor * np.random.normal(0, 1, size=(self.nwalkers, self.nparams))

        sampler = emcee.EnsembleSampler(
            self.nwalkers, 
            self.nparams, 
            self.log_prob,
            moves = moves,
        )

        
        with h5py.File(outfile, "w") as f:
            # Create resizable datasets to store the chain
            dset_pos  = f.create_dataset("chain", (max_n, self.nwalkers, self.nparams), maxshape=(None, self.nwalkers, self.nparams))
            dset_prob = f.create_dataset("lnprob", (max_n, self.nwalkers), maxshape=(None, self.nwalkers))

            for ii, (pos, prob, state) in enumerate(sampler.sample(initial_step, iterations=max_n, progress=True)):

                dset_pos[ii] = pos
                dset_prob[ii] = prob

            tau = emcee.autocorr.integrated_time(dset_pos, c=1, tol=0, quiet=True)
            f.create_dataset("tau", data=tau)

        print(f"Completed chain run for {outfile.name}.")
        return None 
 
    def continue_chain(
            self,
            filename:           str,
            max_new_iterations: int  = int(1e5),
            moves                    = emcee.moves.DEMove()
            ):
        """
        Continue chain from last iteration in file

        Need to pass an argument to sampler to not check for independent walkers
        """

        outfile = Path(self.outpath / filename)
        if not outfile.exists():
            raise FileNotFoundError(f"File {outfile} not found. Run chain first.")
        else:
            print(f"Continuing chain from {outfile.name}, running {max_new_iterations} new steps...")
        
        sampler = emcee.EnsembleSampler(
            self.nwalkers, 
            self.nparams, 
            self.log_prob,
            moves = moves,
        )
        
        
        with h5py.File(outfile, "r+") as restart_file:
            dset_pos  = restart_file["chain"]
            dset_prob = restart_file["lnprob"]

            dset_walkers = dset_pos.shape[1]
            assert dset_walkers == self.nwalkers, f"Number of walkers in file ({dset_walkers}) does not match number of walkers in class ({self.nwalkers})."

            # Load tau from file if already computed, otherwise compute it
            if "tau" in restart_file.keys():
                old_tau = restart_file["tau"][:]
            else:
                old_tau = emcee.autocorr.integrated_time(dset_pos, c=1, tol=0, quiet=True)

            # Save old tau in a new dataset
            tau_index = 0
            while f"tau_{tau_index}" in restart_file.keys():
                tau_index += 1
            restart_file.create_dataset(f"tau_{tau_index}", data=old_tau)
            
            # Use last position of chain as new initial step
            initial_step = dset_pos[-1] 
            old_max_n = dset_pos.shape[0]

            # Expand data sets to store new data
            dset_pos.resize(old_max_n + max_new_iterations, axis=0)
            dset_prob.resize(old_max_n + max_new_iterations, axis=0)

            for ii, (pos, prob, state) in enumerate(sampler.sample(initial_step, iterations=max_new_iterations, progress=True, skip_initial_state_check=True)):
                # Store new data 
                dset_pos[old_max_n + ii] = pos
                dset_prob[old_max_n + ii] = prob

            # Resize datasets to remove potentially unused space
            dset_pos.resize(old_max_n + sampler.iteration, axis=0)
            dset_prob.resize(old_max_n + sampler.iteration, axis=0)

            # Store the autocorrelation time
            tau = emcee.autocorr.integrated_time(dset_pos, c=1, tol=0, quiet=True)
            restart_file["tau"][:] = tau
                
        return None 

    
L4 = Likelihood(walkers_per_param=4, r_min=0.1, r_max=70.0)
# L8 = Likelihood(walkers_per_param=6)
# L10 = Likelihood(walkers_per_param=10)
# L4.run_chain("DE_4w_1e5.hdf5", stddev_factor=1e-3, max_n=int(1e5), moves=emcee.moves.DEMove())
# L8.run_chain("DE_6w_2e5.hdf5", stddev_factor=1e-3, max_n=int(2e5), moves=emcee.moves.DEMove())
# L4.run_chain("DE_4w_1e5_r0.1_70.hdf5", stddev_factor=1e-3, max_n=int(1e5), moves=emcee.moves.DEMove())
