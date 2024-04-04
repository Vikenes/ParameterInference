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

sys.path.append("/uio/hume/student-u74/vetleav/Documents/thesis/emulation/emul_utils")
from _predict import Predictor 

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
        # lnprob      = -0.5 * np.einsum('i,ij,j', delta, self.cov_matrix_inv, delta) 
        return -0.5 * delta @ self.cov_matrix_inv @ delta 
    
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

    def run_chain(
            self,
            filename: str,
            max_n:    int = int(1e5),
            ):
        
        """
        initial_step is being tested. 
        Keeping track of all filenames belonging to different setups,
        until testing is complete. 
        """
        mean_param_values   = np.mean(self.param_priors, axis=1)
        init_param_values = self.get_fiducial_params()

        if filename == "test_mean_1e-3_std1.hdf5":
            initial_step   = mean_param_values + 1e-3 * np.random.normal(0, 1, size=(self.nwalkers, self.nparams))
        elif filename == "test_fidu_1e-3_std1.hdf5":
            initial_step   = init_param_values + 1e-3 * np.random.normal(0, 1, size=(self.nwalkers, self.nparams))
        elif filename == "test_fidu_1_std1e-3.hdf5":
            initial_step   = init_param_values + np.random.normal(0, 1e-3, size=(self.nwalkers, self.nparams))
        else:
            raise ValueError("Invalid filename. Choose one of the predefined filenames.")

        outfile = Path(self.outpath / filename)
        if outfile.exists():
            msg = f"File {outfile} already exists. Choose another filename.\n"
            msg += f"  Run 'continue_chain('{outfile.name}')' to continue from last iteration."
            raise FileExistsError(msg)

        sampler = emcee.EnsembleSampler(
            self.nwalkers, 
            self.nparams, 
            self.log_prob,
        )

        old_tau = np.inf

        with h5py.File(outfile, "w") as f:
            # Create resizable datasets to store the chain
            dset_pos  = f.create_dataset("chain", (max_n, self.nwalkers, self.nparams), maxshape=(None, self.nwalkers, self.nparams))
            dset_prob = f.create_dataset("lnprob", (max_n, self.nwalkers), maxshape=(None, self.nwalkers))

            for ii, (pos, prob, state) in enumerate(sampler.sample(initial_step, iterations=max_n, progress=True)):

                dset_pos[ii] = pos
                dset_prob[ii] = prob

                if sampler.iteration % 100:
                    continue

                tau = sampler.get_autocorr_time(tol=0)
                converged = np.all(tau * 100 < sampler.iteration)
                converged &= np.all(np.abs(old_tau - tau) / tau < 0.11)

                if converged:
                    break

                old_tau = tau

            tau = emcee.autocorr.integrated_time(dset_pos, c=1, tol=0, quiet=True)
            f.create_dataset("tau", data=tau)

        return None 
 

    def continue_chain(
            self,
            filename:   str,
            max_new_iterations:      int = int(1e5),
            ):
        """
        Continue chain from last iteration in file

        Need to pass an argument to sampler to not check for independent walkers
        """

        outfile = Path(self.outpath / filename)
        if not outfile.exists():
            raise FileNotFoundError(f"File {outfile} not found. Run chain first.")
        
        sampler = emcee.EnsembleSampler(
            self.nwalkers, 
            self.nparams, 
            self.log_prob,
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

                if sampler.iteration % 100:
                    # Check for convergence every 100 iterations
                    continue

                # Compute tau, but include data from previous iterations and ommit trailing zeros
                tau = emcee.autocorr.integrated_time(dset_pos[:old_max_n+ii+1], tol=0, quiet=True)
                converged = np.all(tau * 100 < sampler.iteration)
                converged &= np.all(np.abs(old_tau - tau) / tau < 0.11)

                if converged:
                    break

                old_tau = tau

            # Resize datasets to remove potentially unused space
            dset_pos.resize(old_max_n + sampler.iteration, axis=0)
            dset_prob.resize(old_max_n + sampler.iteration, axis=0)

            # Store the autocorrelation time
            tau = emcee.autocorr.integrated_time(dset_pos, c=1, tol=0, quiet=True)
            restart_file["tau"][:] = tau
        
        return None 



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


L = Likelihood(walkers_per_param=4)
# L.run_chain("test_fidu_1_std1e-3.hdf5")
# L.run_chain("test_mean_1e-3_std1.hdf5")
# L.run_chain("test_fidu_1e-3_std1.hdf5")


# L.continue_chain("test_fidu_1e-3_std1.hdf5")
# L.plot_cosmo("test.hdf5")
# L.plot_HOD("test2.h5")
# L.store_chain()
# L.plot_chain()
# print(L.emulator_param_names)
