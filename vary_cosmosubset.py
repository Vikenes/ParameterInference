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
        self.HOD_param_names        = ["log10M1", "sigma_logM", "kappa", "alpha", "log10_ng"]
        self.cosmo_param_names      = ["N_eff", "alpha_s", "ns", "sigma8", "w0", "wa", "wb", "wc"]
        self.load_fixed_and_varying_params(fixed_cosmo_params=["wa", "alpha_s", "w0"])
        self.param_priors           = self.get_parameter_priors()
        self.nparams                = self.param_priors.shape[0]
        self.nwalkers               = int(self.nparams * walkers_per_param)

    def load_fixed_and_varying_params(self, fixed_cosmo_params):
        FIDUCIAL_HOD_params     = pd.read_csv(f"{D13_PATH}/fiducial_data/HOD_parameters_fiducial.csv")
        FIDUCIAL_cosmo_params   = pd.read_csv(f"{D13_PATH}/fiducial_data/cosmological_parameters.dat", sep=" ")
        FIDUCIAL_params         = pd.concat([FIDUCIAL_HOD_params, FIDUCIAL_cosmo_params], axis=1)
        FIDUCIAL_params         = FIDUCIAL_params.iloc[0].to_dict()

        fixed_param_names       = [param for param in self.emulator_param_names if param in fixed_cosmo_params or param in self.HOD_param_names]
        varying_param_names     = [param for param in self.emulator_param_names if param not in fixed_param_names]
        # self.HOD_params         = [FIDUCIAL_params[param] for param in self.emulator_param_names if param in self.HOD_param_names]
        # self.cosmo_params       = [FIDUCIAL_params[param] for param in self.emulator_param_names if param in self.cosmo_param_names]
        self.varying_params     = [FIDUCIAL_params[param] for param in self.emulator_param_names if param not in fixed_param_names]
        varying_param_names     = [param for param in self.emulator_param_names if param not in fixed_param_names]
        # Get the indices of the varying parameters in emulator_param_names
        self.varying_param_indices = [self.emulator_param_names.index(param) for param in varying_param_names]
        self.fixed_param_indices   = [self.emulator_param_names.index(param) for param in fixed_param_names]
        fixed_param_values         = [FIDUCIAL_params[param] for param in self.emulator_param_names if param in fixed_param_names]
        self.input_params          = np.zeros_like(self.emulator_param_names, dtype=float)
        self.input_params[self.fixed_param_indices] = fixed_param_values
        self.fixed_param_names = fixed_param_names



    def get_parameter_priors(self):
        config          = yaml.safe_load(open(f"{self.data_path}/priors_config.yaml"))
        param_priors    = []
        for param_name in self.emulator_param_names:
            if not param_name in self.fixed_param_names:
                param_priors.append(config[param_name])
        param_priors = np.array(param_priors)
        return param_priors
        
    def load_covariance_matrix(self):
        """
        Load covariance matrix and its inverse
        Computed from the wp data loaded in "load_wp_data()"
        """
        cov_matrix          = np.load(self.data_path / "cov_wp_small.npy") / 64.0 # Load covariance matrix
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
    

    def inrange(self, cosmo_params):
        """
        Check if the parameters are within the prior range
        """
        return np.all((cosmo_params >= self.param_priors[:,0]) & (cosmo_params <= self.param_priors[:,1]))
    
    def get_wp_theory(self, varying_params):
        # place varying parameters in self.input_params
        input_params = self.input_params.copy()
        input_params[self.varying_param_indices] = varying_params
        emul_input = np.hstack((
            input_params * np.ones_like(self.r_emul_input), 
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
    
    def log_likelihood(self, cosmo_params):
        wp_theory   = self.get_wp_theory(cosmo_params)
        delta       = wp_theory - self.w_p_data 
        return -0.5 * delta @ self.cov_matrix_inv @ delta 
    
    def log_prob(self, cosmo_params):
        if self.inrange(cosmo_params):
            lnprob = self.log_likelihood(cosmo_params)
        else:
            lnprob = -np.inf
        return lnprob


    def run_chain(
            self,
            filename:      str,
            stddev_factor: float,
            max_n:         int     = int(5e4),
            moves                  = emcee.moves.DEMove()
            ):
        

        outfile = Path(self.outpath / filename)
        if outfile.exists():
            msg = f"File {outfile} already exists. Choose another filename.\n"
            msg += f"  Run 'continue_chain('{outfile.name}')' to continue from last iteration."
            raise FileExistsError(msg)
        else:
            print(f"Running chain, storing in {outfile}...")

        # Initial chain 
        np.random.seed(4200)
        initial_step   = self.varying_params + stddev_factor * np.random.normal(0, 1, size=(self.nwalkers, self.nparams))

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
 
L4 = Likelihood(walkers_per_param=4, r_min=0.0, r_max=105.0)
L4.run_chain("vary_cosmo_wa_alphas_w0_fixed_DE_4w_1e5.hdf5", stddev_factor=1e-3, max_n=int(1e5), moves=emcee.moves.DEMove())
