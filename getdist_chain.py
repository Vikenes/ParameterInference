import time
import matplotlib.backends
import matplotlib.backends.backend_agg
import numpy as np 
import h5py 
from pathlib import Path
import pandas as pd 
import yaml 
import corner 
from getdist import plots, MCSamples
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
            "alpha_s"   : r"$\mathrm{d}n_s/\mathrm{d}\ln{k}$",
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
    
    def print_autocorr_time(
            self,
            filename: str,
            steps: int,
            tau: np.ndarray,
        ):
        tau_min = np.min(tau)
        tau_max = np.max(tau)

        print(f"{filename} | {steps=}")
        print(f"{tau_min =:10.2f} | {steps/tau_min =:4.2f}")
        print(f"{tau_max =:10.2f} | {steps/tau_max =:4.2f}")
        print()
        return None
    
    def print_info(self, filename):
        chainfile = Path(self.chain_path / filename)
        fff = h5py.File(chainfile, "r")
        self.print_autocorr_time(filename, fff['chain'].shape[0], fff["tau"][:])
        print(f"{fff.keys()=}")
        print(f"{fff['chain'].shape=}")
        print(f"{fff['lnprob'].shape=}")
        print(f"{fff['tau'].shape=}")
        print()
        return None
    
    def load_samples(
            self,
            chainfile:      Path,
            burnin:         int = None,
            thin:           int = None,
            burnin_factor:  float = 10,
            thin_factor:    float = 5,
        ):
        if not chainfile.exists():
            raise FileNotFoundError(f"File {chainfile} not found. Run chain first.")
        fff = h5py.File(chainfile, "r")
        tau   = fff["tau"][:]    # (nparams,)

        if burnin is not None and type(burnin) is int:
            burnin = burnin
        else:
            burnin = int(burnin_factor * np.max(tau))
        if thin is not None and type(thin) is int:
            thin   = thin
        else:
            thin   = int(thin_factor * np.min(tau))

        chain       = fff["chain"]   # Acces chain 
        samples = chain[:].reshape(-1, chain.shape[-1])[burnin::thin, ...]   
        fff.close()
        return samples

    def plot_cosmo(
            self,
            filename:       str,
            figname:        str  = None,
            burnin:         int  = None,
            thin:           int  = None,
            burnin_factor:  float = 10,
            thin_factor:    float = 5,
            make_copy:  bool = False,
        ):

        chainfile     = Path(self.chain_path / filename)
        if not chainfile.exists():
            raise FileNotFoundError(f"File {chainfile} not found. Run chain first.")

        with h5py.File(chainfile, "r") as fff:
            # Get number of steps in chain
            n_steps     = fff["chain"].shape[0] # Get number of steps 

        samples = self.load_samples(chainfile, burnin=burnin, thin=thin, burnin_factor=burnin_factor, thin_factor=thin_factor)

    
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
        if figname is None:
            # Set figname to cosmo_filename-stem.png
            if "/" in filename:
                filename = filename.split("/")[-1]
            figname = f"cosmo_{filename.split('.')[0]}.png"

        output_file = Path(f"figures/{figname}")
        if output_file.exists() or show:
            # Add figure title
            fig.suptitle(f"{filename}, {n_steps} steps", fontsize=16)
            plt.show()
            return 
        
        print(f"Saving {output_file} ...")
        fig.savefig(output_file, dpi=200)
        plt.close(fig)


    def plot_cosmo_get_dist(
            self,
            filename:       str,
            figname:        str  = None,
            burnin:         int  = None,
            thin:           int  = None,
            burnin_factor:  float = 10,
            thin_factor:    float = 5,
        ):
        from getdist import plots, MCSamples

        chainfile     = Path(self.chain_path / filename)
        if not chainfile.exists():
            raise FileNotFoundError(f"File {chainfile} not found. Run chain first.")

        with h5py.File(chainfile, "r") as fff:
            # Get number of steps in chain
            n_steps     = fff["chain"].shape[0] # Get number of steps 

        samples = self.load_samples(
            chainfile, 
            burnin=burnin, 
            thin=thin, 
            burnin_factor=burnin_factor, 
            thin_factor=thin_factor
            )


        # Get indices where self.cosmo_param_names are found in self.emulator_param_names
        cosmo_indices           = [self.emulator_param_names.index(param) for param in self.cosmo_param_names]
        cosmo_samples           = samples[:, cosmo_indices] # MCMC samples for cosmological parameters
        cosmo_labels            = [self.param_labels_latex[param] for param in self.cosmo_param_names] # Latex labels for cosmological parameters
        fiducial_params         = self.get_fiducial_params() # Fiducial parameter values
        cosmo_fiducial_params   = [fiducial_params[i] for i in cosmo_indices] # Fiducial parameter values for cosmological parameters
        cosmo_param_ranges      = [tuple(self.param_priors[i]) for i in cosmo_indices]
        fiducial_params_dict    = {label: cosmo_fiducial_params[i] for i, label in enumerate(cosmo_labels)}
        param_limits            = {label: cosmo_param_ranges[i] for i, label in enumerate(cosmo_labels)}
        # plot_prior_ranges       = self.get_plot_priors(cosmo_param_ranges)
        # param_limits            = {label: plot_prior_ranges[i] for i, label in enumerate(cosmo_labels)}

        cosmo_samples = MCSamples(samples=cosmo_samples, names=cosmo_labels, labels=cosmo_labels)
        g = plots.get_subplot_plotter()

        g.triangle_plot(
            cosmo_samples, 
            filled=True,
            markers=fiducial_params_dict,
            )

        if show:
            # Add figure title
            # fig.suptitle(f"{filename}, {n_steps} steps", fontsize=16)
            # Make axis equal
            # plt.axis("equal")
            plt.show()
            return 
        
        if figname is None:
            # Set figname to cosmo_filename-stem.png
            figname = f"cosmo_getdist_{filename.split('.')[0]}.png"

        output_file = Path(f"figures/{figname}")
        if output_file.exists():
            print(f"File {output_file} already exists. Adding _new to filename.")
            figname = f"{output_file.stem}_new{output_file.suffix}"
            output_file = Path(f"figures/{figname}")
        print(f"Saving {output_file} ...")
        # g.savefig(output_file, dpi=200)
        g.export(figname, adir="figures")
        # plt.close(g)

    def plot_cosmo_get_dist_double(
            self,
            filename1:      str,
            filename2:      str,
            figname:        str  = None,
            burnin:         int  = None,
            thin:           int  = None,
            burnin_factor:  float = 10,
            thin_factor:    float = 5,
        ):

        chainfile1     = Path(self.chain_path / filename1)
        chainfile2     = Path(self.chain_path / filename2)
        if not chainfile1.exists():
            raise FileNotFoundError(f"File1 {chainfile1} not found. Run chain first.")
        if not chainfile2.exists():
            raise FileNotFoundError(f"File2 {chainfile2} not found. Run chain first.")

        samples1 = self.load_samples(
            chainfile1, 
            burnin=burnin, 
            thin=thin, 
            burnin_factor=burnin_factor, 
            thin_factor=thin_factor
            )
        samples2 = self.load_samples(
            chainfile2, 
            burnin=burnin, 
            thin=thin, 
            burnin_factor=burnin_factor, 
            thin_factor=thin_factor
            )

        # Get indices where self.cosmo_param_names are found in self.emulator_param_names
        cosmo_indices           = [self.emulator_param_names.index(param) for param in self.cosmo_param_names]
        cosmo_labels            = [self.param_labels_latex[param] for param in self.cosmo_param_names] # Latex labels for cosmological parameters
        fiducial_params         = self.get_fiducial_params() # Fiducial parameter values
        cosmo_fiducial_params   = [fiducial_params[i] for i in cosmo_indices] # Fiducial parameter values for cosmological parameters
        fiducial_params_dict    = {label: cosmo_fiducial_params[i] for i, label in enumerate(cosmo_labels)}
        
        cosmo_samples1          = samples1[:, cosmo_indices] # MCMC samples for cosmological parameters
        cosmo_samples2          = samples2[:, cosmo_indices] # MCMC samples for cosmological parameters

        cosmo_samples1 = MCSamples(samples=cosmo_samples1, names=cosmo_labels, labels=cosmo_labels, label=r"Vary $\mathcal{C}+\mathcal{G}$")
        cosmo_samples2 = MCSamples(samples=cosmo_samples2, names=cosmo_labels, labels=cosmo_labels, label=r"Vary $\mathcal{C}$")

        g = plots.get_subplot_plotter()

        g.triangle_plot(
            [cosmo_samples1, cosmo_samples2], 
            filled=True,
            markers=fiducial_params_dict,
            contour_colors=["blue", "red"],
            contour_args=[{"alpha": 1}, {"alpha": 0.75}],
            legend_loc="upper right",
            )

        if show:
            # Add figure title
            # fig.suptitle(f"{filename}, {n_steps} steps", fontsize=16)
            # Make axis equal
            # plt.axis("equal")
            # plt.suptitle("Testing title")
            plt.show()
            return 
        
        if figname is None:
            # Set figname to cosmo_filename-stem.png
            figname = f"cosmo_compare-{filename1.split('.')[0]}-{filename2.split('.')[0]}.png"

        output_file = Path(f"figures/{figname}")
        if output_file.exists():
            input(f"  ! File {output_file} already exists. Press enter to overwrite ...")
            figname = f"{output_file.stem}_new{output_file.suffix}"
            output_file = Path(f"figures/{figname}")
        print(f"Saving {output_file} ...")
        g.export(figname, adir="figures")

    def plot_HOD(
            self,
            filename:       str,
            figname:        str = None,
            burnin:         int = None,
            thin:           int = None,
            burnin_factor:  float = 10,
            thin_factor:    float = 5,
        ):

        chainfile     = Path(self.chain_path / filename)
        if not chainfile.exists():
            raise FileNotFoundError(f"File {chainfile} not found. Run chain first.")

        with h5py.File(chainfile, "r") as fff:
            # Get number of steps in chain
            n_steps     = fff["chain"].shape[0] # Get number of steps 

        samples = self.load_samples(chainfile, burnin=burnin, thin=thin, burnin_factor=burnin_factor, thin_factor=thin_factor)


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
            # range=HOD_param_ranges,
            max_n_ticks=3,
            use_math_text=True,
            quiet=True,
            )
        
        if show:
            plt.show()
            return
        if figname is None:
            # Set figname to HOD_filename-stem.png.
            if "/" in filename:
                filename = filename.split("/")[-1]
            figname = f"HOD_{filename.split('.')[0]}.png"

        output_file = Path(f"figures/{figname}")
        if output_file.exists() or show:
            # Add figure title
            fig.suptitle(f"{filename}, {n_steps} steps", fontsize=16)
            plt.show()
            return 

        
        print(f"Saving {output_file} ...")
        fig.savefig(output_file, dpi=200)
        plt.close(fig)

    


global show 
show = False
L = Plot_MCMC()
# L.plot_cosmo_get_dist("DE_4w_1e5.hdf5")
# L.
# L.plot_cosmo_get_dist_double(filename1="DE_4w_2e5.hdf5", filename2="vary_cosmo_DE_4w_2e5.hdf5", thin_factor=10)
# L.plot_cosmo_get_dist_double(filename1="DE_4w_1e5.hdf5", filename2="DE_4w_6e5.hdf5", thin_factor=10)
# L.plot_cosmo_get_dist_double(filename1="DE_8w_1e5.hdf5", filename2="DE_8w_2e5.hdf5", thin_factor=10)
L.plot_cosmo_get_dist_double(filename1="DE_10w_2e5.hdf5", filename2="vary_cosmo_DE_4w_2e5.hdf5", thin_factor=10)

# L.plot_cosmo_get_dist_double(filename1="DE_4w_2e5.hdf5", filename2="vary_cosmo_DE_4w_2e5.hdf5", thin_factor=10)
# L.plot_cosmo_get_dist_double(filename1="DE_4w_2e5.hdf5", filename2="vary_cosmo_DE_4w_2e5.hdf5", thin_factor=10)



# L.plot_cosmo("DE_4w_1e5_first.hdf5")


# L.plot_cosmo("fixed_HOD_DE_4w_5e4.hdf5", print_tau=True)
# L.plot_cosmo_get_dist("DE_4w_1e5.hdf5", burnin=None, thin=None)
# L.plot_cosmo_get_dist("DE_8w_1e5.hdf5", burnin=None, thin=None)
# L.plot_cosmo_get_dist("DE_10w_1e5.hdf5", burnin=None, thin=None)
# L.print_info("MGGLAM/MGGLAM_DE_4w_1e5.hdf5")
# L.print_info("vary_cosmo_DE_4w_2e5.hdf5")
# L.plot_cosmo("vary_cosmo_DE_4w_2e5.hdf5")

# L.plot_cosmo("DE_10w_1e5.hdf5")





