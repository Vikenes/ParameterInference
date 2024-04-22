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
        emulator_path       = "emulator_data/sliced_r/emulators/batch_size_3040",
        emulator_version    = 2,
    ):
        self.data_path              = Path(data_path)
        self.emulator_path          = Path(f"{emulator_path}/version_{emulator_version}")

        emul_path_suffix            = "_".join(self.emulator_path.parts[-3:])
        self.chain_path             = Path(self.data_path / "chains" / emul_path_suffix)
        
        # Load parameter names, labels, priors, and fiducial values from emulator config
        # Ensures correct order of parameters when predicting 
        self.emulator_param_names   = self.load_emulator_param_names(self.emulator_path)
        self.nparams                = len(self.emulator_param_names)

        self.HOD_param_names        = ["log10_ng", "log10M1", "sigma_logM", "kappa", "alpha"]
        self.cosmo_param_names      = ["N_eff", "alpha_s", "ns", "sigma8", "w0", "wa", "wb", "wc"]
        self.param_priors           = self.get_parameter_priors()
        
        self.load_plot_quantities()        




    def load_emulator_param_names(self, path):
        with open(path / "config.yaml", "r") as fp:
            config = yaml.safe_load(fp)
        return config["data"]["feature_columns"][:-1]

    def get_param_names_latex(self):
        HOD_param_labels = {
            "log10M1"       : r"$\log{M_1}$",
            "sigma_logM"    : r"$\sigma_{\log{M}}$",
            "kappa"         : r"$\kappa$",
            "alpha"         : r"$\alpha$",
            "log10_ng"      : r"$\log{n_g}$",
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
    

    def load_plot_quantities(self):
        param_labels_latex          = self.get_param_names_latex()
        fiducial_params             = self.get_fiducial_params() # Fiducial parameter values

        # Indices where cosmological parameters are found in emulator_param_names
        self.cosmo_indices          = [self.emulator_param_names.index(param) for param in self.cosmo_param_names]
        self.cosmo_labels           = [param_labels_latex[param] for param in self.cosmo_param_names] # Latex labels for cosmological parameters
        cosmo_fiducial_params       = [fiducial_params[i] for i in self.cosmo_indices] # Fiducial parameter values for cosmological parameters
        self.fiducial_params_cosmo  = {label: cosmo_fiducial_params[i] for i, label in enumerate(self.cosmo_labels)} # Fiducial parameter values for cosmological parameters
        cosmo_param_ranges          = [tuple(self.param_priors[i]) for i in self.cosmo_indices]
        self.prior_ranges_cosmo     = {label: cosmo_param_ranges[i] for i, label in enumerate(self.cosmo_labels)}



        # Indices where HOD parameters are found in emulator_param_names
        self.HOD_indices            = [self.emulator_param_names.index(param) for param in self.HOD_param_names]
        self.HOD_labels             = [param_labels_latex[param] for param in self.HOD_param_names] # Latex labels for HOD parameters
        HOD_fiducial_params         = [fiducial_params[i] for i in self.HOD_indices] # Fiducial parameter values for HOD parameters
        self.fiducial_params_HOD    = {label: HOD_fiducial_params[i] for i, label in enumerate(self.HOD_param_names)} # Fiducial parameter values for HOD parameters
        HOD_param_ranges            = [tuple(self.param_priors[i]) for i in self.HOD_indices]
        self.prior_ranges_HOD       = {label: HOD_param_ranges[i] for i, label in enumerate(self.HOD_labels)}



    def print_info(self, filename):
        chainfile = Path(self.chain_path / filename)
        fff = h5py.File(chainfile, "r")
        tau = fff["tau"][:]
        steps = fff['chain'].shape[0]

        tau_min = np.min(tau)
        tau_max = np.max(tau)

        print(f"# {filename} #")
        print(f"  - {tau_min =:10.2f} | {steps/tau_min =:4.2f}")
        print(f"  - {tau_max =:10.2f} | {steps/tau_max =:4.2f}")
        print(f"  - Chain shape:  {fff['chain'].shape}")
        print(f"  - Total chains: {fff['chain'].shape[0] * fff['chain'].shape[1]:.2e}")
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
        ):
        chainfile     = Path(self.chain_path / filename)
        if not chainfile.exists():
            raise FileNotFoundError(f"File {chainfile} not found. Run chain first.")

        samples = self.load_samples(
            chainfile, 
            burnin          = burnin, 
            thin            = thin, 
            burnin_factor   = burnin_factor, 
            thin_factor     = thin_factor
            )

        cosmo_samples = MCSamples(
            samples = samples[:, self.cosmo_indices], 
            names   = self.cosmo_labels, 
            labels  = self.cosmo_labels
            )
        g = plots.get_subplot_plotter()

        g.triangle_plot(
            cosmo_samples, 
            filled  = True,
            markers = self.fiducial_params_cosmo,
            )

        if show:
            # Add figure title
            # plt.suptitle(f"{filename}, {n_steps} steps", fontsize=16)
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
        g.export(figname, adir="figures")

    def plot_HOD(
            self,
            filename:       str,
            figname:        str  = None,
            burnin:         int  = None,
            thin:           int  = None,
            burnin_factor:  float = 10,
            thin_factor:    float = 5,
        ):
        chainfile     = Path(self.chain_path / filename)
        if not chainfile.exists():
            raise FileNotFoundError(f"File {chainfile} not found. Run chain first.")

        samples = self.load_samples(
            chainfile, 
            burnin          = burnin, 
            thin            = thin, 
            burnin_factor   = burnin_factor, 
            thin_factor     = thin_factor
            )

        HOD_samples = MCSamples(
            samples = samples[:, self.HOD_indices],
            names   = self.HOD_param_names,
            labels  = self.HOD_labels,
            )
        g = plots.get_subplot_plotter()

        g.triangle_plot(
            HOD_samples, 
            filled  = True,
            markers = self.fiducial_params_HOD,
            )

        if show:
            # Add figure title
            # plt.suptitle(f"{filename}, {n_steps} steps", fontsize=16)
            plt.show()
            return 
        
        if figname is None:
            # Set figname to cosmo_filename-stem.png
            figname = f"HOD_getdist_{filename.split('.')[0]}.png"

        output_file = Path(f"figures/{figname}")
        if output_file.exists():
            print(f"File {output_file} already exists. Adding _new to filename.")
            figname = f"{output_file.stem}_new{output_file.suffix}"
            output_file = Path(f"figures/{figname}")
        print(f"Saving {output_file} ...")
        g.export(figname, adir="figures")


    def plot_cosmo_double(
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
            burnin          = burnin, 
            thin            = thin, 
            burnin_factor   = burnin_factor, 
            thin_factor     = thin_factor
            )
        samples2 = self.load_samples(
            chainfile2, 
            burnin          = burnin, 
            thin            = thin, 
            burnin_factor   = burnin_factor, 
            thin_factor     = thin_factor
            )

        
        cosmo_samples1 = MCSamples(
            samples = samples1[:, self.cosmo_indices], 
            names   = self.cosmo_labels, 
            labels  = self.cosmo_labels, 
            label   = r"Varying $\mathcal{C}+\mathcal{G}$")
        cosmo_samples2 = MCSamples(
            samples = samples2[:, self.cosmo_indices], 
            names   = self.cosmo_labels, 
            labels  = self.cosmo_labels, 
            label   = r"Varying $\mathcal{C}$")

        g = plots.get_subplot_plotter()

        g.triangle_plot(
            [cosmo_samples1, cosmo_samples2], 
            filled          = True,
            markers         = self.fiducial_params_cosmo,
            contour_colors  = ["blue", "red"],
            contour_args    = [{"alpha": 1}, {"alpha": 0.75}],
            legend_loc      = "upper right",
            )

        if show:
            # Add figure title
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

    def plot_HOD_double(
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
            burnin          = burnin, 
            thin            = thin, 
            burnin_factor   = burnin_factor, 
            thin_factor     = thin_factor
            )
        samples2 = self.load_samples(
            chainfile2, 
            burnin          = burnin, 
            thin            = thin, 
            burnin_factor   = burnin_factor, 
            thin_factor     = thin_factor
            )
        if samples1.shape[1] == len(self.HOD_indices):
            HOD_indices1 = [HOD_idx - len(self.cosmo_indices) for HOD_idx in self.HOD_indices]
        else:
            HOD_indices1 = self.HOD_indices

        if samples2.shape[1] == len(self.HOD_indices):
            HOD_indices2 = [HOD_idx - len(self.cosmo_indices) for HOD_idx in self.HOD_indices]
        else:
            HOD_indices2 = self.HOD_indices

        
        HOD_samples1 = MCSamples(
            samples = samples1[:, HOD_indices1], 
            names   = self.HOD_param_names, 
            labels  = self.HOD_labels,
            label   = r"Varying $\mathcal{C}+\mathcal{G}$")
        HOD_samples2 = MCSamples(
            samples = samples2[:, HOD_indices2],
            names   = self.HOD_param_names, 
            labels  = self.HOD_labels, 
            label   = r"Varying $\mathcal{G}$")
        g = plots.get_subplot_plotter()

        g.triangle_plot(
            [HOD_samples1, HOD_samples2], 
            filled          = True,
            markers         = self.fiducial_params_HOD,
            contour_colors  = ["blue", "red"],
            contour_args    = [{"alpha": 1}, {"alpha": 0.75}],
            legend_loc      = "upper right",
            )

        if show:
            # Add figure title
            # plt.suptitle("Testing title")
            plt.show()
            return 
        
        if figname is None:
            # Set figname to HOD_filename-stem.png
            figname = f"HOD_compare-{filename1.split('.')[0]}-{filename2.split('.')[0]}.png"

        output_file = Path(f"figures/{figname}")
        if output_file.exists():
            input(f"  ! File {output_file} already exists. Press enter to overwrite ...")
            figname = f"{output_file.stem}_new{output_file.suffix}"
            output_file = Path(f"figures/{figname}")
        print(f"Saving {output_file} ...")
        g.export(figname, adir="figures")


global show 
show = False
L = Plot_MCMC()
# L.plot_cosmo("DE_4w_1e5.hdf5")
# L.plot_HOD("DE_4w_1e5.hdf5")
# L.plot_cosmo_double(filename1="DE_4w_1e5.hdf5", filename2="vary_cosmo_DE_4w_1e5.hdf5", )
# L.plot_HOD_double(  filename1="DE_4w_1e5.hdf5", filename2="vary_HOD_DE_4w_1e5.hdf5",   )

