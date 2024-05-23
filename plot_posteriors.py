import matplotlib.backends
import matplotlib.backends.backend_agg
import numpy as np 
import h5py 
from pathlib import Path
import pandas as pd 
import yaml 
from getdist import plots, MCSamples
import matplotlib
import subprocess
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams.update({
    'font.size': 20, 
    # 'axes.labelsize': 20,
    })
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
# matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{physics}'
params = {'xtick.top': True, 'ytick.right': True, 'xtick.direction': 'in', 'ytick.direction': 'in'}
plt.rcParams.update(params)
D13_PATH = "/mn/stornext/d13/euclid_nobackup/halo/AbacusSummit/emulation_files/"


class Plot_MCMC:
    def __init__(
        self,
        data_path           = "data/inference_data",
        emulator_path       = "data/emulator_data/sliced_r/emulators/batch_size_3040",
        emulator_version    = 2,
    ):
        self.data_path              = Path(data_path)
        self.emulator_path          = Path(f"{emulator_path}/version_{emulator_version}")

        emul_path_suffix            = "_".join(self.emulator_path.parts[-3:])
        self.chain_path             = Path(self.data_path / "chains" / emul_path_suffix)
        
        # Load parameter names, labels, priors, and fiducial values from emulator config
        # Ensures correct order of parameters when predicting 
        self.emulator_param_names   = self.load_emulator_param_names(self.emulator_path)

        self.HOD_param_names        = ["log10_ng", "log10M1", "sigma_logM", "alpha", "kappa"]
        self.cosmo_param_names      = ["wb", "wc", "sigma8", "w0", "wa", "ns", "alpha_s",  "N_eff"]
        self.nparams                = len(self.emulator_param_names)
        self.nparams_HOD            = len(self.HOD_param_names)
        self.nparams_cosmo          = len(self.cosmo_param_names)
        
        self.load_parameter_priors()
        self.load_param_names_latex()
        self.load_fiducial_params()
        self.load_plot_quantities()   





    def load_emulator_param_names(self, path):
        with open(path / "config.yaml", "r") as fp:
            config = yaml.safe_load(fp)
        return config["data"]["feature_columns"][:-1]

    def load_param_names_latex(self):
        self.HOD_latex_labels_dict = {
            "log10M1"       : r"$\log{M_1}$",
            "sigma_logM"    : r"$\sigma_{\log{M}}$",
            "kappa"         : r"$\kappa$",
            "alpha"         : r"$\alpha$",
            "log10_ng"      : r"$\log{n_g}$",
        }
        self.cosmo_latex_labels_dict = {
            "N_eff"     : r"$N_\mathrm{eff}$",
            "alpha_s"   : r"$\mathrm{d}n_s/\mathrm{d}\ln{k}$",
            "ns"        : r"$n_s$",
            "sigma8"    : r"$\sigma_8$",
            "w0"        : r"$w_0$",
            "wa"        : r"$w_a$",
            "wb"        : r"$\omega_b$",
            "wc"        : r"$\omega_\mathrm{cdm}$",
        }

        return None
    
    def load_fiducial_params(self):
        fiducial_HOD_params     = pd.read_csv(f"{D13_PATH}/fiducial_data/HOD_parameters_fiducial.csv")
        fiducial_cosmo_params   = pd.read_csv(f"{D13_PATH}/fiducial_data/cosmological_parameters.dat", sep=" ")
        self.fiducial_cosmo_params_dict = fiducial_cosmo_params.iloc[0].to_dict()
        self.fiducial_HOD_params_dict   = fiducial_HOD_params.iloc[0].to_dict()
        return None

    def load_parameter_priors(self):
        config          = yaml.safe_load(open(f"{self.data_path}/priors_config.yaml"))
        self.param_priors    = {}
        for param_name in self.emulator_param_names:
            self.param_priors[param_name] = config[param_name]
        return None     

    def load_plot_quantities(self):
        # Indices where cosmological parameters are found in emulator_param_names
        self.cosmo_indices          = [self.emulator_param_names.index(param) for param in self.cosmo_param_names]
        self.cosmo_labels           = [self.cosmo_latex_labels_dict[param] for param in self.cosmo_param_names] # Latex labels for cosmological parameters
        self.fiducial_cosmo         = {key: self.fiducial_cosmo_params_dict[key] for key in self.cosmo_param_names}
        self.cosmo_param_ranges     = {key: self.param_priors[key] for key in self.cosmo_param_names}

        # Indices where HOD parameters are found in emulator_param_names
        self.HOD_indices            = [self.emulator_param_names.index(param) for param in self.HOD_param_names]
        self.HOD_labels             = [self.HOD_latex_labels_dict[param] for param in self.HOD_param_names] # Latex labels for HOD parameters
        self.fiducial_HOD           = {key: self.fiducial_HOD_params_dict[key] for key in self.HOD_param_names}
        self.HOD_param_ranges       = {key: self.param_priors[key] for key in self.HOD_param_names}

    def load_fixed_and_varying_params(self, fixed_cosmo_params):
        varying_emulator_param_names     = [param for param in self.emulator_param_names if param not in fixed_cosmo_params]
        self.varying_cosmo_param_names   = [param for param in self.cosmo_param_names if param not in fixed_cosmo_params]
        self.fixed_cosmo_param_names     = [param for param in self.emulator_param_names if param in fixed_cosmo_params or param in self.HOD_param_names]
        self.varying_cosmo_indices_full  = [self.emulator_param_names.index(param) for param in self.varying_cosmo_param_names]
        self.varying_cosmo_indices_short = [varying_emulator_param_names.index(param) for param in self.varying_cosmo_param_names]
        self.fiducial_cosmo_varying      = {key: self.fiducial_cosmo_params_dict[key] for key in self.varying_cosmo_param_names}
        self.varying_cosmo_labels        = [self.cosmo_latex_labels_dict[param] for param in self.varying_cosmo_param_names]
       
    def get_plot_label_fixed_cosmo(self, fixed_cosmo_params):
        label_str = r"\{"
        N_fixed_params = len(fixed_cosmo_params)
        for ii, param in enumerate(fixed_cosmo_params):
            label_str += self.cosmo_latex_labels_dict[param]
            if ii < N_fixed_params - 1:
                label_str += ", "
        label_str += r"\}"
        return fr"Varying $\mathcal{{C}}$, fixed $\mathcal{{G}}$+{label_str}"

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
        fff.close()
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

        chain       = fff["chain"]   # Access chain
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
            to_thesis:      bool  = False,
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
            names   = self.cosmo_param_names, 
            labels  = self.cosmo_labels,  
            label   = r"Varying $\mathcal{C}+\mathcal{G}$")
        cosmo_samples2 = MCSamples(
            samples = samples2[:, self.cosmo_indices], 
            names   = self.cosmo_param_names, 
            labels  = self.cosmo_labels,  
            label   = r"Varying $\mathcal{C}$, fixed $\mathcal{G}$")

        g = plots.get_subplot_plotter(scaling=False)
        g.settings.axes_labelsize       = 24
        g.settings.axes_fontsize        = 16
        g.settings.legend_fontsize      = 24
        g.settings.subplot_size_ratio   = 1
        g.settings.axis_tick_max_labels = 3
        g.settings.axis_marker_lw       = 0.8
        g.settings.axis_marker_color    = "black"


        g.triangle_plot(
            [cosmo_samples1, cosmo_samples2], 
            filled          = True,
            markers         = self.fiducial_cosmo, 
            contour_colors  = ["blue", "red"],
            contour_args    = [{"alpha": 1}, {"alpha": 0.7}],
            legend_loc      = "upper right",
            )
        
        # for row in range(len(g.subplots)):
        #     for col in range(len(g.subplots)):
        #         ax = g.subplots[row, col] 
        #         if row == len(g.subplots)-1:  # Bottom row
        #             ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=3))
        #             ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        #             ax.tick_params(axis="x", which='minor', bottom=True)#, top=False)

        #         if col == 0:  # First column
        #             ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=3))
        #             ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        #             ax.tick_params(axis="y", which='minor', left=True)#, top=False)

        # for ax in g.subplots.flatten():
        #     try:
        #         ax.label_outer(remove_inner_ticks=True)
        #     except:
        #         pass 
        
        if show and not to_thesis:
            plt.show()
            return 
        
        if figname is None:
            # Set figname to cosmo_filename-stem.png
            figname = f"cosmo_compare-{filename1.split('.')[0]}-{filename2.split('.')[0]}.png"

        if not type(figname) is str:
            raise ValueError("figname must be a string.")
        if not to_thesis:
            g.export(figname, adir="figures")

        else:
            figname_stem = figname.split('.')[0] if "." in figname else figname
            output_file_png = f"{figname_stem}.png"
            output_file_pdf = f"{figname_stem}.pdf"

            if Path(f"figures/thesis_figures/{output_file_png}").exists() or Path(f"figures/thesis_figures/{output_file_pdf}").exists():
                _input = input(f"File {output_file_png} already exists. Overwrite? (y/n): ")
                if _input.lower() != "y":
                    print("Exiting without saving.")
                    return
            print(f"Saving {output_file_pdf} ...")
            g.export(output_file_pdf, adir="figures/thesis_figures")
            print(f"Saving {output_file_png} ...")
            g.export(output_file_png, adir="figures/thesis_figures")
            _input = input("Push to git? (y/n): ")
            if _input.lower() == "y":
                try:
                    subprocess.check_call(["git", "-C", "figures/thesis_figures", "pull"])
                    subprocess.check_call(["git", "-C", "figures/thesis_figures", "add", f"{figname_stem}*"])
                    subprocess.check_call(["git", "-C", "figures/thesis_figures", "commit", "-m", f"new figs {figname_stem}"])
                    subprocess.check_call(["git", "-C", "figures/thesis_figures", "push"])
                except subprocess.CalledProcessError as e:
                    print(f"Error: {e}")

    def plot_cosmo_double_no_wb(
            self,
            filename1:      str,
            filename2:      str,
            figname:        str  = None,
            burnin:         int  = None,
            thin:           int  = None,
            burnin_factor:  float = 10,
            thin_factor:    float = 5,
            to_thesis:      bool  = False,
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
            samples = samples1[:, self.cosmo_indices[1:]], 
            names   = self.cosmo_param_names[1:], 
            labels  = self.cosmo_labels[1:],  
            label   = r"Varying $\mathcal{C}+\mathcal{G}$")
        cosmo_samples2 = MCSamples(
            samples = samples2[:, self.cosmo_indices[1:]], 
            names   = self.cosmo_param_names[1:], 
            labels  = self.cosmo_labels[1:],  
            label   = r"Varying $\mathcal{C}$, fixed $\mathcal{G}$")

        g = plots.get_subplot_plotter(scaling=False)
        g.settings.axes_labelsize       = 24
        g.settings.axes_fontsize        = 16
        g.settings.legend_fontsize      = 24
        g.settings.subplot_size_ratio   = 1
        g.settings.axis_tick_max_labels = 3
        g.settings.axis_marker_lw       = 0.8
        g.settings.axis_marker_color    = "black"


        g.triangle_plot(
            [cosmo_samples1, cosmo_samples2], 
            filled          = True,
            markers         = self.fiducial_cosmo, 
            contour_colors  = ["blue", "red"],
            contour_args    = [{"alpha": 1}, {"alpha": 0.7}],
            legend_loc      = "upper right",
            )
        
        if show and not to_thesis:
            # g.export("test.pdf", adir="figures")
            plt.show()
            return 
        
        if figname is None:
            # Set figname to cosmo_filename-stem.png
            figname = f"cosmo_compare-{filename1.split('.')[0]}-{filename2.split('.')[0]}.png"

        if not type(figname) is str:
            raise ValueError("figname must be a string.")
        if not to_thesis:
            g.export(figname, adir="figures")

        else:
            figname_stem = figname.split('.')[0] if "." in figname else figname
            output_file_png = f"{figname_stem}.png"
            output_file_pdf = f"{figname_stem}.pdf"

            if Path(f"figures/thesis_figures/{output_file_png}").exists() or Path(f"figures/thesis_figures/{output_file_pdf}").exists():
                _input = input(f"File {output_file_png} already exists. Overwrite? (y/n): ")
                if _input.lower() != "y":
                    print("Exiting without saving.")
                    return
            print(f"Saving {output_file_pdf} ...")
            g.export(output_file_pdf, adir="figures/thesis_figures")
            print(f"Saving {output_file_png} ...")
            g.export(output_file_png, adir="figures/thesis_figures")
            _input = input("Push to git? (y/n): ")
            if _input.lower() == "y":
                try:
                    subprocess.check_call(["git", "-C", "figures/thesis_figures", "pull"])
                    subprocess.check_call(["git", "-C", "figures/thesis_figures", "add", f"{figname_stem}*"])
                    subprocess.check_call(["git", "-C", "figures/thesis_figures", "commit", "-m", f"new figs {figname_stem}"])
                    subprocess.check_call(["git", "-C", "figures/thesis_figures", "push"])
                except subprocess.CalledProcessError as e:
                    print(f"Error: {e}")

    def plot_cosmo_double_fixed_params(
            self,
            filename1:      str,
            filename2:      str,
            fixed_params:   list,
            include_wb:     bool,
            figname:        str  = None,
            burnin:         int  = None,
            thin:           int  = None,
            burnin_factor:  float = 10,
            thin_factor:    float = 5,
            to_thesis:      bool  = False,
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
        
        self.load_fixed_and_varying_params(fixed_cosmo_params=fixed_params)
        assert len(self.varying_cosmo_param_names) == samples2.shape[-1], "Number of varying parameters does not match number of columns in samples2"
        
        if include_wb:
            varying_cosmo_indices_full  = self.varying_cosmo_indices_full
            varying_cosmo_param_names   = self.varying_cosmo_param_names
            varying_cosmo_labels        = self.varying_cosmo_labels
            varying_cosmo_indices_short = self.varying_cosmo_indices_short

        else:
            varying_cosmo_indices_full  = self.varying_cosmo_indices_full[1:]
            varying_cosmo_param_names   = self.varying_cosmo_param_names[1:]
            varying_cosmo_labels        = self.varying_cosmo_labels[1:]
            varying_cosmo_indices_short = self.varying_cosmo_indices_short[1:]


        cosmo_samples1 = MCSamples(
            samples = samples1[:, varying_cosmo_indices_full], 
            names   = varying_cosmo_param_names, 
            labels  = varying_cosmo_labels, 
            label   = r"Varying $\mathcal{C}$, fixed $\mathcal{G}$"
            )
        cosmo_samples2 = MCSamples(
            samples = samples2[:, varying_cosmo_indices_short], 
            names   = varying_cosmo_param_names, 
            labels  = varying_cosmo_labels, 
            label   = self.get_plot_label_fixed_cosmo(fixed_params)
            )

        g = plots.get_subplot_plotter(scaling=False)
        g.settings.axes_labelsize       = 22
        g.settings.axes_fontsize        = 15
        g.settings.legend_fontsize      = 20
        g.settings.axis_tick_max_labels = 3
        g.settings.subplot_size_ratio   = 1
        g.settings.axis_marker_color    = "black"
        g.settings.axis_marker_lw       = 0.8
        plt.rcParams.update({'axes.labelpad': 10})

        g.triangle_plot(
            [cosmo_samples1, cosmo_samples2], 
            filled          = True,
            markers         = self.fiducial_cosmo_varying,
            contour_colors  = ["blue", "red"],
            contour_args    = [{"alpha": 1}, {"alpha": 0.75}],
            legend_loc      = "upper right",
            )

        if show and not to_thesis:
            plt.show()
            return 
        
        if figname is None:
            # Set figname to cosmo_filename-stem.png
            figname = f"cosmo_compare-{filename1.split('.')[0]}-{filename2.split('.')[0]}.png"

        if not type(figname) is str:
            raise ValueError("figname must be a string.")
        if not to_thesis:
            g.export(figname, adir="figures")

        else:
            figname_stem = figname.split('.')[0] if "." in figname else figname
            output_file_png = f"{figname_stem}.png"
            output_file_pdf = f"{figname_stem}.pdf"

            if Path(f"figures/thesis_figures/{output_file_png}").exists() or Path(f"figures/thesis_figures/{output_file_pdf}").exists():
                _input = input(f"File {output_file_png} already exists. Overwrite? (y/n): ")
                if _input.lower() != "y":
                    print("Exiting without saving.")
                    return
            print(f"Saving {output_file_pdf} ...")
            g.export(output_file_pdf, adir="figures/thesis_figures")
            print(f"Saving {output_file_png} ...")
            g.export(output_file_png, adir="figures/thesis_figures")
            _input = input("Push to git? (y/n): ")
            if _input.lower() == "y":
                try:
                    subprocess.check_call(["git", "-C", "figures/thesis_figures", "pull"])
                    subprocess.check_call(["git", "-C", "figures/thesis_figures", "add", f"{figname_stem}*"])
                    subprocess.check_call(["git", "-C", "figures/thesis_figures", "commit", "-m", f"new figs {figname_stem}"])
                    subprocess.check_call(["git", "-C", "figures/thesis_figures", "push"])
                except subprocess.CalledProcessError as e:
                    print(f"Error: {e}")
    
    def plot_single_varying_param_double(
            self, 
            filename1:      str,
            filename2:      str,
            vary_param1:    str,
            vary_param2:    str,
            burnin_factor:  int  = 5,
            thin_factor:    int  = 1,
            to_thesis:       bool = False,
    ):
        chainfile1     = Path(self.chain_path / filename1)
        chainfile2     = Path(self.chain_path / filename2)

        fff1 = h5py.File(chainfile1, "r")
        fff2 = h5py.File(chainfile2, "r")
        samples1 = fff1["chain"][:]
        samples2 = fff2["chain"][:]
        samples1 = samples1.reshape(samples1.shape[0] * samples1.shape[1])[1000 :: 5]
        samples2 = samples2.reshape(samples2.shape[0] * samples2.shape[1])[1000 :: 5]
        
        cosmo_samples = MCSamples(
            samples = [samples1, samples2], 
            names   = [vary_param1, vary_param2], 
            labels  = [self.cosmo_latex_labels_dict[vary_param1], self.cosmo_latex_labels_dict[vary_param2]], 
            )
       
        g = plots.get_subplot_plotter(5)
        # g.settings.axis_marker_lw = 1
        # g.settings.axis_marker_ls = "solid"
        g.settings.axes_labelsize       = 24
        g.settings.axes_fontsize        = 16
        g.settings.legend_fontsize      = 24
        g.settings.subplot_size_ratio   = 1
        g.settings.axis_tick_max_labels = 3
        g.settings.axis_marker_lw       = 0.8
        g.settings.axis_marker_color    = "black"
        g.plots_1d(
            cosmo_samples, 
            [vary_param1, vary_param2], 
            markers=[self.fiducial_cosmo_params_dict[vary_param1], self.fiducial_cosmo_params_dict[vary_param2]],
            nx=2
            )
        
        
        # Set xlims
        for i, ax in enumerate(g.subplots.flatten()):
            if i == 0:
                lim = self.param_priors[vary_param1]
            else:
                lim = self.param_priors[vary_param2]
            ax.set_xlim(lim)
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5, min_n_ticks=5))#, min_n_tcks=5, prune='both'))

        if show and not to_thesis:
            plt.show()
            return

        figname_stem = "vary_ns_and_alpha_s_only"
        output_file_png = f"{figname_stem}.png"
        output_file_pdf = f"{figname_stem}.pdf"

        if Path(f"figures/thesis_figures/{output_file_png}").exists() or Path(f"figures/thesis_figures/{output_file_pdf}").exists():
            _input = input(f"File {output_file_png} already exists. Overwrite? (y/n): ")
            if _input.lower() != "y":
                print("Exiting without saving.")
                return
        print(f"Saving {output_file_pdf} ...")
        g.export(output_file_pdf, adir="figures/thesis_figures")
        print(f"Saving {output_file_png} ...")
        g.export(output_file_png, adir="figures/thesis_figures")



    def plot_HOD_double(
            self,
            filename1:      str,
            filename2:      str,
            figname:        str  = None,
            burnin:         int  = None,
            thin:           int  = None,
            burnin_factor:  float = 10,
            thin_factor:    float = 5,
            to_thesis:      bool  = False,
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
            label   = r"Varying $\mathcal{G}$, fixed $\mathcal{C}$")
        
        g = plots.get_subplot_plotter(scaling=False)
        g.settings.axes_labelsize = 22
        g.settings.axes_fontsize = 18
        g.settings.legend_fontsize = 20
        g.settings.subplot_size_ratio = 1
        g.settings.axis_marker_lw       = 0.8
        g.settings.axis_marker_color    = "black"
        
        g.triangle_plot(
            [HOD_samples1, HOD_samples2], 
            filled          = True,
            markers         = self.fiducial_HOD,
            contour_colors  = ["blue", "red"],
            contour_args    = [{"alpha": 1}, {"alpha": 0.75}],
            legend_loc      = "upper right",
            )

        if show and not to_thesis:
            plt.show()
            return 
        
        if figname is None:
            # Set figname to HOD_filename-stem.png
            figname = f"HOD_compare-{filename1.split('.')[0]}-{filename2.split('.')[0]}.png"
        
        if not type(figname) is str:
            raise ValueError("figname must be a string.")
        if not to_thesis:
            g.export(figname, adir="figures")

        else:
            figname_stem = figname.split('.')[0] if "." in figname else figname
            output_file_png = f"{figname_stem}.png"
            output_file_pdf = f"{figname_stem}.pdf"
            if Path(f"figures/thesis_figures/{output_file_png}").exists() or Path(f"figures/thesis_figures/{output_file_pdf}").exists():
                _input = input(f"File {output_file_png} already exists. Overwrite? (y/n): ")
                if _input.lower() != "y":
                    print("Exiting without saving.")
                    return
            print(f"Saving {output_file_png} ...")
            g.export(output_file_png, adir="figures/thesis_figures")
            print(f"Saving {output_file_pdf} ...")
            g.export(output_file_pdf, adir="figures/thesis_figures")
            _input = input("Push to git? (y/n): ")
            if _input.lower() == "y":
                try:
                    subprocess.check_call(["git", "-C", "figures/thesis_figures", "pull"])
                    subprocess.check_call(["git", "-C", "figures/thesis_figures", "add", f"{figname_stem}*"])
                    subprocess.check_call(["git", "-C", "figures/thesis_figures", "commit", "-m", f"new figs {figname_stem}"])
                    subprocess.check_call(["git", "-C", "figures/thesis_figures", "push"])
                except subprocess.CalledProcessError as e:
                    print(f"Error: {e}")

    def plot_HOD_double_no_kappa(
            self,
            filename1:      str,
            filename2:      str,
            figname:        str  = None,
            burnin:         int  = None,
            thin:           int  = None,
            burnin_factor:  float = 10,
            thin_factor:    float = 5,
            to_thesis:      bool  = False,
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
        HOD_indices1 = HOD_indices1[:-1]
        HOD_indices2 = HOD_indices2[:-1]

        
        HOD_samples1 = MCSamples(
            samples = samples1[:, HOD_indices1], 
            names   = self.HOD_param_names[:-1], 
            labels  = self.HOD_labels[:-1],
            label   = r"Varying $\mathcal{C}+\mathcal{G}$")
        HOD_samples2 = MCSamples(
            samples = samples2[:, HOD_indices2],
            names   = self.HOD_param_names[:-1], 
            labels  = self.HOD_labels[:-1], 
            label   = r"Varying $\mathcal{G}$, fixed $\mathcal{C}$")
        
        g = plots.get_subplot_plotter(scaling=False)
        g.settings.axes_labelsize = 22
        g.settings.axes_fontsize = 18
        g.settings.legend_fontsize = 20
        g.settings.subplot_size_ratio = 1
        g.settings.axis_marker_lw       = 0.8
        g.settings.axis_marker_color    = "black"

        
        g.triangle_plot(
            [HOD_samples1, HOD_samples2], 
            filled          = True,
            markers         = self.fiducial_HOD,
            contour_colors  = ["blue", "red"],
            contour_args    = [{"alpha": 1}, {"alpha": 0.75}],
            legend_loc      = "upper right",
            )

        if show and not to_thesis:
            plt.show()
            return 
        
        if figname is None:
            # Set figname to HOD_filename-stem.png
            figname = f"HOD_compare-{filename1.split('.')[0]}-{filename2.split('.')[0]}.png"
        
        if not type(figname) is str:
            raise ValueError("figname must be a string.")
        if not to_thesis:
            g.export(figname, adir="figures")

        else:
            figname_stem = figname.split('.')[0] if "." in figname else figname
            output_file_png = f"{figname_stem}.png"
            output_file_pdf = f"{figname_stem}.pdf"
            if Path(f"figures/thesis_figures/{output_file_png}").exists() or Path(f"figures/thesis_figures/{output_file_pdf}").exists():
                _input = input(f"File {output_file_png} already exists. Overwrite? (y/n): ")
                if _input.lower() != "y":
                    print("Exiting without saving.")
                    return
            print(f"Saving {output_file_png} ...")
            g.export(output_file_png, adir="figures/thesis_figures")
            print(f"Saving {output_file_pdf} ...")
            g.export(output_file_pdf, adir="figures/thesis_figures")
            _input = input("Push to git? (y/n): ")
            if _input.lower() == "y":
                try:
                    subprocess.check_call(["git", "-C", "figures/thesis_figures", "pull"])
                    subprocess.check_call(["git", "-C", "figures/thesis_figures", "add", f"{figname_stem}*"])
                    subprocess.check_call(["git", "-C", "figures/thesis_figures", "commit", "-m", f"new figs {figname_stem}"])
                    subprocess.check_call(["git", "-C", "figures/thesis_figures", "push"])
                except subprocess.CalledProcessError as e:
                    print(f"Error: {e}")
   

global show 
show = True
# show = False

L = Plot_MCMC()

# L.plot_single_varying_param(
#     filename="vary_ns_DE_4w_1e4.hdf5",
#     vary_param="ns",
# )

L.plot_single_varying_param_double(
    filename1="vary_ns_DE_4w_1e4.hdf5",
    filename2="vary_alpha_s_DE_4w_1e4.hdf5",
    vary_param1="ns",
    vary_param2="alpha_s",
    to_thesis=True
)
# L.plot_cosmo_double(
#     filename1="DE_4w_1e5_r0.1_70.hdf5", 
#     filename2="vary_cosmo_DE_4w_1e5_r0.1_70.hdf5", 
#     figname="MCMC_cosmo_posteriors_full_r_lim.pdf", 
#     to_thesis=False
#     )


#============================
# Finished
#============================
# L.plot_cosmo_double_no_wb(
#     filename1="DE_8w_2e5.hdf5",
#     filename2="vary_cosmo_DE_8w_2e5.hdf5",
#     figname="MCMC_cosmo_posteriors_no_wb_8w_2e5steps.pdf",
#     to_thesis=True
#     )
# L.plot_HOD_double_no_kappa(
#     filename1="DE_8w_2e5.hdf5",
#     filename2="vary_HOD_DE_8w_2e5.hdf5",
#     figname="MCMC_HOD_posteriors_no_kappa_8w_2e5steps.pdf",
#     to_thesis=True
# )

# L.plot_cosmo_double_fixed_params(
#     filename1="vary_cosmo_DE_8w_2e5.hdf5", 
#     filename2="vary_cosmo_EoS_fixed_DE_4w_1e5.hdf5", 
#     fixed_params=["wb", "w0", "wa"],
#     figname="MCMC_cosmo_posteriors_no_wb_EoS_fixed_4w_1e5.pdf", 
#     include_wb=True,
#     to_thesis=True
# )
# L.plot_cosmo_double_fixed_params(
#     filename1="vary_cosmo_DE_8w_2e5.hdf5", 
#     filename2="vary_cosmo_spectral_index_fixed_DE_4w_1e5.hdf5", 
#     fixed_params=["wb", "ns", "alpha_s"],
#     figname="MCMC_cosmo_posteriors_no_wb_spectral_index_fixed_4w_1e5.pdf", 
#     include_wb=True,
#     to_thesis=True
# )
#### FULL TRIANGLES
# L.plot_cosmo_double(
#     filename1="DE_8w_2e5.hdf5", 
#     filename2="vary_cosmo_DE_8w_2e5.hdf5", 
#     figname="MCMC_cosmo_posteriors_full.pdf", 
#     to_thesis=True
#     )
# L.plot_HOD_double(
#     filename1="DE_8w_2e5.hdf5", 
#     filename2="vary_HOD_DE_8w_2e5.hdf5",
#     figname="MCMC_HOD_posteriors_full.pdf", 
#     to_thesis=True
#     )

# =================
# old with w0,wa,alpha_s
# =================
# L.plot_cosmo_double_fixed_params(
#     filename1="vary_cosmo_DE_4w_1e5.hdf5", 
#     filename2="vary_cosmo_wa_alphas_w0_fixed_DE_4w_1e5.hdf5", 
#     fixed_params=["w0", "wa", "alpha_s"],
#     figname="MCMC_cosmo_posteriors_wa_alphas_w0_fixed.pdf", 
#     to_thesis=True
# )