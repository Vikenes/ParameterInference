import numpy as np 
import h5py 
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
from scipy.integrate import simps
import pandas as pd 

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
        self.predictor = Predictor.from_path(f"{LIGHTING_LOGS_PATH}/version_{version}")

    def __call__(
        self,
        params,
    ):
        return self.predictor(np.array(params)).reshape(-1)
    



class Likelihood:
    def __init__(
        self,
        data_path = "/mn/stornext/d5/data/vetleav/HOD_AbacusData/covariance_data_fiducial",
        emulator_path = "emulator_data/vary_r/emulators/compare_scaling",
        emulator_version = 6
    ):
        self.data_path = Path(data_path)
        self.load_wp_data()
        self.load_fiducial_xi()
        
        self.load_covariance_data()

        self.emulator = xi_emulator_class(emulator_path, emulator_version)


    def load_covariance_data(self):
        """
        Load covariance matrix and its inverse
        Computed from the wp data loaded in "load_wp_data()"
        """
        self.cov_matrix        = np.load(self.data_path / "cov_wp_fiducial.npy")
        self.cov_matrix_inv    = np.linalg.inv(self.cov_matrix)
        
    def load_wp_data(self):
        """
        Load fiducial wp data
        computed from fiducial AbacusSummit simulation: c000_ph000-c000_ph024
        """
        WP = h5py.File(self.data_path / "wp_from_sz_fiducial_ng_fixed.hdf5", "r")
        r_perp_lst = []
        w_p_lst = []
        for ph in range(25):
            WP_sim = WP[f"AbacusSummit_base_c000_ph{str(ph).zfill(3)}"]
            r_perp_lst.append(WP_sim["r_perp"][:])
            w_p_lst.append(WP_sim["w_p"][:])

        WP.close()

        self.r_perp   = np.array(r_perp_lst)
        self.w_p_data = np.array(w_p_lst)

    def load_fiducial_xi(self):
        """
        Load fiducial xi data
        computed from fiducial AbacusSummit simulation: c000_ph000-c000_ph024
        """
        XI = h5py.File(self.data_path / "tpcf_r_fiducial_ng_fixed.hdf5", "r")
        r_lst = []
        xi_lst = []
        for ph in range(25):
            XI_sim = XI[f"AbacusSummit_base_c000_ph{str(ph).zfill(3)}"]
            r_lst.append(XI_sim["r"][:])
            xi_lst.append(XI_sim["xi"][:])

        XI.close()

        self.r_data     = np.array(r_lst)
        self.xi_data    = np.array(xi_lst)

        # plt.plot(self.r[0], self.xi_data[0])
        # plt.xscale("log")
        # plt.yscale("log")
        # plt.show()

    def xi_to_wp(self, r, xi):
        """
        Convert xi to wp
        """
        



    def __call__(
        self,
        params,
    ):
        r = self.r_data[0]
        Nr = len(r)
        emul_input = np.column_stack((
            np.vstack(
                [params] * Nr
                )
            ,r
            ))
        
        xi = self.emulator(emul_input)
        xiR_func = IUS(
            r, xi
        )

        r_perp = self.r_perp[0].reshape(-1,1)
        r_para = np.linspace(0, 100, int(1000)).reshape(1,-1)

        w_p_theory = 2.0 * simps(
            xiR_func(
                np.sqrt(r_perp**2 + r_para**2)
            ),
            r_para,
            axis=-1
        )

        # plt.plot(self.r_perp[0], self.r_perp[0] * w_p_theory, label="theory")
        # plt.plot(self.r_perp[0], self.r_perp[0] * self.w_p_data[0], label="data")
        # plt.xscale("log")
        # plt.yscale("log")
        # plt.legend()
        # plt.show()

        

        # w_p_theory = self.xi_to_wp(xi)
        delta = self.w_p_data[0] - w_p_theory# ).flatten()
        # print(f"{w_p_theory.shape=}")
        # print(f"{self.w_p_data.shape=}")    
        # exit()

        # print(f"{delta.shape=}")
        # print(f"{self.cov_matrix_inv.shape=}")
        mp = delta.T @ self.cov_matrix_inv @ delta
        print(mp)

        exit()
        lnprob = -0.5 * np.einsum('i,ij,j', delta, self.cov_matrix_inv, delta) 
        return lnprob


L = Likelihood()
# L.load_fiducial_xi()
param_names = ['wb', 'wc', 'sigma8', 'ns', 'alpha_s', 'N_eff', 'w0', 'wa', 'sigma_logM', 'alpha', 'kappa', 'log10M1', 'log10Mmin']
HOD_param_names = ['sigma_logM', 'alpha', 'kappa', 'log10M1', 'log10Mmin']
TPCF_data = h5py.File(D5_PATH+"TPCF_test_ng_fixed.hdf5", "r")
c0 = TPCF_data["AbacusSummit_base_c000_ph000"]["node0"]
HOD_params = pd.read_csv(f"{D13_PATH}/AbacusSummit_base_c000_ph000/HOD_parameters/HOD_parameters_fiducial_ng_fixed.csv")
# print(HOD_params)
# print(c0.attrs.keys())
np.random.seed(123)
params = []
for param in param_names:
    if param in HOD_param_names:
        pval = HOD_params[param].values[0]
    else:
        pval = c0.attrs[param]
    params.append(pval + np.random.normal(0, 1) * 1e-4)

# params = [0.02282, 0.12, 0.808181, 0.9649, 0.0, 3.0328, -1.0, 0.0, 0.7184170121274857, 0.9941066904333122, 0.538306161066293, 13.87001633956449, 13.604313483674156]

lnprob = L(params)

print(lnprob)