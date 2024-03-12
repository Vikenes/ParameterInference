import numpy as np
import sys
import h5py
import matplotlib
import time 
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec

Nr = int(100)

file_xiS_sim = h5py.File(
    f"/mn/stornext/d8/data/chengzor/void/MGGLAMx100/GR/data/xi_vg_HOD_LOWZ_z0.25_Rcut25.11_voxel.hdf5", 
    'r',
)
s_bin_centre = file_xiS_sim['xi_vg_S'][f'box1']['s'][...]
xi0 = np.zeros([s_bin_centre.shape[0], Nr])
xi2 = np.zeros([s_bin_centre.shape[0], Nr])

_lst0 = []
_lst2 = []
t0 = time.time()
for jj in range(Nr):
    # for c000_ph000-ph024
    box_id = int(jj + 1) 
    xi0[:, jj] = file_xiS_sim['xi_vg_S'][f'box{box_id}']['xi_vg_S0'][...] # wp from s_z
    # xi2[:, jj] = file_xiS_sim['xi_vg_S'][f'box{box_id}']['xi_vg_S2'][...] #  



xi_all = np.zeros([s_bin_centre.shape[0] * 2, Nr])
xi_all[:s_bin_centre.shape[0]] = xi0
xi_all[s_bin_centre.shape[0]:s_bin_centre.shape[0]*2] = xi2


Cov_all = np.cov(xi_all)
R_all = np.corrcoef(xi_all)

print(f"Time: {time.time() - t0:.2f} s")  
# np.save(
#     f'cov_xiS02_GLAM',
#     Cov_all,
# )
# np.save(
#     f'coef_xiS02_GLAM',
#     R_all,
# )

fig = plt.figure(figsize=(5, 5))
gs = gridspec.GridSpec(1, 1,)
ax0 = plt.subplot(gs[0])

ax0.imshow(R_all, cmap='bwr',)
# plt.savefig(
    # "coef_GLAM.pdf",
    # bbox_inches="tight",
    # pad_inches=0.05,
# )

# plt.show()


