import os
import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
sns.set_palette("bright")

# linestyles = ['-', '--', '-.', ':', '-', '--', '-.']


# file = {
#     'BSpline':
#     'multiscale_results/representation/mountain/Bspline_s9_5_LR8e3_E2000_T3e7_1',
#     'MultiStageNet':
#     'multiscale_results/representation/mountain/Mscale2_ST16_3_LR8e3_E4000_1',
#     'MultiHierNet':
#     'multiscale_results/representation/mountain/MscaleHier_ST4_LR8e3_E4000_T3e7_1',
#     'Wire':
#     'multiscale_results/representation/mountain/WIRE_s8_o7_LR1e2_E2000_T3e7_1',
#     'MultiLayerNet':
#     'multiscale_results/representation/mountain/MscaleHL_s1o9_ST4_SHF384_LR8e3_E4000_T3e7_1',
#     'Posenc':
#     'multiscale_results/representation/mountain/Posenc_s8_o7_LR8e4_E2000_T3e7_SNR2_2',
#     'SIREN':
#     'multiscale_results/representation/mountain/SIREN_s30_o7_LR8e3_E2000_T3e7_SNR2_2'
# }

file = {'MultiLayerNet': 'multiscale_results/sisr/Face/DS_4/MscaleHL_s1o9_ST4_SHF384_L1e3_E2000_1',
        'MultiLayerNet1': 'multiscale_results/sisr/Face/DS_4/MscaleHL_s1o9_ST1o15_SHF384_L1e3_E2000_1',}

plt.figure(figsize=(8, 6))
epochs = np.arange(500)
for key, value in file.items():
    path = os.path.join(value, 'info.mat')
    mat = io.loadmat(path)
    for k in mat.keys():
        if '__' not in k:
            # print(k)
            mse = mat[k]['mse_array'][0][0]
            mse = mse[:, 0:500]
            mse = mse.flatten()
            print(mse.shape)
            plt.plot(epochs, mse, label=key, linewidth=2)
            plt.legend()

plt.xlabel('Epoch', fontsize=10)
plt.ylabel('Mean Squared Error', fontsize=10)
plt.legend(fontsize=10, loc='upper right', bbox_to_anchor=(0.99, 0.99), 
           ncol=1, framealpha=0.8)

# Set y-axis to logarithmic scale and start from a small positive value
plt.yscale('log')
plt.ylim(1e-5, 0.4)  # Adjust the upper limit as needed to fit your data

# Set x-axis to start from 0
plt.xlim(0, 500)

plt.tick_params(axis='both', which='major', labelsize=10)
plt.grid(True, which="both", ls="-", alpha=0.2)

# Add a horizontal line at y=0 to emphasize the bottom of the plot
plt.axhline(y=1e-5, color='black', linewidth=0.5, alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(file['MultiLayerNet'],'loss_curve.png'), bbox_inches='tight')
# if k == 'mse_array':
#     plt.plot(v[0], label=key)
