# Evaluation of results
# B-Splines from sigmoids: sigma that provides the best PSNR
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

# Load the results
mat = sio.loadmat("/home/atk23/wire/bspline_results/sigmoid_k/metrics.mat")
variables = {}
for key in mat.keys():
    if not key.startswith('__'):
        variables[key] = mat[key]
sigma = list(variables.keys())
print(np.shape(sigma))
max_psnr = 0
for types in sigma[0:10]:        
    values = mat[types][0, 0]
    psnr = values['Best PSNR'][0, 0]
    if psnr > max_psnr:
        max_psnr = psnr
        sigma0 = values['Sigma0'][0, 0]

print(f"Best PSNR: {max_psnr} at sigma0 = {sigma0}")

# print(mat.keys())
# max_psnr = np.max(mat["Best PSNR"])
# max_psnr_idx = np.argmax(mat["Best PSNR"])
# print(f"Best PSNR: {max_psnr} at k = {mat['k'][max_psnr_idx]}")
