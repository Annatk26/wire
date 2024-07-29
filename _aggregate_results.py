import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

files_metrics = {
    'Wire': "multiscale_results/sisr/WIRE_s9_Ds8_o8_LR5e3_E2000_1/metrics.mat",
    'BSpline': "multiscale_results/sisr/Bspline_s9_Ds8_LR1e3_E2000_1/metrics.mat",
    'Multiscale_1': "multiscale_results/sisr/MscaleHL_s1o9_Ds8_ST4_SHF384_LR1e3_E2000 _1/metrics.mat",
    'Multiscale_2': "multiscale_results/sisr/Mscale2_ST6_Ds8_LR8e3_E4000_1/metrics.mat"
}

files_info = {
    'Wire': "multiscale_results/sisr/WIRE_s9_Ds8_o8_LR5e3_E2000_1/info.mat",
    'BSpline': "multiscale_results/sisr/Bspline_s9_Ds8_LR1e3_E2000_1/info.mat",
    'Multiscale_1': "multiscale_results/sisr/MscaleHL_s1o9_Ds8_ST4_SHF384_LR1e3_E2000 _1/info.mat",
    'Multiscale_2': "multiscale_results/sisr/Mscale2_ST6_Ds8_LR8e3_E4000_1/info.mat"
}

### Tabulation of results 
variables = {}
data = {}

for i, filepath in files_metrics.items():
    mat = sio.loadmat(filepath)
    for nonlin in mat.keys():
        if not nonlin.startswith('__'):
            values = mat[nonlin][0, 0]
            for key in mat[nonlin].dtype.names:
                if key not in data:
                    data[key] = []
            for label in values.dtype.names:
                if isinstance(values[label][0], (list, np.ndarray)):
                    data[label].append(values[label][0].tolist())
                else:
                    data[label].append(values[label][0, 0])

# Create a DataFrame from the dictionary where the keys are the columns and the nonlin types are the rows
df = pd.DataFrame(data, index=list(files_metrics.keys()))
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)  # Allows the DataFrame to expand to full width
pd.set_option('display.max_colwidth', None)  # Prevents truncation of column contents
print(df)

### Save image
for i, filepath in files_info.items():
    mat = sio.loadmat(filepath)
    for nonlin in mat.keys():
        if not nonlin.startswith('__'):
            values = mat[nonlin][0, 0]
            image = values['rec']
            plt.imsave(save_path, np.clip(abs(image), 0, 1), 
                vmin=0.0,
                vmax=1.0)