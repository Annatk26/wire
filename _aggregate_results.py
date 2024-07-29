import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

filepath = "multiscale_results/sisr/DS_8"
nonlin_filename = {
    'Wire': "WIRE_s9_Ds8_o8_LR5e3_E2000_1",
    'BSpline': "Bspline_s9_Ds8_LR1e3_E2000_1",
    'Multiscale_1': "MscaleHL_s1o9_Ds8_ST4_SHF384_LR1e3_E2000 _1",
    'Multiscale_2': "Mscale2_ST6_Ds8_LR8e3_E4000_1"
}

### Tabulation of results 
variables = {}
data = {}

for i, file in nonlin_filename.items():
    mat = sio.loadmat(os.path.join(filepath, file, "metrics.mat"))
    for nonlin in mat.keys():
        if not nonlin.startswith('__'):
            values = mat[nonlin][0, 0]
            for key in mat[nonlin].dtype.names:
                if key not in data:
                    data[key] = []
            for label in values.dtype.names:
                if isinstance(values[label][0], (list, np.ndarray)):
                    # formatted_values = [x:.4f for x in values[label][0]] if isinstance(values[label][0], (list, np.ndarray)) else f"{values[label][0, 0]:.4f}"
                    # data[label].append(formatted_values)
                    data[label].append(np.round(values[label][0], 4).tolist())
                else:
                    data[label].append(values[label][0, 0])

# Create a DataFrame from the dictionary where the keys are the columns and the nonlin types are the rows
df = pd.DataFrame(data, index=list(nonlin_filename.keys()))
df = df.applymap(lambda x: round(x, 4) if isinstance(x, (int, float)) else x)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)  # Allows the DataFrame to expand to full width
pd.set_option('display.max_colwidth', None)  # Prevents truncation of column contents
df.to_markdown(os.path.join(filepath, "Agg_results.md"))


### Save image
images = []
labels = []
for i, file in nonlin_filename.items():
    mat = sio.loadmat(os.path.join(filepath, file, "info.mat"))
    labels.append(i)
    for nonlin in mat.keys():
        if not nonlin.startswith('__'):
            values = mat[nonlin][0, 0]
            images.append(values['rec'])
n_rows = 2
n_cols = 2
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10), facecolor='#f7f7f7')
gs = fig.add_gridspec(n_rows, n_cols, wspace=0.2, hspace=0.2)  # Adjust wspace and hspace

axes = axes.flatten()
for i, (img, ax, label) in enumerate(zip(images, axes, labels)):
    row = i // n_cols
    col = i % n_cols
    ax.imshow(np.clip(abs(img), 0, 1), aspect='auto')
    ax.set_title(label, fontsize=16, weight='bold', pad=10)
    ax.axis('off')

plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
plt.savefig(os.path.join(filepath, 'Output_img.png'), bbox_inches='tight')