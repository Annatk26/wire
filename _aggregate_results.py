import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

filepath = "multiscale_results/denoise/T30.0_SNR2/Final"
# save_file = "multiscale_results/sisr/DS_4/Noise"
save_file = filepath
os.makedirs(save_file, exist_ok=True)

nonlin_filename = {
    'Wire': "WIRE_s8_o7_LR5e3_E2000_2",
    'BSpline': "BSpline_s9_LR4e3_1",
    'Multiscale_1': "MscaleHL_s1o9_ST4_3_SHF384_LR8e3_E4000_1",
    'Multiscale_2': "Mscale2_ST4_3_LR8e3_E4000_2",
    'Multiscale_Hierarchical_2Stages': "MscaleHier_ST4_LR8e3_E4000_1",
    # 'Multiscale_Hierarchical_3Stages': "MscaleHier_ST4_3_LR8e3_E4000_1"
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
                if isinstance(values[label][0], (list, np.ndarray)) and isinstance(values[label][0][0], (int, float, np.number)):
                    data[label].append(np.round(values[label][0], 4).tolist())
                else:
                    data[label].append(values[label][0, 0])

# Create a DataFrame from the dictionary where the keys are the columns and the nonlin types are the rows
for key, value in data.items():
    print(f"Key: {key}, Length: {len(value)}")
print(data)

df = pd.DataFrame(data, index=list(nonlin_filename.keys()))
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)  # Allows the DataFrame to expand to full width
pd.set_option('display.max_colwidth', None)  # Prevents truncation of column contents
df.to_markdown(os.path.join(save_file, "Agg_results.md"))


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
n_rows = 3
n_cols = 2
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10), facecolor='#f7f7f7')
gs = fig.add_gridspec(n_rows, n_cols, wspace=0.2, hspace=0.2)  # Adjust wspace and hspace

axes = axes.flatten()
for i, (img, ax, label) in enumerate(zip(images, axes, labels)):
    row = i // n_cols
    col = i % n_cols
    ax.imshow(np.clip(abs(img), 0, 1), aspect='auto', cmap='gray')
    ax.set_title(label, fontsize=16, weight='bold', pad=10)
    ax.axis('off')

plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
plt.savefig(os.path.join(save_file, 'Output_img.png'), bbox_inches='tight')