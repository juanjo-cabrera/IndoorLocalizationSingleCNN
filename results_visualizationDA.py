import matplotlib.pyplot as plt
import numpy as np

# Data from the table
training_datasets = ['Baseline', 'DA 1', 'DA 2', 'DA 3', 'DA 4', 'DA 5', 'DA 6']
# training_datasets = ['Baseline', 'Augmented 1 \n (Spotlights)', 'Augmented 2 \n (Shadows)', 'Augmented 3 \n (Brightness / Darkness)', 'Augmented 4 \n (Contrast)', 'Augmented 5 \n (Saturation)', 'Augmented 6 \n (Rotations)']
cloudy_errors = [0.22, 0.23, 0.22, 0.22, 0.22, 0.24, 0.22]
night_errors = [0.26, 0.28, 0.27, 0.27, 0.27, 0.27, 0.27]
sunny_errors = [0.83, 0.86, 0.93, 0.69, 0.57, 1.03, 0.73]
global_errors = [0.41, 0.43, 0.44, 0.37,0.34, 0.48, 0.38]
font = 25

# Plotting
fig, ax = plt.subplots(figsize=(12, 8))
cloudy_line, = ax.plot(training_datasets, cloudy_errors, marker='o', linestyle='-', label='Cloudy', linewidth=4)
night_line, = ax.plot(training_datasets, night_errors, marker='o', linestyle='-', label='Night', linewidth=4)
sunny_line, = ax.plot(training_datasets, sunny_errors, marker='o', linestyle='-', label='Sunny', linewidth=4)
global_line, = ax.plot(training_datasets, global_errors, marker='o', linestyle='-', label='Global', linewidth=4)
# Labels and title
ax.set_xlabel('Training Dataset', fontsize=font)
ax.set_ylabel('Hierarchical Localization Error (m)', fontsize=font)
ax.set_title('Hierarchical Localization Error for Different Training Datasets', fontsize=font)
ax.legend(fontsize=font)
ax.grid(True)

# Offset for labels
offset = 0.01


# Adding values as text to each point
for i, dataset in enumerate(training_datasets):
    if i == 5:
        ax.text(i, cloudy_errors[i] - 0.05, f'{cloudy_errors[i]:.2f}', color=cloudy_line.get_color(), ha='center', va='bottom', fontsize=font)
    else:
        ax.text(i, cloudy_errors[i], f'{cloudy_errors[i]:.2f}', color=cloudy_line.get_color(), ha='center',
                va='bottom', fontsize=font)
    ax.text(i, night_errors[i] + offset, f'{night_errors[i]:.2f}', color=night_line.get_color(), ha='center', va='bottom', fontsize=font)
    if np.bitwise_or(i == 2, i == 5):
        ax.text(i, sunny_errors[i] + 0.05, f'{sunny_errors[i]:.2f}', color=sunny_line.get_color(), ha='center',
                va='top', fontsize=font)
    else:
        ax.text(i, sunny_errors[i] - 0.02, f'{sunny_errors[i]:.2f}', color=sunny_line.get_color(), ha='center', va='top', fontsize=font)
    ax.text(i, global_errors[i] + 0.06, f'{global_errors[i]:.2f}', color=global_line.get_color(), ha='center', va='top',
            fontsize=font)
# Rotar etiquetas del eje x para mejor visibilidad
# plt.xticks(rotation=45, ha="right", fontsize=font)
plt.xticks(fontsize=font)
# Mostrar gráfico
plt.tight_layout()
plt.show()

# PLOT WITH ESTANDAR DESVIATION
# Data from the table
training_datasets = ['Baseline', 'DA 1', 'DA 2', 'DA 3', 'DA 4', 'DA 5', 'DA 6']
# training_datasets = ['Baseline', 'Augmented 1 \n (Spotlights)', 'Augmented 2 \n (Shadows)', 'Augmented 3 \n (Brightness / Darkness)', 'Augmented 4 \n (Contrast)', 'Augmented 5 \n (Saturation)', 'Augmented 6 \n (Rotations)']
cloudy_errors = [0.22, 0.23, 0.22, 0.22, 0.22, 0.24, 0.22]
night_errors = [0.26, 0.28, 0.27, 0.27, 0.27, 0.27, 0.27]
sunny_errors = [0.83, 0.86, 0.93, 0.69, 0.57, 1.03, 0.73]
global_errors = [0.41, 0.43, 0.44, 0.37,0.34, 0.48, 0.38]

cloudy_std = [0.18, 0.20, 0.19, 0.18, 0.18, 0.22, 0.20]
night_std = [0.16, 0.18, 0.16, 0.36, 0.16, 0.17, 0.15]
sunny_std = [1.74, 1.5, 1.98, 1.38, 0.87, 2.06, 1.34]
global_std = [0.98, 0.86, 1.11, 0.8, 0.51, 1.17, 0.76]


font = 25

# Plotting
fig, ax = plt.subplots(figsize=(12, 8))
ax.errorbar(training_datasets, cloudy_errors, yerr=cloudy_std, fmt='o', linestyle='-', label='Cloudy', linewidth=4)
ax.errorbar(training_datasets, night_errors, yerr=night_std, fmt='o', linestyle='-', label='Night', linewidth=4)
ax.errorbar(training_datasets, sunny_errors, yerr=sunny_std, fmt='o', linestyle='-', label='Sunny', linewidth=4)
ax.errorbar(training_datasets, global_errors, yerr=global_std, fmt='o', linestyle='-', label='Global', linewidth=4)

# Labels and title
ax.set_xlabel('Training Dataset', fontsize=font)
ax.set_ylabel('Hierarchical Localization Error (m)', fontsize=font)
ax.set_title('Hierarchical Localization Error for Different Training Datasets', fontsize=font)
ax.legend(fontsize=font)
ax.grid(True)


# Offset for labels
offset = 0.01


# Adding values as text to each point
for i, dataset in enumerate(training_datasets):
    if i == 5:
        ax.text(i, cloudy_errors[i] - 0.05, f'{cloudy_errors[i]:.2f}', color=cloudy_line.get_color(), ha='center', va='bottom', fontsize=font)
    else:
        ax.text(i, cloudy_errors[i], f'{cloudy_errors[i]:.2f}', color=cloudy_line.get_color(), ha='center',
                va='bottom', fontsize=font)
    ax.text(i, night_errors[i] + offset, f'{night_errors[i]:.2f}', color=night_line.get_color(), ha='center', va='bottom', fontsize=font)
    if np.bitwise_or(i == 2, i == 5):
        ax.text(i, sunny_errors[i] + 0.05, f'{sunny_errors[i]:.2f}', color=sunny_line.get_color(), ha='center',
                va='top', fontsize=font)
    else:
        ax.text(i, sunny_errors[i] - 0.02, f'{sunny_errors[i]:.2f}', color=sunny_line.get_color(), ha='center', va='top', fontsize=font)
    ax.text(i, global_errors[i] + 0.06, f'{global_errors[i]:.2f}', color=global_line.get_color(), ha='center', va='top',
            fontsize=font)
# Rotar etiquetas del eje x para mejor visibilidad
# plt.xticks(rotation=45, ha="right", fontsize=font)
plt.xticks(fontsize=font)
# Mostrar gráfico
plt.tight_layout()
plt.show()
