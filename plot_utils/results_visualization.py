import matplotlib.pyplot as plt
import numpy as np

# Datos de la tabla
backbone_models = ['AlexNet', 'ResNet-152', 'ResNeXt-101', 'MobileNetV3', 'EfficientNetV2', 'ConvNeXt Large']
cloudy_errors = [0.25, 0.58, 0.49, 0.24, 0.42, 0.22]
night_errors = [0.27, 0.47, 0.68, 0.28, 0.37, 0.26]
sunny_errors = [2.20, 3.81, 2.74, 1.96, 2.70, 0.83]
global_errors = [0.81, 1.46, 1.20, 0.75, 1.05, 0.41]
font = 25
# Configuración del gráfico
fig, ax = plt.subplots(figsize=(12, 8))
cloudy_line, = ax.plot(backbone_models, cloudy_errors, marker='o', linestyle='-', label='Cloudy', linewidth=4)
night_line, = ax.plot(backbone_models, night_errors, marker='o', linestyle='-', label='Night', linewidth=4)
sunny_line, = ax.plot(backbone_models, sunny_errors, marker='o', linestyle='-', label='Sunny', linewidth=4)
global_line, = ax.plot(backbone_models, global_errors, marker='o', linestyle='-', label='Global', linewidth=4)

# Etiquetas y título con texto más grande
ax.set_xlabel('Backbone Models', fontsize=font)
ax.set_ylabel('Hierarchical Localization Error (m)', fontsize=font)
ax.set_title('Hierarchical Localization Error for Different Backbone Models', fontsize=font)
ax.legend(fontsize=font)
ax.grid(True)

# Offset vertical para las etiquetas
offset = 0.22

# Etiquetas con los valores de cada punto y texto más grande
for i, model in enumerate(backbone_models):

    if np.bitwise_or(i == 1, i == 4):
        ax.text(i, cloudy_errors[i] + 0.1, f'{cloudy_errors[i]:.2f}', color=cloudy_line.get_color(), ha='center',
                va='bottom', fontsize=font)
        ax.text(i, night_errors[i] - 0.2, f'{night_errors[i]:.2f}', color=night_line.get_color(), ha='center',
                va='bottom', fontsize=font)
    elif i == 5:
        ax.text(i, cloudy_errors[i] - 0.2, f'{cloudy_errors[i]:.2f}', color=cloudy_line.get_color(), ha='center',
                va='bottom', fontsize=font)
        ax.text(i + 0.05, night_errors[i] + 0.045, f'{night_errors[i]:.2f}', color=night_line.get_color(), ha='center',
                va='bottom', fontsize=font)

    else:
        ax.text(i, cloudy_errors[i] - 0.2, f'{cloudy_errors[i]:.2f}', color=cloudy_line.get_color(), ha='center',
                va='bottom', fontsize=font)
        ax.text(i, night_errors[i] + 0.1, f'{night_errors[i]:.2f}', color=night_line.get_color(), ha='center',
                va='bottom', fontsize=font)
    if i == 1:
        ax.text(i, sunny_errors[i] + 0.16, f'{sunny_errors[i]:.2f}', color=sunny_line.get_color(), ha='center',
                va='top', fontsize=font)
    elif i == 0:
        ax.text(i-0.1, sunny_errors[i] + 0.2, f'{sunny_errors[i]:.2f}', color=sunny_line.get_color(), ha='center',
                va='top', fontsize=font)
    elif i == 5:
        ax.text(i + 0.1, sunny_errors[i] + 0.2, f'{sunny_errors[i]:.2f}', color=sunny_line.get_color(), ha='center',
                va='top', fontsize=font)
    else:
        ax.text(i, sunny_errors[i] + 0.28, f'{sunny_errors[i]:.2f}', color=sunny_line.get_color(), ha='center', va='top', fontsize=font)
    ax.text(i, global_errors[i] + 0.28, f'{global_errors[i]:.2f}', color=global_line.get_color(), ha='center', va='top', fontsize=font)
# Rotar etiquetas del eje x para mejor visibilidad
# plt.xticks(rotation=0, ha="right", fontsize=16)
plt.xticks(fontsize=font)
# Mostrar gráfico
plt.tight_layout()
plt.show()
