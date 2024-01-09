import matplotlib.pyplot as plt
import numpy as np

# Datos de la tabla
backbone_models = ['AlexNet_noDA', 'resnet_152_noDA', 'resnext_noDA', 'mobilenet_noDA', 'efficientnet_noDA', 'convnext_noDA']
cloudy_errors = [0.25, 0.58, 0.49, 0.24, 0.42, 0.22]
night_errors = [0.27, 0.47, 0.68, 0.28, 0.37, 0.26]
sunny_errors = [2.20, 3.81, 2.74, 1.96, 2.70, 0.83]

# Configuración del gráfico
fig, ax = plt.subplots(figsize=(10, 6))
cloudy_line, = ax.plot(backbone_models, cloudy_errors, marker='o', linestyle='-', label='Cloudy')
night_line, = ax.plot(backbone_models, night_errors, marker='o', linestyle='-', label='Night')
sunny_line, = ax.plot(backbone_models, sunny_errors, marker='o', linestyle='-', label='Sunny')

# Etiquetas y título
ax.set_xlabel('Backbone Models')
ax.set_ylabel('Hierarchical Localization Error (m)')
ax.set_title('Hierarchical Localization Error for Different Backbone Models')
ax.legend()
ax.grid(True)

# Offset vertical para las etiquetas
offset = 0.1

# Etiquetas con los valores de cada punto
for i, model in enumerate(backbone_models):
    ax.text(i, cloudy_errors[i] + offset, f'{cloudy_errors[i]:.2f}', color=cloudy_line.get_color(), ha='center', va='bottom')
    ax.text(i, night_errors[i] + offset, f'{night_errors[i]:.2f}', color=night_line.get_color(), ha='center', va='bottom')
    ax.text(i, sunny_errors[i] - offset, f'{sunny_errors[i]:.2f}', color=sunny_line.get_color(), ha='center', va='top')

# Rotar etiquetas del eje x para mejor visibilidad
plt.xticks(rotation=45, ha="right")

# Mostrar gráfico
plt.tight_layout()
plt.show()
