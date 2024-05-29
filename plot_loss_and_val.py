import matplotlib.pyplot as plt

# Lista de épocas
epochs = list(range(1, 31))

# Datos de pérdida
loss_history = [
    1.9510, 1.5201, 0.9279, 0.4692, 0.2190, 0.1292, 0.0796, 0.0537,
    0.0351, 0.0272, 0.0211, 0.0185, 0.0151, 0.0140, 0.0119, 0.0107,
    0.0094, 0.0085, 0.0079, 0.0073, 0.0068, 0.0062, 0.0059, 0.0055,
    0.0052, 0.0049, 0.0047, 0.0044, 0.0042, 0.0040
]

# Datos de precisión de validación
val_accuracy_history = [
    42.45, 67.99, 86.33, 97.66, 97.84, 98.92, 98.92, 99.10,
    99.10, 99.28, 99.28, 99.28, 98.92, 99.28, 99.28, 99.28,
    99.46, 99.46, 99.46, 99.28, 99.46, 99.64, 99.28, 99.28,
    99.28, 99.64, 99.46, 99.64, 99.46, 99.28
]

# Generar los gráficos
plt.figure(figsize=(12, 5))

# Gráfico de pérdida
plt.subplot(1, 2, 1)
plt.plot(epochs, loss_history, label='Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss per Epoch')
plt.legend()

# Gráfico de precisión de validación
plt.subplot(1, 2, 2)
plt.plot(epochs, val_accuracy_history, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy (%)')
plt.title('Validation Accuracy per Epoch')
plt.legend()

# Mostrar los gráficos
plt.show()
