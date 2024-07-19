from config import PARAMS
import torch.nn as nn
from torch import optim
import torch
from eval.validation import compute_validation
import matplotlib.pyplot as plt


def train(model, train_dataloader, validation_dataloader, model_name, max_epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    model = model.cuda()
    model.train(True)
    print(model)

    loss_history = []
    val_history = []
    epoch_count = []
    val_history.append(0.0)


    for epoch in range(max_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        average_loss = running_loss / len(train_dataloader)
        loss_history.append(average_loss)
        epoch_count.append(epoch + 1)

        val_accuracy = compute_validation(model, validation_dataloader)
        print(f"Epoch {epoch + 1}/{max_epochs}, Loss: {average_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

        if val_accuracy > max(val_history):
            torch.save(model, PARAMS.dataset_folder + 'models/' + model_name + '.pth')

        val_history.append(val_accuracy)
        if val_accuracy == 100:
            break



    # Generar los gráficos
    plt.figure(figsize=(12, 5))

    # Pérdida durante el entrenamiento
    plt.subplot(1, 2, 1)
    plt.plot(epoch_count, loss_history, label='Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')
    plt.legend()

    # Precisión de validación durante el entrenamiento
    plt.subplot(1, 2, 2)
    val_history.pop(0)
    plt.plot(epoch_count, val_history, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy (%)')
    plt.title('Validation Accuracy per Epoch')
    plt.legend()

    plt.show()