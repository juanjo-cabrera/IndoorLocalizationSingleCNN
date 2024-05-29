
import torch.nn as nn
from torch import optim
import torch
from validation import compute_validation
import matplotlib.pyplot as plt


def train(model, train_dataloader, validation_dataloader, model_name, max_epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    model = model.cuda()
    model.train(True)
    print(model)

    counter = []
    loss_history = []
    val_history = []
    epoch_count = []
    val_history.append(0)
    iteration_number= 0

    # for epoch in range(0, max_epochs):
    #     for i, data in enumerate(train_dataloader, 0):
    #         input, label = data
    #         input, label = input.cuda(), label.cuda()
    #         optimizer.zero_grad()
    #         output = model(input)
    #         loss = criterion(output, label)
    #         loss.backward()
    #         optimizer.step()
    #         if i % 10 == 0 :
    #             print("Epoch number {}\n Current loss {}".format(epoch, loss.item()))
    #             iteration_number +=10
    #             counter.append(iteration_number)
    #             loss_history.append(loss.item())
    #             val_accuracy = compute_validation(model, validation_dataloader)
    #             model.train(True)
    #             print(' Validation accuracy: ', val_accuracy)
    #             print('\n')
    #             if val_accuracy > max(val_history):
    #                 torch.save(model, model_name + '_prueba')
    #             if val_accuracy == 100:
    #                 break
    #             val_history.append(val_accuracy)
    #     if val_accuracy == 100:
    #         break

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
        val_history.append(val_accuracy)

        print(f"Epoch {epoch + 1}/{max_epochs}, Loss: {average_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

        if val_accuracy > max(val_history):
            torch.save(model, model_name + '_prueba')

        if val_accuracy == 100:
            break

    # # Generar los gráficos
    # plt.figure(figsize=(12, 5))
    #
    # # Pérdida durante el entrenamiento
    # plt.subplot(1, 2, 1)
    # plt.plot(counter, loss_history, label='Loss')
    # plt.xlabel('Iterations')
    # plt.ylabel('Loss')
    # plt.title('Training Loss')
    # plt.legend()
    #
    # # Precisión de validación durante el entrenamiento
    # plt.subplot(1, 2, 2)
    # plt.plot(counter, val_history[1:], label='Validation Accuracy')  # Skip initial 0 value
    # plt.xlabel('Iterations')
    # plt.ylabel('Validation Accuracy (%)')
    # plt.title('Validation Accuracy')
    # plt.legend()

    plt.show()

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
    plt.plot(epoch_count, val_history, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy (%)')
    plt.title('Validation Accuracy per Epoch')
    plt.legend()

    plt.show()