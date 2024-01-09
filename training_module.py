
import torch.nn as nn
from torch import optim
import torch
from validation import compute_validation


def train(model, train_dataloader, validation_dataloader, model_name, max_epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    model = model.cuda()
    model.train(True)
    print(model)

    counter = []
    loss_history = []
    val_history = []
    val_history.append(0)
    iteration_number= 0

    for epoch in range(0, max_epochs):
        for i, data in enumerate(train_dataloader, 0):
            input, label = data
            input, label = input.cuda(), label.cuda()
            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            if i % 10 == 0 :
                print("Epoch number {}\n Current loss {}".format(epoch, loss.item()))
                iteration_number +=10
                counter.append(iteration_number)
                loss_history.append(loss.item())
                val_accuracy = compute_validation(model, validation_dataloader)
                model.train(True)
                print(' Validation accuracy: ', val_accuracy)
                print('\n')
                if val_accuracy > max(val_history):
                    torch.save(model, model_name)
                if val_accuracy == 100:
                    break
                val_history.append(val_accuracy)
        if val_accuracy == 100:
            break


