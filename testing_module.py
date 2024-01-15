import numpy as np
import torch
from evaluate import FreiburgMap
from datasets import *
import csv

def test(model, map_data, test_dataloader):

    model = model.cuda() 
    model.eval()

    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    #model.classifier[1].register_forward_hook(get_activation('latent_vector'))
    model.avgpool.register_forward_hook(get_activation('latent_vector'))
    freiburg_map = FreiburgMap(map_data, model, activation)

    #Testing Accuracy
    
    errors = []
    total = 0
    correct = 0

    with torch.no_grad():
        for data in test_dataloader:
            test_img, test_coor, label = data[0].cuda(), data[1].cuda(), data[2].cuda()
            error, room_predicted = freiburg_map.evaluate_error_position(test_img, test_coor, model, activation)
            errors.append(error)
            total += 1
            correct += (room_predicted == label).sum().item()

    error_loc = np.mean(errors)
    accuracy = 100 * correct / total
    print('Error localización test images (m):', error_loc)
    print('Accuracy of the network on test images:', accuracy)
    return accuracy, error_loc

ruta = '/home/arvc/Juanjo/develop/Extension Orlando/'
models_names = ['AlexNet', 'resnet_152', 'convnext', 'resnext', 'efficientnet', 'mobilenet']
training_sequences_names = ['noDA', 'DA1', 'DA2', 'DA3', 'DA4', 'DA5', 'DA6']
results = '/home/arvc/Juanjo/develop/Extension Orlando/global_results.csv'
with open(results, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(
        ["Model_name", "Training Dataset", "Room Retrieval", "Error Loc. (m) "])
    for model_base_name in models_names:
        for dataset_name in training_sequences_names:
            model_name = ruta + model_base_name + '_' + dataset_name
            test_model = torch.load(model_name).cuda()
            accuracy, error_loc = test(test_model, map_data, global_test_dataloader)
            writer.writerow([model_name, dataset_name, accuracy, error_loc])







