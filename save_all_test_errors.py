import numpy as np
import torch
from evaluate import FreiburgMap
from datasets import *
import csv


def compute_errors(errors):
    errors_cuadrado = np.power(errors, 2)
    mae = np.mean(errors)
    mse = np.mean(errors_cuadrado)
    rmse = np.sqrt(mse)
    varianza = np.mean(np.power(errors - mae, 2))
    desv = np.sqrt(varianza)
    return mae, varianza, desv, mse, rmse


def test(model, map_data, test_dataloader):
    model = model.cuda()
    model.eval()

    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    # model.classifier[1].register_forward_hook(get_activation('latent_vector'))
    model.avgpool.register_forward_hook(get_activation('latent_vector'))
    freiburg_map = FreiburgMap(map_data, model, activation)

    # Testing Accuracy

    errors = []
    total = 0
    correct = 0
    times = []

    with torch.no_grad():
        for data in test_dataloader:
            test_img, test_coor, label = data[0].cuda(), data[1].cuda(), data[2].cuda()
            error, room_predicted, processing_time = freiburg_map.evaluate_error_position(test_img, test_coor, model,
                                                                                          activation)
            errors.append(error)
            times.append(processing_time)
            total += 1
            correct += (room_predicted == label).sum().item()

    # mae, varianza, desv, mse, rmse = compute_errors(errors)
    # print('Mean Absolute Error in test images (m):', mae)
    # print('Varianza:', varianza)
    # print('Desviacion', desv)
    # print('Mean Square Error (m2)', mse)
    # print('Root Mean Square Error (m)', rmse)
    # mean_processing_time = np.mean(times)
    # accuracy = 100 * correct / total
    # print('Accuracy of the network on test images:', accuracy)
    return errors


ruta = '/home/arvc/Juanjo/develop/Extension Orlando/'
models_names = ['AlexNet', 'resnet_152', 'resnext', 'efficientnet', 'mobilenet']
# models_names = ['convnext']
training_sequences_names = ['noDA']
illuminations = ['cloudy', 'night', 'sunny']
# training_sequences_names = ['noDA', 'DA1', 'DA2', 'DA3', 'DA4', 'DA5', 'DA6']
base_results = '/home/arvc/Juanjo/develop/Extension Orlando/'
# Read data from CSV files
for model_base_name in models_names:
    print(model_base_name)
    for dataset_name in training_sequences_names:
        print(dataset_name)
        for illumination in illuminations:
            print(illumination)
            # Construct the file path
            results = base_results + illumination + '_' + model_base_name + '_' + dataset_name + '.csv'
            with open(results, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Errors'])

                model_name = ruta + model_base_name + '_' + dataset_name
                test_model = torch.load(model_name).cuda()
                if illumination == 'cloudy':
                    errors = test(test_model, map_data, test_dataloader_cloudy)
                elif illumination == 'night':
                    errors = test(test_model, map_data, test_dataloader_night)
                elif illumination == 'sunny':
                    errors = test(test_model, map_data, test_dataloader_sunny)

                for error in errors:
                    writer.writerow([error])

#
# print('NIGHT')
# base_results = '/home/arvc/Juanjo/develop/Extension Orlando/night_convnext_'
# for model_base_name in models_names:
#     for dataset_name in training_sequences_names:
#         results = base_results + dataset_name + '.csv'
#         with open(results, 'w', newline='') as file:
#             writer = csv.writer(file)
#             writer.writerow(['Errors'])
#
#             model_name = ruta + model_base_name + '_' + dataset_name
#             test_model = torch.load(model_name).cuda()
#             errors = test(test_model, map_data, test_dataloader_night)
#             for error in errors:
#                 writer.writerow([error])
#
#
#
# print('SUNNY')
# base_results = '/home/arvc/Juanjo/develop/Extension Orlando/sunny_convnext_'
# for model_base_name in models_names:
#     for dataset_name in training_sequences_names:
#         results = base_results + dataset_name + '.csv'
#         with open(results, 'w', newline='') as file:
#             writer = csv.writer(file)
#             writer.writerow(['Errors'])
#
#             model_name = ruta + model_base_name + '_' + dataset_name
#             test_model = torch.load(model_name).cuda()
#             errors = test(test_model, map_data, test_dataloader_sunny)
#             for error in errors:
#                 writer.writerow([error])
