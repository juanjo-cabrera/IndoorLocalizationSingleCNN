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

    #model.classifier[1].register_forward_hook(get_activation('latent_vector'))
    model.avgpool.register_forward_hook(get_activation('latent_vector'))
    freiburg_map = FreiburgMap(map_data, model, activation)

    #Testing Accuracy
    
    errors = []
    total = 0
    correct = 0
    times = []

    with torch.no_grad():
        for data in test_dataloader:
            test_img, test_coor, label = data[0].cuda(), data[1].cuda(), data[2].cuda()
            error, room_predicted, processing_time = freiburg_map.evaluate_error_position(test_img, test_coor, model, activation)
            errors.append(error)
            times.append(processing_time)
            total += 1
            correct += (room_predicted == label).sum().item()

    mae, varianza, desv, mse, rmse = compute_errors(errors)
    print('Mean Absolute Error in test images (m):', mae)
    print('Varianza:', varianza)
    print('Desviacion', desv)
    print('Mean Square Error (m2)', mse)
    print('Root Mean Square Error (m)', rmse)
    mean_processing_time = np.mean(times)
    accuracy = 100 * correct / total
    print('Accuracy of the network on test images:', accuracy)
    return accuracy, mae, varianza, desv, mse, rmse, mean_processing_time

ruta = '/home/arvc/Juanjo/develop/Extension Orlando/'
models_names = ['AlexNet', 'resnet_152', 'convnext', 'resnext', 'efficientnet', 'mobilenet']
training_sequences_names = ['noDA', 'DA1', 'DA2', 'DA3', 'DA4', 'DA5', 'DA6']

# results = '/home/arvc/Juanjo/develop/Extension Orlando/global_results_after_revision.csv'
# with open(results, 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(
#         ["Model_name", "Training Dataset", "Room Retrieval", "MAE (m) ", 'Varianza (m2)', 'Desv. (m)', 'MSE (m)', 'RMSE (m)', 'Mean Time (s)'])
#     for model_base_name in models_names:
#         for dataset_name in training_sequences_names:
#             model_name = ruta + model_base_name + '_' + dataset_name
#             test_model = torch.load(model_name).cuda()
#             accuracy, mae, varianza, desv, mse, rmse, mean_processing_time = test(test_model, map_data, global_test_dataloader)
#             writer.writerow([model_name, dataset_name, accuracy, mae, varianza, desv, mse, rmse, mean_processing_time])

print('CLOUDY')
results = '/home/arvc/Juanjo/develop/Extension Orlando/cloudy_results_after_revision.csv'
with open(results, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(
        ["Model_name", "Training Dataset", "Room Retrieval", "MAE (m) ", 'Varianza (m2)', 'Desv. (m)', 'MSE (m)', 'RMSE (m)', 'Mean Time (s)'])
    for model_base_name in models_names:
        for dataset_name in training_sequences_names:
            model_name = ruta + model_base_name + '_' + dataset_name
            test_model = torch.load(model_name).cuda()
            accuracy, mae, varianza, desv, mse, rmse, mean_processing_time = test(test_model, map_data, test_dataloader_cloudy)
            writer.writerow([model_name, dataset_name, accuracy, mae, varianza, desv, mse, rmse, mean_processing_time])

print('NIGHT')
results = '/home/arvc/Juanjo/develop/Extension Orlando/night_results_after_revision.csv'
with open(results, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(
        ["Model_name", "Training Dataset", "Room Retrieval", "MAE (m) ", 'Varianza (m2)', 'Desv. (m)', 'MSE (m)', 'RMSE (m)', 'Mean Time (s)'])
    for model_base_name in models_names:
        for dataset_name in training_sequences_names:
            model_name = ruta + model_base_name + '_' + dataset_name
            test_model = torch.load(model_name).cuda()
            accuracy, mae, varianza, desv, mse, rmse, mean_processing_time = test(test_model, map_data, test_dataloader_night)
            writer.writerow([model_name, dataset_name, accuracy, mae, varianza, desv, mse, rmse, mean_processing_time])

print('SUNNY')
results = '/home/arvc/Juanjo/develop/Extension Orlando/sunny_results_after_revision.csv'
with open(results, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(
        ["Model_name", "Training Dataset", "Room Retrieval", "MAE (m) ", 'Varianza (m2)', 'Desv. (m)', 'MSE (m)', 'RMSE (m)', 'Mean Time (s)'])
    for model_base_name in models_names:
        for dataset_name in training_sequences_names:
            model_name = ruta + model_base_name + '_' + dataset_name
            test_model = torch.load(model_name).cuda()
            accuracy, mae, varianza, desv, mse, rmse, mean_processing_time = test(test_model, map_data, test_dataloader_sunny)
            writer.writerow([model_name, dataset_name, accuracy, mae, varianza, desv, mse, rmse, mean_processing_time])



