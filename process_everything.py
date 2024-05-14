from datasets import *
from training_module import train
from testing_module import test
from models import *
import csv
from torchvision.models import resnet152, convnext_large, resnext101_64x4d, efficientnet_v2_l, mobilenet_v3_large, alexnet

def run(model, model_name, train_dataloader):
    torch.multiprocessing.freeze_support()

    print('Start training of' + model_name)
    # train(model, train_dataloader, validation_dataloader, model_name, max_epochs=30)

    test_model = torch.load(model_name).cuda()
    accuracy_cloudy, error_loc_cloudy = test(test_model, map_data, test_dataloader_cloudy)
    accuracy_night, error_loc_night = test(test_model, map_data, test_dataloader_night)
    accuracy_sunny, error_loc_sunny = test(test_model, map_data, test_dataloader_sunny)

    return accuracy_cloudy, accuracy_night, accuracy_sunny, error_loc_cloudy, error_loc_night, error_loc_sunny


if __name__ == '__main__':
    results = '/home/arvc/Juanjo/develop/Extension Orlando/results.csv'
    models_names = ['AlexNet', 'resnet_152', 'convnext', 'resnext', 'efficientnet', 'mobilenet']
    #models = [convnext, resnext, efficientnet, mobilenet]
    training_sequences = [train_noDA_dataloader, train_DA1_dataloader, train_DA2_dataloader, train_DA3_dataloader, train_DA4_dataloader, train_DA5_dataloader, train_DA6_dataloader]
    training_sequences_names = ['noDA', 'DA1', 'DA2', 'DA3', 'DA4', 'DA5', 'DA6']
    with open(results, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Model_name", "Training Dataset", "Room Retrieval Cloudy", "Room Retrieval Night", "Room Retrieval Sunny", "Error Loc. (m) Cloudy", "Error Loc. (m) Night", "Error Loc. (m) Sunny"])
        for model_base_name in models_names:
            i = 0
            for training_sequence in training_sequences:
                if model_base_name == 'AlexNet':
                    model = alexnet(pretrained=True)
                    model.classifier[6] = nn.Linear(4096, 9)
                elif model_base_name == 'resnet_152':
                    model = resnet152(weights='IMAGENET1K_V2')
                    model.fc = nn.Linear(2048, 9)
                elif model_base_name == 'convnext':
                    model = convnext_large(weights='IMAGENET1K_V1')
                    model.classifier[2] = nn.Linear(1536, 9)
                elif model_base_name == 'resnext':
                    model = resnext101_64x4d(weights='IMAGENET1K_V1')
                    model.fc = nn.Linear(2048, 9)
                elif model_base_name == 'efficientnet':
                    model = efficientnet_v2_l(weights='IMAGENET1K_V1')
                    model.classifier[1] = nn.Linear(1280, 9)
                elif model_base_name == 'mobilenet':
                    model = mobilenet_v3_large(weights='IMAGENET1K_V1')
                    model.classifier[3] = nn.Linear(1280, 9)

                dataset_name = training_sequences_names[i]
                model_name = model_base_name + '_' + dataset_name
                accuracy_cloudy, accuracy_night, accuracy_sunny, error_loc_cloudy, error_loc_night, error_loc_sunny = run(model, model_name, training_sequence)
                writer.writerow([model_name, dataset_name, accuracy_cloudy, accuracy_night, accuracy_sunny, error_loc_cloudy, error_loc_night, error_loc_sunny])
                i += 1


