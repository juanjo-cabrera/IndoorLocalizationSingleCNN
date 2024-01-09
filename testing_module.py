import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import multiprocessing
from multiprocessing import Process
import torchvision.models as models
import torch
from evaluate import FreiburgMap






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









