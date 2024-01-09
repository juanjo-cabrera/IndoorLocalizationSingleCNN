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


class Config():
   testing_cloudy_dir = '/home/arvc/Juanjo/Datasets/Friburgo/TestCloudy'
   testing_night_dir = '/home/arvc/Juanjo/Datasets/Friburgo/TestNight'
   testing_sunny_dir = '/home/arvc/Juanjo/Datasets/Friburgo/TestSunny'

   map_dir = '/home/arvc/Juanjo/Datasets/Friburgo/Entrenamiento'

   training_noDA_dir = '/home/arvc/Juanjo/Datasets/Friburgo/Entrenamiento'
   training_DA1_dir = '/home/arvc/Juanjo/Datasets/Friburgo/DA_carpetas/DA1'
   training_DA2_dir = '/home/arvc/Juanjo/Datasets/Friburgo/DA_carpetas/DA2'
   training_DA3_dir = '/home/arvc/Juanjo/Datasets/Friburgo/DA_carpetas/DA3'
   training_DA4_dir = '/home/arvc/Juanjo/Datasets/Friburgo/DA_carpetas/DA4'
   training_DA5_dir = '/home/arvc/Juanjo/Datasets/Friburgo/DA_carpetas/DA5'
   training_DA6_dir = '/home/arvc/Juanjo/Datasets/Friburgo/DA_carpetas/DA6'

   validation_dir= '/home/arvc/Juanjo/Datasets/Friburgo/Validacion'
   train_batch_size = 16
   validation_batch_size = 512


class TestDataset(Dataset):
    def __init__(self,imageFolderDataset, transform=None,should_invert=True):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.should_invert = should_invert

    def __getitem__(self,index):
        def coordenadas(ruta):
           x_index = ruta.index('_x')
           y_index = ruta.index('_y')
           a_index = ruta.index('_a')
           x=ruta[x_index+2:y_index]
           y=ruta[y_index+2:a_index]
           coor_list= [x,y]
           coor=torch.from_numpy(np.array(coor_list,dtype=np.float32))
           return coor

        img_tuple = self.imageFolderDataset.imgs[index]
        img = self.imageFolderDataset[index][0]
        coor = coordenadas(img_tuple[0])
        room_label = self.imageFolderDataset.targets[index]
        return img, coor, room_label
    
    def __len__(self):
        return len(self.imageFolderDataset.imgs)
    

test_data_cloudy = dset.ImageFolder(root=Config.testing_cloudy_dir, transform=transforms.ToTensor())
test_dataset_cloudy = TestDataset(test_data_cloudy)
test_dataloader_cloudy = DataLoader(test_dataset_cloudy,
                        shuffle=False,
                        num_workers=0,
                        batch_size=1)

test_data_night = dset.ImageFolder(root=Config.testing_night_dir, transform=transforms.ToTensor())
test_dataset_night = TestDataset(test_data_night)
test_dataloader_night = DataLoader(test_dataset_night,
                        shuffle=False,
                        num_workers=0,
                        batch_size=1)

test_data_sunny = dset.ImageFolder(root=Config.testing_sunny_dir, transform=transforms.ToTensor())
test_dataset_sunny = TestDataset(test_data_sunny)
test_dataloader_sunny = DataLoader(test_dataset_sunny,
                        shuffle=False,
                        num_workers=0,
                        batch_size=1)

train_noDA_data = dset.ImageFolder(root=Config.training_noDA_dir, transform=transforms.ToTensor())
train_noDA_dataloader = DataLoader(train_noDA_data,
                        shuffle=True,
                        num_workers=0,
                        batch_size=Config.train_batch_size)

train_DA1_data = dset.ImageFolder(root=Config.training_DA1_dir, transform=transforms.ToTensor())
train_DA1_dataloader = DataLoader(train_DA1_data,
                        shuffle=True,
                        num_workers=0,
                        batch_size=Config.train_batch_size)

train_DA2_data = dset.ImageFolder(root=Config.training_DA2_dir, transform=transforms.ToTensor())
train_DA2_dataloader = DataLoader(train_DA2_data,
                        shuffle=True,
                        num_workers=0,
                        batch_size=Config.train_batch_size)

train_DA3_data = dset.ImageFolder(root=Config.training_DA3_dir, transform=transforms.ToTensor())
train_DA3_dataloader = DataLoader(train_DA3_data,
                        shuffle=True,
                        num_workers=0,
                        batch_size=Config.train_batch_size)

train_DA4_data = dset.ImageFolder(root=Config.training_DA4_dir, transform=transforms.ToTensor())
train_DA4_dataloader = DataLoader(train_DA4_data,
                        shuffle=True,
                        num_workers=0,
                        batch_size=Config.train_batch_size)

train_DA5_data = dset.ImageFolder(root=Config.training_DA5_dir, transform=transforms.ToTensor())
train_DA5_dataloader = DataLoader(train_DA5_data,
                        shuffle=True,
                        num_workers=0,
                        batch_size=Config.train_batch_size)

train_DA6_data = dset.ImageFolder(root=Config.training_DA6_dir, transform=transforms.ToTensor())
train_DA6_dataloader = DataLoader(train_DA6_data,
                        shuffle=True,
                        num_workers=0,
                        batch_size=Config.train_batch_size)

validation_data = dset.ImageFolder(root=Config.validation_dir, transform= transforms.ToTensor())
validation_dataloader = DataLoader(validation_data,
                        shuffle=True,
                        num_workers=0,
                        batch_size=Config.train_batch_size)


map_data = dset.ImageFolder(root=Config.map_dir, transform=transforms.ToTensor())
