import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import numpy as np
import torch
from config import PARAMS


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
           x = ruta[x_index+2:y_index]
           y = ruta[y_index+2:a_index]
           coor_list = [x,y]
           coor = torch.from_numpy(np.array(coor_list,dtype=np.float32))
           return coor

        img_tuple = self.imageFolderDataset.imgs[index]
        img = self.imageFolderDataset[index][0]
        coor = coordenadas(img_tuple[0])
        room_label = self.imageFolderDataset.targets[index]
        return img, coor, room_label
    
    def __len__(self):
        return len(self.imageFolderDataset.imgs)
    

test_data_cloudy = dset.ImageFolder(root=PARAMS.testing_cloudy_dir, transform=transforms.ToTensor())
test_dataset_cloudy = TestDataset(test_data_cloudy)
test_dataloader_cloudy = DataLoader(test_dataset_cloudy,
                        shuffle=False,
                        num_workers=0,
                        batch_size=1)

test_data_night = dset.ImageFolder(root=PARAMS.testing_night_dir, transform=transforms.ToTensor())
test_dataset_night = TestDataset(test_data_night)
test_dataloader_night = DataLoader(test_dataset_night,
                        shuffle=False,
                        num_workers=0,
                        batch_size=1)

test_data_sunny = dset.ImageFolder(root=PARAMS.testing_sunny_dir, transform=transforms.ToTensor())
test_dataset_sunny = TestDataset(test_data_sunny)
test_dataloader_sunny = DataLoader(test_dataset_sunny,
                        shuffle=False,
                        num_workers=0,
                        batch_size=1)

global_test_dataset = torch.utils.data.ConcatDataset([test_dataset_cloudy, test_dataset_night, test_dataset_sunny])
global_test_dataloader = DataLoader(global_test_dataset,
                        shuffle=False,
                        num_workers=0,
                        batch_size=1)


train_noDA_data = dset.ImageFolder(root=PARAMS.training_noDA_dir, transform=transforms.ToTensor())
train_noDA_dataloader = DataLoader(train_noDA_data,
                        shuffle=True,
                        num_workers=0,
                        batch_size=PARAMS.train_batch_size)

train_DA1_data = dset.ImageFolder(root=PARAMS.training_DA1_dir, transform=transforms.ToTensor())
train_DA1_dataloader = DataLoader(train_DA1_data,
                        shuffle=True,
                        num_workers=0,
                        batch_size=PARAMS.train_batch_size)

train_DA2_data = dset.ImageFolder(root=PARAMS.training_DA2_dir, transform=transforms.ToTensor())
train_DA2_dataloader = DataLoader(train_DA2_data,
                        shuffle=True,
                        num_workers=0,
                        batch_size=PARAMS.train_batch_size)

train_DA3_data = dset.ImageFolder(root=PARAMS.training_DA3_dir, transform=transforms.ToTensor())
train_DA3_dataloader = DataLoader(train_DA3_data,
                        shuffle=True,
                        num_workers=0,
                        batch_size=PARAMS.train_batch_size)

train_DA4_data = dset.ImageFolder(root=PARAMS.training_DA4_dir, transform=transforms.ToTensor())
train_DA4_dataloader = DataLoader(train_DA4_data,
                        shuffle=True,
                        num_workers=0,
                        batch_size=PARAMS.train_batch_size)

train_DA5_data = dset.ImageFolder(root=PARAMS.training_DA5_dir, transform=transforms.ToTensor())
train_DA5_dataloader = DataLoader(train_DA5_data,
                        shuffle=True,
                        num_workers=0,
                        batch_size=PARAMS.train_batch_size)

train_DA6_data = dset.ImageFolder(root=PARAMS.training_DA6_dir, transform=transforms.ToTensor())
train_DA6_dataloader = DataLoader(train_DA6_data,
                        shuffle=True,
                        num_workers=0,
                        batch_size=PARAMS.train_batch_size)

validation_data = dset.ImageFolder(root=PARAMS.validation_dir, transform= transforms.ToTensor())
validation_dataloader = DataLoader(validation_data,
                        shuffle=True,
                        num_workers=0,
                        batch_size=PARAMS.train_batch_size)


map_data = dset.ImageFolder(root=PARAMS.map_dir, transform=transforms.ToTensor())
