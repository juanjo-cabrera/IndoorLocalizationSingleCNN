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
from torchvision.models import resnet50,resnet152, convnext_large, resnext101_64x4d, efficientnet_v2_l, mobilenet_v3_large, alexnet

alexnet_noDA = alexnet(pretrained=True)
alexnet_noDA.classifier[6] = nn.Linear(4096, 9)

alexnet_DA1 = alexnet(pretrained=True)
alexnet_DA1.classifier[6] = nn.Linear(4096, 9)

alexnet_DA2 = alexnet(pretrained=True)
alexnet_DA2.classifier[6] = nn.Linear(4096, 9)

alexnet_DA3 = alexnet(pretrained=True)
alexnet_DA3.classifier[6] = nn.Linear(4096, 9)

alexnet_DA4 = alexnet(pretrained=True)
alexnet_DA4.classifier[6] = nn.Linear(4096, 9)

alexnet_DA5 = alexnet(pretrained=True)
alexnet_DA5.classifier[6] = nn.Linear(4096, 9)

alexnet_DA6 = alexnet(pretrained=True)
alexnet_DA6.classifier[6] = nn.Linear(4096, 9)

resnet_50 = resnet50(weights='IMAGENET1K_V2')
resnet_50.fc = nn.Linear(2048, 9)


resnet_152_noDA = resnet152(weights='IMAGENET1K_V2')
resnet_152_noDA.fc = nn.Linear(2048, 9)

resnet_152_DA1 = resnet152(weights='IMAGENET1K_V2')
resnet_152_DA1.fc = nn.Linear(2048, 9)

resnet_152_DA2 = resnet152(weights='IMAGENET1K_V2')
resnet_152_DA2.fc = nn.Linear(2048, 9)

resnet_152_DA3 = resnet152(weights='IMAGENET1K_V2')
resnet_152_DA3.fc = nn.Linear(2048, 9)

resnet_152_DA4 = resnet152(weights='IMAGENET1K_V2')
resnet_152_DA4.fc = nn.Linear(2048, 9)

resnet_152_DA5 = resnet152(weights='IMAGENET1K_V2')
resnet_152_DA5.fc = nn.Linear(2048, 9)

resnet_152_DA6 = resnet152(weights='IMAGENET1K_V2')
resnet_152_DA6.fc = nn.Linear(2048, 9)

convnext_noDA = convnext_large(weights='IMAGENET1K_V1')
convnext_noDA.classifier[2] = nn.Linear(1536, 9)

convnext_DA1 = convnext_large(weights='IMAGENET1K_V1')
convnext_DA1.classifier[2] = nn.Linear(1536, 9)



resnext = resnext101_64x4d(weights='IMAGENET1K_V1')
resnext.fc = nn.Linear(2048, 9)

efficientnet = efficientnet_v2_l(weights='IMAGENET1K_V1')
efficientnet.classifier[1] = nn.Linear(1280, 9)

mobilenet = mobilenet_v3_large(weights='IMAGENET1K_V1')
mobilenet.classifier[3] = nn.Linear(1280, 9)