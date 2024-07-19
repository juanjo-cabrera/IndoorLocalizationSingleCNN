import torch
import torch.nn as nn
from torchvision.models import resnet50,resnet152, convnext_large, resnext101_64x4d, efficientnet_v2_l, mobilenet_v3_large, alexnet
from thop import profile
from config import PARAMS

alexnet_noDA = alexnet(pretrained=True)
alexnet_noDA.classifier[6] = nn.Linear(4096, PARAMS.num_classes)

alexnet_DA1 = alexnet(pretrained=True)
alexnet_DA1.classifier[6] = nn.Linear(4096, PARAMS.num_classes)


alexnet_DA2 = alexnet(pretrained=True)
alexnet_DA2.classifier[6] = nn.Linear(4096,  PARAMS.num_classes)


alexnet_DA3 = alexnet(pretrained=True)
alexnet_DA3.classifier[6] = nn.Linear(4096,  PARAMS.num_classes)


alexnet_DA4 = alexnet(pretrained=True)
alexnet_DA4.classifier[6] = nn.Linear(4096,  PARAMS.num_classes)


alexnet_DA5 = alexnet(pretrained=True)
alexnet_DA5.classifier[6] = nn.Linear(4096,  PARAMS.num_classes)

alexnet_DA6 = alexnet(pretrained=True)
alexnet_DA6.classifier[6] = nn.Linear(4096,  PARAMS.num_classes)


resnet_50 = resnet50(weights='IMAGENET1K_V2')
resnet_50.fc = nn.Linear(2048,  PARAMS.num_classes)


resnet_152_noDA = resnet152(weights='IMAGENET1K_V2')
resnet_152_noDA.fc = nn.Linear(2048,  PARAMS.num_classes)


resnet_152_DA1 = resnet152(weights='IMAGENET1K_V2')
resnet_152_DA1.fc = nn.Linear(2048,  PARAMS.num_classes)


resnet_152_DA2 = resnet152(weights='IMAGENET1K_V2')
resnet_152_DA2.fc = nn.Linear(2048,  PARAMS.num_classes)


resnet_152_DA3 = resnet152(weights='IMAGENET1K_V2')
resnet_152_DA3.fc = nn.Linear(2048,  PARAMS.num_classes)


resnet_152_DA4 = resnet152(weights='IMAGENET1K_V2')
resnet_152_DA4.fc = nn.Linear(2048,  PARAMS.num_classes)


resnet_152_DA5 = resnet152(weights='IMAGENET1K_V2')
resnet_152_DA5.fc = nn.Linear(2048,  PARAMS.num_classes)


resnet_152_DA6 = resnet152(weights='IMAGENET1K_V2')
resnet_152_DA6.fc = nn.Linear(2048,  PARAMS.num_classes)


convnext_noDA = convnext_large(weights='IMAGENET1K_V1')
convnext_noDA.classifier[2] = nn.Linear(1536,  PARAMS.num_classes)


convnext_DA1 = convnext_large(weights='IMAGENET1K_V1')
convnext_DA1.classifier[2] = nn.Linear(1536,  PARAMS.num_classes)




resnext = resnext101_64x4d(weights='IMAGENET1K_V1')
resnext.fc = nn.Linear(2048,  PARAMS.num_classes)


efficientnet = efficientnet_v2_l(weights='IMAGENET1K_V1')
efficientnet.classifier[1] = nn.Linear(1280,  PARAMS.num_classes)


mobilenet = mobilenet_v3_large(weights='IMAGENET1K_V1')
mobilenet.classifier[3] = nn.Linear(1280,  PARAMS.num_classes)


input = torch.randn(1, 3, 512, 128)
flops, params = profile(alexnet_noDA, inputs=(input, ))
print(f" ####Model: AlexNet, FLOPs: {flops/ 1e9} G, Parameters: {params}")

flops, params = profile(resnet_50, inputs=(input, ))
print(f" ####Model: ResNet-50, FLOPs: {flops/ 1e9} G, Parameters: {params}")

flops, params = profile(resnet_152_DA1, inputs=(input, ))
print(f" ####Model: ResNet-152, FLOPs: {flops/ 1e9} G, Parameters: {params}")

flops, params = profile(convnext_DA1, inputs=(input, ))
print(f" ####Model: ConvNext-L, FLOPs: {flops/ 1e9} G, Parameters: {params}")

flops, params = profile(resnext, inputs=(input, ))
print(f" ####Model: ResNext, FLOPs: {flops/ 1e9} G, Parameters: {params}")

flops, params = profile(efficientnet, inputs=(input, ))
print(f" ####Model: EfficientNet, FLOPs: {flops/ 1e9} G, Parameters: {params}")

flops, params = profile(mobilenet, inputs=(input, ))
print(f" ####Model: MobileNet, FLOPs: {flops/ 1e9} G, Parameters: {params}")



# Para obtener el tamaño de los parámetros
param_size = sum(p.numel() * p.element_size() for p in convnext_DA1.parameters())
param_size_MB = param_size / (1024 ** 2)

# Para obtener el tamaño de las activaciones (requiere un forward pass)
# dummy_input = torch.randn(1, 3, 224, 224)  # ajustar según la entrada de tu modelo
with torch.no_grad():
    _ = convnext_DA1(input)

activation_size = torch.cuda.memory_allocated()  # Obtiene memoria de activaciones
activation_size_MB = activation_size / (1024 ** 2)

# Estimación total
total_size_MB = param_size_MB + activation_size_MB
print(f'Tamaño total de la memoria estimada: {total_size_MB} MB')

