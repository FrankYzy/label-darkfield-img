import torchvision
import torch
import torch.nn as nn

model_name = 'vgg11_bn'

# model = torchvision.models.resnet34(pretrained=False, progress=True)
# model.fc = torch.nn.Linear(in_features=512, out_features=2, bias=True)
# model = torchvision.models.alexnet(pretrained=True, progress=True)
# model.classifier[6] = torch.nn.Linear(in_features=4096, out_features=2, bias=True)
model = torchvision.models.vgg11_bn(pretrained=False, progress=True)
model.classifier[6] = torch.nn.Linear(in_features=4096, out_features=3, bias=True)


# model = torchvision.models.inception_v3(pretrained=False,progress=True)
# model.fc = torch.nn.Linear(in_features=2048, out_features=2, bias=True)