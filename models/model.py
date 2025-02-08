import torch
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn


class MyResNet(nn.Module):
    def __init__(self):
        super().__init__()

        # подгружаем модель
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        # заменяем слой
        self.model.fc = nn.LazyLinear(10)

        # # замораживаем слои
        # for i in self.model.parameters():
        #     i.requires_grad = False

        # размораживаем только последний, который будем обучать
        # self.model.fc.weight.requires_grad = True
        # self.model.fc.bias.requires_grad = True

    def forward(self, x):
        return self.model(x)