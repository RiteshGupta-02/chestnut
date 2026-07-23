import torch
import torch.nn as nn
import torchvision.models as models
from chestxray_dataset import ChestXray14Dataset, get_transforms, get_dataloaders
# from exception import CustomException
# from logger import logging

class cheXNet(nn.Module):
    def __init__(self,num_classes = 14):  # 14 are the number of dieseases
        super().__init__()
        self.model = models.densenet121(weights = models.DenseNet121_Weights.DEFAULT)
        self.model.classifier = nn.Linear(self.model.classifier.in_features,num_classes)
        

    def forward(self, x):
        output = self.model(x)
        # output = torch.sigmoid(output)
        return output

def get_model(num_classes = 14,device = 'cpu'):
    model=cheXNet(num_classes)
    model.to(device)
    return model

if __name__ == "__main__":
    model = get_model(num_classes=14, device='cpu')
    print(model)
    dummy = torch.randn(1, 3, 224, 224)  # fake one image
    output = model(dummy)
    print(output.shape)  # should print torch.Size([1, 14])
    print(output)        # 14 numbers all between 0 and 1