import torch
import torch.nn as nn
from torchvision import models

_model_instance = None

def get_model():
    global _model_instance
    if _model_instance is None:
        # Load a pre-trained ResNet18 and overwrite the last layer
        _model_instance = models.resnet18(pretrained=True)
        num_ftrs = _model_instance.fc.in_features
        _model_instance.fc = nn.Linear(num_ftrs, 3)
        _model_instance.eval()
    return _model_instance
