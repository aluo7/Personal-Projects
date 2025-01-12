# resnet_3d.py

import torch
import torch.nn as nn
from torchvision.models.video import r3d_18
from torchvision.models.video import R3D_18_Weights

def get_resnet_3d_model(pretrained=True, num_classes=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if pretrained:
        weights = R3D_18_Weights.DEFAULT
        model = r3d_18(weights=weights)
    else:
        model = r3d_18()
    
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    model.eval()
    
    return model
