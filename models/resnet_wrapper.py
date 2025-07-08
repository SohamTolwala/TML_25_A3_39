import torch.nn as nn
import torchvision.models as models

class ResNetWrapper(nn.Module):
    def __init__(self, base='resnet18', num_classes=10):
        super().__init__()
        assert base in ['resnet18', 'resnet34', 'resnet50']
        self.backbone = getattr(models, base)(pretrained=False)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)
