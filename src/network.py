import torch
from torch import nn


class MultiTaskResNet18(nn.Module):
    """ResNet18 backbone with classification and bounding-box regression heads."""
    def __init__(self, base_model, num_classes):
        super().__init__()
        in_features = base_model.fc.in_features
        base_model.fc = nn.Identity()
        self.backbone = base_model
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )
        self.regressor = nn.Sequential(
            nn.Linear(in_features, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 4), nn.Sigmoid(),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features), self.regressor(features)
