import torch
import torch.nn as nn
from torchvision import models


class SimpleEncoder(nn.Module):
    """The v1 3-block encoder. Do not change — the entire paper depends on this."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten()
        )

    def forward(self, x):
        return self.features(x)


class SSLClassifier(nn.Module):
    """The v1 classifier head. Do not change."""
    def __init__(self, encoder, num_classes):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.encoder(x))


class ResNetClassifier(nn.Module):
    """The v1 ResNet-18 baseline. Kept for external comparison."""
    def __init__(self, num_classes, freeze_early=True):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        if freeze_early:
            for param in list(self.encoder.parameters())[:30]:
                param.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        features = self.encoder(x).squeeze(-1).squeeze(-1)
        return self.classifier(features)


class SimCLR(nn.Module):
    """The v1 SimCLR wrapper. Projector is discarded after pretraining."""
    def __init__(self, encoder, projection_dim=64):
        super().__init__()
        self.encoder = encoder
        self.projector = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, projection_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        z = nn.functional.normalize(self.projector(h), dim=1)
        return h, z