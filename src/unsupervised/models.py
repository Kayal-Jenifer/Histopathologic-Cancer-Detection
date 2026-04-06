import torch
import torch.nn as nn

import torchvision.models as models


class Autoencoder(nn.Module):
    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),  # 96 -> 48
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 48 -> 24
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 24 -> 12
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.encoder_fc = nn.Sequential(nn.Flatten(), nn.Linear(128 * 12 * 12, latent_dim))
        self.decoder_fc = nn.Sequential(nn.Linear(latent_dim, 128 * 12 * 12), nn.ReLU(inplace=True))
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 12 -> 24
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 24 -> 48
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),  # 48 -> 96
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder_cnn(x)
        z = self.encoder_fc(h)
        return z

    def forward(self, x):
        z = self.encode(x)
        h = self.decoder_fc(z).view(-1, 128, 12, 12)
        x_hat = self.decoder_cnn(h)
        return x_hat, z


class BaselineCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 96 -> 48
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 48 -> 24
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 24 -> 12
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(256, 1)

    def forward(self, x):
        h = self.features(x).flatten(1)
        return self.classifier(h).squeeze(1)


def build_resnet50_feature_extractor() -> nn.Module:
    m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    m.fc = nn.Identity()
    return m
