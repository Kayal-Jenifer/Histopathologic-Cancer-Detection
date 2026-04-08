from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float, use_batchnorm: bool):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.extend(
            [
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
                nn.Dropout2d(p=dropout),
            ]
        )
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CustomCNN(nn.Module):
    def __init__(
        self,
        filters: tuple[int, ...] = (32, 64, 128, 256),
        dropout: float = 0.4,
        use_batchnorm: bool = True,
    ):
        super().__init__()

        blocks: list[nn.Module] = []
        in_channels = 3
        for out_channels in filters:
            block_dropout = dropout * 0.5 if out_channels < 128 else dropout
            blocks.append(ConvBlock(in_channels, out_channels, block_dropout, use_batchnorm))
            in_channels = out_channels

        self.features = nn.Sequential(*blocks, nn.AdaptiveAvgPool2d((1, 1)))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x).squeeze(1)
