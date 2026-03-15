import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """Conv → BatchNorm → ReLU → MaxPool (optional)"""
    def __init__(self, in_c, out_c, pool=True):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()

    def forward(self, x):
        x = self.pool(F.relu(self.bn(self.conv(x))))
        return x

class HandwritingDeepCNN(nn.Module):
    """
    Deeper CNN model with BatchNorm and dropout.
    Input: 1 x 64 x 64
    Output: 62-class prediction
    """
    def __init__(self, num_classes: int = 62):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(1, 64),       # 64x32x32
            ConvBlock(64, 128),     # 128x16x16
            ConvBlock(128, 256),    # 256x8x8
            ConvBlock(256, 256),    # 256x4x4
            ConvBlock(256, 256, pool=False),  # 256x4x4
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 256x1x1
            nn.Flatten(),             # 256
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
