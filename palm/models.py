"""
模型定义模块
将特征提取、分类头和不同架构拆分成清晰的组件。
"""

from typing import Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv-BN-ReLU 堆叠，可选 2×2 最大池化。"""

    def __init__(self, channels: Iterable[int], use_pool: bool = True):
        super().__init__()
        channels = list(channels)
        assert len(channels) >= 2, "channels 至少包含输入和一个输出通道数"

        layers: List[nn.Module] = []
        for in_c, out_c in zip(channels[:-1], channels[1:]):
            layers.append(nn.Conv2d(in_c, out_c, 3, padding=1))
            layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)
        self.pool = nn.MaxPool2d(2, 2) if use_pool else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        if self.pool is not None:
            x = self.pool(x)
        return x


class FeatureExtractor(nn.Module):
    """
    基础特征提取骨干：128×128 灰度输入 -> 256×16×16 特征图。
    """

    def __init__(self):
        super().__init__()
        self.stages = nn.Sequential(
            ConvBlock([1, 32, 64], use_pool=True),     # 128 -> 64
            ConvBlock([64, 128, 128], use_pool=True),  # 64 -> 32
            ConvBlock([128, 256], use_pool=True),      # 32 -> 16
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stages(x)


class FeatureNet(nn.Module):
    """
    掌纹特征提取网络，输出定长特征向量。
    """

    def __init__(self, feature_dim: int = 128, dropout: float = 0.5):
        super().__init__()
        self.backbone = FeatureExtractor()
        self.projection = nn.Sequential(
            nn.Linear(256 * 16 * 16, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, feature_dim),
        )

    def forward(self, x: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        feats = self.backbone(x)
        feats = torch.flatten(feats, 1)
        feats = self.projection(feats)
        if normalize:
            feats = F.normalize(feats, p=2, dim=1)
        return feats


class PalmClassifier(nn.Module):
    """
    分类模型：FeatureNet + 线性分类头，支持输出特征用于对比/蒸馏。
    """

    def __init__(self, num_classes: int, feature_dim: int = 128, normalize_features: bool = True):
        super().__init__()
        self.normalize_features = normalize_features
        self.backbone = FeatureNet(feature_dim=feature_dim)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        feats = self.backbone(x, normalize=self.normalize_features)
        logits = self.classifier(feats)
        if return_features:
            return logits, feats
        return logits


class INet(FeatureNet):
    """兼容旧接口的别名，等同于 FeatureNet。"""

    def __init__(self, feature_dim: int = 128):
        super().__init__(feature_dim=feature_dim)


class VGG(nn.Module):
    """
    VGG 风格分类网络（保留原始实现，便于对比实验）。
    """

    def __init__(self, num_classes: int = 386):
        super().__init__()

        self.stage1 = ConvBlock([1, 64, 64], use_pool=True)          # 128 -> 64
        self.stage2 = ConvBlock([64, 128, 128], use_pool=True)       # 64 -> 32
        self.stage3 = ConvBlock([128, 256, 256, 256], use_pool=True)  # 32 -> 16
        self.stage4 = ConvBlock([256, 512, 512, 512], use_pool=True)  # 16 -> 8

        self.classifier = nn.Sequential(
            nn.Linear(512 * 8 * 8, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
