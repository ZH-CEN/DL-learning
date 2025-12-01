"""
MobileOne 模型实现（可重参数化），以及基于其的 Siamese/分类封装。
"""

from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class MobileOneBlock(nn.Module):
    """
    MobileOne 基础块：训练阶段包含 3x3、1x1、identity 分支，推理阶段可合并为单卷积。
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2

        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.use_identity = in_channels == out_channels and stride == 1
        if self.use_identity:
            self.id_bn = nn.BatchNorm2d(out_channels)

        self.reparam_conv: Optional[nn.Conv2d] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.reparam_conv is not None:
            return F.relu(self.reparam_conv(x))

        out = self.bn3(self.conv3(x)) + self.bn1(self.conv1(x))
        if self.use_identity:
            out += self.id_bn(x)
        return F.relu(out)

    def reparameterize(self) -> None:
        """将多分支折叠为单卷积，便于部署推理。"""
        if self.reparam_conv is not None:
            return

        kernel, bias = self._merge_kernels()
        self.reparam_conv = nn.Conv2d(
            in_channels=kernel.size(1),
            out_channels=kernel.size(0),
            kernel_size=kernel.size(2),
            padding=kernel.size(2) // 2,
            bias=True,
        )
        self.reparam_conv.weight.data = kernel
        self.reparam_conv.bias.data = bias

        # 删除训练分支
        del self.conv3, self.bn3
        del self.conv1, self.bn1
        if self.use_identity:
            del self.id_bn

    def _merge_kernels(self) -> tuple[torch.Tensor, torch.Tensor]:
        k3, b3 = self._fuse_conv_bn(self.conv3, self.bn3)
        k1, b1 = self._fuse_conv_bn(self.conv1, self.bn1)
        kernel = k3 + F.pad(k1, [1, 1, 1, 1])

        if self.use_identity:
            kid, bid = self._fuse_identity_bn(self.id_bn, kernel.size(1))
            kernel += kid
            bias = b3 + b1 + bid
        else:
            bias = b3 + b1
        return kernel, bias

    @staticmethod
    def _fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
        std = torch.sqrt(bn.running_var + bn.eps)
        weight = conv.weight * (bn.weight / std).reshape(-1, 1, 1, 1)
        bias = bn.bias - bn.running_mean * bn.weight / std
        return weight, bias

    @staticmethod
    def _fuse_identity_bn(bn: nn.BatchNorm2d, channels: int) -> tuple[torch.Tensor, torch.Tensor]:
        weight = torch.zeros((channels, channels, 3, 3), device=bn.weight.device, dtype=bn.weight.dtype)
        for c in range(channels):
            weight[c, c, 1, 1] = 1.0
        std = torch.sqrt(bn.running_var + bn.eps)
        weight = weight * (bn.weight / std).reshape(-1, 1, 1, 1)
        bias = bn.bias - bn.running_mean * bn.weight / std
        return weight, bias


class MobileOne(nn.Module):
    """
    MobileOne 主干（简化版）。输入：1x128x128 灰度图；输出：feature_dim 向量。
    """

    def __init__(
        self,
        feature_dim: int = 128,
        num_blocks: Sequence[int] = (1, 2, 4),
        base_channels: int = 64,
        normalize: bool = False,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        in_channels = 1
        out_channels = base_channels

        for nb in num_blocks:
            for _ in range(nb):
                layers.append(MobileOneBlock(in_channels, out_channels))
                in_channels = out_channels
            out_channels *= 2

        self.features = nn.Sequential(*layers)
        self.fc = nn.Linear(in_channels, feature_dim)
        self.normalize = normalize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.mean(dim=[2, 3])  # GAP
        x = self.fc(x)
        if self.normalize:
            x = F.normalize(x, p=2, dim=1)
        return x


class MobileOneClassifier(nn.Module):
    """基于 MobileOne 的分类模型。"""

    def __init__(self, num_classes: int, feature_dim: int = 128, normalize_features: bool = True):
        super().__init__()
        self.normalize_features = normalize_features
        self.backbone = MobileOne(feature_dim=feature_dim, normalize=False)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        feats = self.backbone(x)
        if self.normalize_features:
            feats = F.normalize(feats, p=2, dim=1)
        logits = self.classifier(feats)
        if return_features:
            return logits, feats
        return logits


class SiameseMobileOne(nn.Module):
    """Siamese 结构，两个分支共享 MobileOne。"""

    def __init__(self, feature_dim: int = 128):
        super().__init__()
        self.backbone = MobileOne(feature_dim=feature_dim, normalize=True)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.backbone(x1), self.backbone(x2)


def reparameterize_model(model: nn.Module) -> None:
    """遍历模型，将所有 MobileOneBlock 折叠为单卷积。"""
    for module in model.modules():
        if isinstance(module, MobileOneBlock):
            module.reparameterize()
