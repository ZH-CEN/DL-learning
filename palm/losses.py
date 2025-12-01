"""
损失函数模块
包含分类/度量学习常用的损失定义。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ContrastiveLoss(nn.Module):
    """
    基于余弦距离的对比损失 (同/diff 样本对)
    """

    def __init__(self, margin=0.5):
        """
        Args:
            margin: 余弦距离的 margin，范围 [0, 2]
        """
        super().__init__()
        self.margin = margin

    def forward(self, feature1, feature2, label):
        feature1 = F.normalize(feature1, p=2, dim=1)
        feature2 = F.normalize(feature2, p=2, dim=1)
        cosine_similarity = F.cosine_similarity(feature1, feature2)
        cosine_distance = 1 - cosine_similarity
        loss = torch.mean(
            label * torch.pow(cosine_distance, 2)
            + (1 - label) * torch.pow(torch.clamp(self.margin - cosine_distance, min=0.0), 2)
        )
        return loss


class TripletLoss(nn.Module):
    """
    三元组损失：拉近 (anchor, positive)，推远 (anchor, negative)
    """

    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist = F.pairwise_distance(anchor, negative)
        loss = torch.mean(torch.clamp(pos_dist - neg_dist + self.margin, min=0.0))
        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss：处理类别不平衡，聚焦难分类样本。
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits: (N, C), targets: (N,)
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(1)).float()
        pt = (probs * targets_one_hot).sum(dim=1)
        focal_weight = self.alpha * torch.pow(1 - pt, self.gamma)
        loss = -(focal_weight * (log_probs * targets_one_hot).sum(dim=1))
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class MarginLoss(nn.Module):
    """
    Margin Loss：类似 Triplet，对 (anchor, positive, negative) 施加 margin 约束。
    """

    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist = F.pairwise_distance(anchor, negative)
        loss = torch.mean(torch.clamp(pos_dist - neg_dist + self.margin, min=0.0))
        return loss


class MSELoss(nn.Module):
    """
    包装版 MSE，便于统一接口。
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        return F.mse_loss(pred, target, reduction=self.reduction)


class ArcFaceLoss(nn.Module):
    """
    ArcFace / cosine margin 损失。
    参考论文：Additive Angular Margin Loss for Deep Face Recognition
    """

    def __init__(self, num_classes: int, feature_dim: int, margin: float = 0.5, scale: float = 64.0):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.margin = margin
        self.scale = scale
        self.weight = nn.Parameter(torch.randn(num_classes, feature_dim))

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, features: torch.Tensor, labels: torch.Tensor):
        # 归一化特征和权重
        features = F.normalize(features, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)

        # cos(theta)
        cosine = F.linear(features, weight)  # (N, C)
        sine = torch.sqrt(torch.clamp(1.0 - torch.pow(cosine, 2), min=1e-7))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)

        # 仅对目标类添加 margin
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        logits = one_hot * phi + (1.0 - one_hot) * cosine

        # 简单的 easy margin 处理
        logits = torch.where(cosine > self.th, logits, cosine - self.mm)

        logits *= self.scale
        loss = F.cross_entropy(logits, labels)
        return loss
