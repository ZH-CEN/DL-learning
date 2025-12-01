"""
损失函数模块
包含各种损失函数的定义
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    基于余弦相似度的对比损失
    """
    
    def __init__(self, margin=0.5):
        """
        Args:
            margin: 余弦距离的margin，范围[0, 2]
                   对于余弦相似度[-1, 1]，转换为余弦距离[0, 2]
        """
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, feature1, feature2, label):
        """
        Args:
            feature1: 第一个特征向量 (batch_size, feature_dim)
            feature2: 第二个特征向量 (batch_size, feature_dim)
            label: 标签，1表示相同身份，0表示不同身份 (batch_size,)
            
        Returns:
            loss: 对比损失值
        """
        # L2归一化
        feature1 = F.normalize(feature1, p=2, dim=1)
        feature2 = F.normalize(feature2, p=2, dim=1)
        
        # 计算余弦相似度
        cosine_similarity = F.cosine_similarity(feature1, feature2)
        
        # 转换为余弦距离 [0, 2]
        cosine_distance = 1 - cosine_similarity
        
        # 对比损失
        loss = torch.mean(
            label * torch.pow(cosine_distance, 2) +
            (1 - label) * torch.pow(torch.clamp(self.margin - cosine_distance, min=0.0), 2)
        )
        return loss


class TripletLoss(nn.Module):
    """
    三元组损失
    """
    
    def __init__(self, margin=0.5):
        """
        Args:
            margin: 边界值
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        """
        Args:
            anchor: 锚点特征 (batch_size, feature_dim)
            positive: 正样本特征 (batch_size, feature_dim)
            negative: 负样本特征 (batch_size, feature_dim)
            
        Returns:
            loss: 三元组损失值
        """
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist = F.pairwise_distance(anchor, negative)
        loss = torch.mean(torch.clamp(pos_dist - neg_dist + self.margin, min=0.0))
        return loss
