"""
训练模块
包含各种训练函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from .models import INet
from .datasets import PalmDataset, AuthDataset, ContrastivePairDataset
from .losses import ContrastiveLoss
from .config import get_transform


def train_classifier(cfg, data_root, epochs=50, lr=0.001, cache=False, num_workers=0, save_path="best_classifier.pth"):
    """
    训练分类模型
    
    Args:
        cfg: 配置字典
        data_root: 数据集根目录
        epochs: 训练轮数
        lr: 学习率
        cache: 是否缓存数据
        num_workers: 数据加载线程数
        save_path: 模型保存路径
        
    Returns:
        训练结果字典
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = get_transform(cfg)
    
    # 创建数据集
    dataset = PalmDataset(data_root, transform=transform, cache=cache)
    train_len = int(len(dataset) * 0.8)
    val_len = len(dataset) - train_len
    train_set, val_set = random_split(dataset, [train_len, val_len])
    
    batch_size = cfg["train"]["batch_size"]
    train_loader = DataLoader(train_set, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size, shuffle=False, num_workers=num_workers)
    
    # 创建模型
    num_classes = len(dataset.id2idx)
    model = INet(feature_dim=128).to(device)
    classifier = nn.Linear(128, num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(classifier.parameters()),
        lr=lr,
        weight_decay=cfg["train"]["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg["train"]["lr_step_size"],
        gamma=cfg["train"]["lr_gamma"]
    )
    
    best_acc = 0.0
    
    print(f"开始训练分类模型 - {num_classes} 个类别")
    print("="*60)
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        classifier.train()
        running_loss = 0.0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            features = model(inputs)
            features_norm = F.normalize(features, p=2, dim=1)
            outputs = classifier(features_norm)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        scheduler.step()
        
        # 验证阶段
        model.eval()
        classifier.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                features = model(inputs)
                features_norm = F.normalize(features, p=2, dim=1)
                outputs = classifier(features_norm)
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = 100 * correct / total
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Val Acc={acc:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), save_path)
            print(f"✓ 保存最佳模型，准确率: {best_acc:.2f}%")
    
    print(f"\n训练完成！最佳准确率: {best_acc:.2f}%")
    
    return {
        "best_metric": best_acc / 100,
        "best_path": save_path,
        "final_loss": avg_loss
    }


def train_contrastive(cfg, data_root, epochs=50, lr=0.001, margin=0.5, feature_dim=128, 
                      num_workers=0, save_path="best_contrastive_model.pth"):
    """
    使用对比损失训练认证模型
    
    Args:
        cfg: 配置字典
        data_root: 数据集根目录
        epochs: 训练轮数
        lr: 学习率
        margin: 对比损失的margin
        feature_dim: 特征维度
        num_workers: 数据加载线程数
        save_path: 模型保存路径
        
    Returns:
        训练结果字典
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = get_transform(cfg)
    
    # 创建数据集
    train_set = ContrastivePairDataset(data_root, mode='train', transform=transform)
    test_set = ContrastivePairDataset(data_root, mode='test', transform=transform)
    
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=num_workers)
    
    # 创建模型
    model = INet(feature_dim=feature_dim).to(device)
    criterion = ContrastiveLoss(margin=margin)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    
    best_loss = float('inf')
    best_acc = 0.0
    
    print(f"开始训练对比损失模型 - margin={margin}, feature_dim={feature_dim}")
    print("="*60)
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        
        for img1, img2, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            features1 = model(img1)
            features2 = model(img2)
            
            loss = criterion(features1, features2, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        scheduler.step()
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        pos_distances = []
        neg_distances = []
        
        with torch.no_grad():
            for img1, img2, labels in test_loader:
                img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
                
                features1 = model(img1)
                features2 = model(img2)
                
                loss = criterion(features1, features2, labels)
                val_loss += loss.item()
                
                # 计算准确率
                features1_norm = F.normalize(features1, p=2, dim=1)
                features2_norm = F.normalize(features2, p=2, dim=1)
                cosine_similarity = F.cosine_similarity(features1_norm, features2_norm)
                cosine_distance = 1 - cosine_similarity
                
                # 收集距离统计
                for dist, label in zip(cosine_distance, labels):
                    if label == 1.0:
                        pos_distances.append(dist.item())
                    else:
                        neg_distances.append(dist.item())
                
                # 使用阈值判断
                threshold = 0.5
                predictions = (cosine_distance < threshold).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        val_loss /= len(test_loader)
        val_acc = 100 * correct / total
        
        avg_pos_dist = sum(pos_distances) / len(pos_distances) if pos_distances else 0
        avg_neg_dist = sum(neg_distances) / len(neg_distances) if neg_distances else 0
        
        print(f"Epoch {epoch+1}: Train Loss={avg_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")
        print(f"  正样本距离: {avg_pos_dist:.4f}, 负样本距离: {avg_neg_dist:.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"✓ 保存最佳模型，验证损失: {best_loss:.4f}, 准确率: {val_acc:.2f}%")
    
    print(f"\n训练完成！最佳验证损失: {best_loss:.4f}, 准确率: {best_acc:.2f}%")
    
    return {
        "best_metric": best_loss,
        "best_path": save_path,
        "best_acc": best_acc / 100
    }
