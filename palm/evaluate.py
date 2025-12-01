"""
评估模块
包含模型评估和认证功能
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm


def extract_features(model, dataloader):
    """
    提取特征向量
    
    Args:
        model: 特征提取模型
        dataloader: 数据加载器
        
    Returns:
        features: 特征张量 (N, feature_dim)
        labels: 标签张量 (N,)
    """
    model.eval()
    device = next(model.parameters()).device
    
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, labels in tqdm(dataloader, desc="提取特征"):
            imgs = imgs.to(device)
            features = model(imgs)
            features = F.normalize(features, p=2, dim=1)
            
            all_features.append(features.cpu())
            all_labels.append(labels)
    
    return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)


def build_gallery(model, dataloader, log_func=print):
    """
    构建特征画廊（每个身份的平均特征）
    
    Args:
        model: 特征提取模型
        dataloader: 数据加载器
        log_func: 日志函数
        
    Returns:
        gallery_features: 画廊特征 (num_identities, feature_dim)
        identity_ids: 身份ID列表
    """
    log_func("构建特征画廊...")
    features, labels = extract_features(model, dataloader)
    
    # 对每个身份计算平均特征
    unique_labels = torch.unique(labels)
    avg_features = []
    
    for label in unique_labels:
        mask = labels == label
        avg_feat = features[mask].mean(dim=0, keepdim=True)
        avg_features.append(avg_feat)

    gallery_features = torch.cat(avg_features, dim=0)
    # 均值后再归一化，避免模长缩小导致距离被低估
    gallery_features = F.normalize(gallery_features, p=2, dim=1)

    return gallery_features, unique_labels.tolist()


def evaluate_authentication(model, gallery_loader, query_loader, threshold=0.6, log_func=print):
    """
    评估认证模型的FAR和FRR
    
    Args:
        model: 特征提取模型
        gallery_loader: 画廊数据加载器
        query_loader: 查询数据加载器
        threshold: 认证阈值
        log_func: 日志函数
    """
    device = next(model.parameters()).device
    
    # 构建画廊
    gallery_features, _ = build_gallery(model, gallery_loader, log_func=log_func)
    gallery_features = gallery_features.to(device)
    
    log_func(f"\n使用阈值 {threshold} 进行认证测试...")
    
    genuine_scores = []  # 真实匹配的距离
    impostor_scores = []  # 冒充者的距离
    
    # 提取查询特征
    query_features, query_labels = extract_features(model, query_loader)
    query_features = query_features.to(device)
    
    for feat, label in tqdm(zip(query_features, query_labels), total=len(query_labels), desc="认证测试"):
        # 计算余弦距离（与训练时一致）
        cosine_sim = F.cosine_similarity(feat.unsqueeze(0), gallery_features, dim=1)
        distances = 1 - cosine_sim  # 距离越小越相似
        min_dist, matched_idx = torch.min(distances, dim=0)
        
        if matched_idx.item() == label.item():
            genuine_scores.append(min_dist.item())
        else:
            impostor_scores.append(min_dist.item())
    
    # 计算FAR和FRR
    genuine_scores = torch.tensor(genuine_scores)
    impostor_scores = torch.tensor(impostor_scores)

    def compute_metrics(th):
        frr = (genuine_scores > th).float().mean().item() * 100  # 误拒率
        far = (impostor_scores < th).float().mean().item() * 100  # 误识率
        return far, frr

    # 自动寻找 FAR≈FRR 的阈值，便于参考
    best_th = threshold
    best_gap = float("inf")
    if len(genuine_scores) > 0 and len(impostor_scores) > 0:
        all_scores = torch.cat([genuine_scores, impostor_scores])
        scan_thresholds = torch.linspace(all_scores.min(), all_scores.max(), steps=200)
        for th in scan_thresholds:
            far_tmp, frr_tmp = compute_metrics(th)
            gap = abs(far_tmp - frr_tmp)
            if gap < best_gap:
                best_gap = gap
                best_th = th.item()
    far, frr = compute_metrics(threshold)
    
    log_func(f"\n认证性能评估:")
    log_func(f"阈值: {threshold}")
    log_func(f"FRR (误拒率): {frr:.2f}%")
    log_func(f"FAR (误识率): {far:.2f}%")
    log_func(f"平均真实距离: {genuine_scores.mean():.4f}")
    log_func(f"平均冒充距离: {impostor_scores.mean():.4f}")
    log_func(f"推荐阈值(近似EER): {best_th:.4f} (gap={best_gap:.4f})")
    log_func("="*60)
    
    return {
        "FRR": frr,
        "FAR": far,
        "genuine_mean": genuine_scores.mean().item(),
        "impostor_mean": impostor_scores.mean().item(),
        "best_threshold": best_th,
        "best_gap": best_gap,
    }


def authenticate_single(model, query_img, gallery_features, threshold=0.6):
    """
    单张图像认证
    
    Args:
        model: 特征提取模型
        query_img: 查询图像张量 (1, C, H, W)
        gallery_features: 画廊特征 (N, feature_dim)
        threshold: 认证阈值
        
    Returns:
        is_authenticated: 是否认证成功
        min_distance: 最小距离
        matched_idx: 匹配的索引
    """
    model.eval()
    device = next(model.parameters()).device
    
    with torch.no_grad():
        query_img = query_img.to(device)
        query_feature = model(query_img)
        query_feature = F.normalize(query_feature, p=2, dim=1)
        
        # 计算与画廊的距离
        distances = torch.cdist(query_feature, gallery_features.to(device)).squeeze(0)
        min_distance, matched_idx = torch.min(distances, dim=0)
        
        is_authenticated = min_distance.item() < threshold
    
    return is_authenticated, min_distance.item(), matched_idx.item()
