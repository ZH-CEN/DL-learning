"""
训练模块
包含分类和度量学习的训练函数。
"""

from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from .models import INet, PalmClassifier, ResNetBackbone, ResNetClassifier
from .mobileone import MobileOne, MobileOneClassifier
from .datasets import PalmDataset, AuthDataset, ContrastivePairDataset, TripletDataset
from .losses import ContrastiveLoss, TripletLoss, MarginLoss, FocalLoss, MSELoss
from .config import get_transform, load_loss_config


def train_classifier(
    cfg,
    data_root,
    epochs=50,
    lr=0.001,
    cache=True,
    num_workers=0,
    save_path="best_classifier.pth",
    loss_type: str = "ce",
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    backbone: str = "inet",
    loss_config_dir: str = "config",
    log_file: str | None = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = get_transform(cfg)

    # 按“手型+人”划分，同一只手的样本按比例切分 train/val
    all_samples = sorted(Path(data_root).glob("*.bmp"))
    id_groups: dict[str, list[Path]] = {}
    for path in all_samples:
        pid = PalmDataset._get_identity(path.name)  # e.g., F_100 / S_100
        id_groups.setdefault(pid, [])
        id_groups[pid].append(path)

    train_paths: list[Path] = []
    val_paths: list[Path] = []
    split_ratio = 0.8
    for paths in id_groups.values():
        paths = sorted(paths)
        if not paths:
            continue
        cutoff = max(1, int(len(paths) * split_ratio))
        if cutoff >= len(paths):
            cutoff = len(paths) - 1  # 至少留一张做验证
        train_paths.extend(paths[:cutoff])
        val_paths.extend(paths[cutoff:])

    # 固定标签映射，避免 train/val 顺序漂移
    all_ids = sorted(id_groups.keys())
    id2idx = {pid: idx for idx, pid in enumerate(all_ids)}

    train_set = PalmDataset(data_root, transform=transform, cache=cache, samples=train_paths, id2idx=id2idx)
    val_set = PalmDataset(data_root, transform=transform, cache=cache, samples=val_paths, id2idx=id2idx)

    batch_size = cfg["train"]["batch_size"]
    train_loader = DataLoader(train_set, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size, shuffle=False, num_workers=num_workers)

    num_classes = len(train_set.id2idx)
    feature_dim = 128

    # 选择分类模型
    backbone = backbone.lower()
    if backbone == "mobileone":
        model = MobileOneClassifier(num_classes=num_classes, feature_dim=feature_dim, normalize_features=True).to(device)
    elif backbone in {"resnet18", "resnet34"}:
        model = ResNetClassifier(
            num_classes=num_classes, feature_dim=feature_dim, name=backbone, normalize_features=True
        ).to(device)
    else:
        model = PalmClassifier(num_classes=num_classes, feature_dim=feature_dim, normalize_features=True).to(device)

    # 加载损失相关配置（config/<loss_type>.yml）
    loss_type = loss_type.lower()
    loss_cfg = load_loss_config(loss_type, loss_config_dir)
    default_epochs = 50
    default_lr = 0.001
    default_focal_alpha = 0.25
    default_focal_gamma = 2.0

    focal_alpha = loss_cfg.get("focal_alpha", focal_alpha if focal_alpha != default_focal_alpha else default_focal_alpha)
    focal_gamma = loss_cfg.get("focal_gamma", focal_gamma if focal_gamma != default_focal_gamma else default_focal_gamma)

    if epochs == default_epochs:
        epochs = loss_cfg.get("epochs", epochs)
    if lr == default_lr:
        lr = loss_cfg.get("learning_rate", lr)

    if loss_type == "focal":
        criterion = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
    elif loss_type == "mse":
        criterion = MSELoss()
    else:  # 默认交叉熵
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=cfg["train"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg["train"]["lr_step_size"], gamma=cfg["train"]["lr_gamma"]
    )

    best_acc = 0.0

    log_file = log_file or Path("logs") / f"train_classifier_{datetime.now():%Y%m%d_%H%M%S}.log"
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    log_f = open(log_file, "a", encoding="utf-8")

    def log(msg: str):
        print(msg)
        log_f.write(msg + "\n")
        log_f.flush()

    log(f"开始训练分类模型 - {num_classes} 个类别")
    log(
        f"超参: epochs={epochs}, lr={lr}, batch_size={batch_size}, loss={loss_type}, "
        f"backbone={backbone}, num_workers={num_workers}, cache={cache}"
    )
    log("模型结构:")
    for line in repr(model).splitlines():
        log(f"  {line}")
    log("=" * 60)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            if loss_type == "mse":
                targets = F.one_hot(labels, num_classes=num_classes).float()
                loss = criterion(outputs, targets)
            else:
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        scheduler.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        log(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Val Acc={acc:.2f}%")

        if acc > best_acc:
            best_acc = acc
            torch.save(
                {
                    "backbone": model.backbone.state_dict(),
                    "classifier": model.classifier.state_dict(),
                    "num_classes": num_classes,
                    "feature_dim": feature_dim,
                    "backbone_type": backbone,
                },
                save_path,
            )
            log(f"✓ 保存最佳模型，准确率: {best_acc:.2f}% -> {save_path}")

    log(f"\n训练完成！最佳准确率: {best_acc:.2f}%")
    log_f.close()
    return {"best_metric": best_acc / 100, "best_path": save_path, "final_loss": avg_loss, "log_file": str(log_file)}


def train_contrastive(
    cfg,
    data_root,
    epochs=50,
    lr=0.001,
    margin=0.5,
    feature_dim=128,
    cache=True,
    num_workers=0,
    save_path="best_contrastive_model.pth",
    loss_type: str = "contrastive",
    loss_config_dir: str = "config",
    backbone: str = "inet",
    batch_size: int | None = None,
    log_file: str | None = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = get_transform(cfg)

    loss_type = loss_type.lower()
    is_pair_loss = loss_type == "contrastive"

    # 覆盖超参：config/<loss_type>.yml 仅在使用默认值时才覆盖，命令行显式传入优先
    loss_cfg = load_loss_config(loss_type, loss_config_dir)
    default_epochs = 50
    default_lr = 0.001
    default_margin = 0.5
    default_feature_dim = 128
    default_batch_size = 32
    default_weight_decay = 1e-4
    default_lr_step = 15
    default_lr_gamma = 0.5

    if margin == default_margin:
        margin = loss_cfg.get("margin", margin)
    if feature_dim == default_feature_dim:
        feature_dim = loss_cfg.get("feature_dim", feature_dim)
    if epochs == default_epochs:
        epochs = loss_cfg.get("epochs", epochs)
    if lr == default_lr:
        lr = loss_cfg.get("learning_rate", lr)

    if batch_size is None:
        batch_size = loss_cfg.get("batch_size", default_batch_size)
    weight_decay = loss_cfg.get("weight_decay", default_weight_decay)
    lr_step_size = loss_cfg.get("lr_step_size", default_lr_step)
    lr_gamma = loss_cfg.get("lr_gamma", default_lr_gamma)

    if is_pair_loss:
        train_set = ContrastivePairDataset(data_root, mode="train", transform=transform, cache=cache)
        test_set = ContrastivePairDataset(data_root, mode="test", transform=transform, cache=cache)
    else:
        train_set = TripletDataset(data_root, mode="train", transform=transform, cache=cache)
        test_set = TripletDataset(data_root, mode="test", transform=transform, cache=cache)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    backbone = backbone.lower()
    if backbone == "mobileone":
        model = MobileOne(feature_dim=feature_dim, normalize=False).to(device)
    elif backbone in {"resnet18", "resnet34"}:
        model = ResNetBackbone(name=backbone, feature_dim=feature_dim, normalize=False).to(device)
    else:
        model = INet(feature_dim=feature_dim).to(device)
    if loss_type == "contrastive":
        criterion = ContrastiveLoss(margin=margin)
    elif loss_type == "triplet":
        criterion = TripletLoss(margin=margin)
    elif loss_type == "margin":
        criterion = MarginLoss(margin=margin)
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)

    best_loss = float("inf")
    best_acc = 0.0

    log_file = log_file or Path("logs") / f"train_contrastive_{datetime.now():%Y%m%d_%H%M%S}.log"
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    log_f = open(log_file, "a", encoding="utf-8")

    def log(msg: str):
        print(msg)
        log_f.write(msg + "\n")
        log_f.flush()

    log(
        f"开始训练度量学习模型 - loss={loss_type}, margin={margin}, feature_dim={feature_dim}, "
        f"batch_size={batch_size}, backbone={backbone}"
    )
    log(
        f"超参: epochs={epochs}, lr={lr}, weight_decay={weight_decay}, "
        f"lr_step={lr_step_size}, lr_gamma={lr_gamma}, num_workers={num_workers}, cache={cache}"
    )
    log("模型结构:")
    for line in repr(model).splitlines():
        log(f"  {line}")
    log("=" * 60)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            if is_pair_loss:
                img1, img2, labels = batch
                img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
                features1 = model(img1)
                features2 = model(img2)
                loss = criterion(features1, features2, labels)
            else:
                anchor, positive, negative = batch
                anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
                f_anchor = model(anchor)
                f_pos = model(positive)
                f_neg = model(negative)
                loss = criterion(f_anchor, f_pos, f_neg)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        scheduler.step()

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        pos_distances = []
        neg_distances = []

        with torch.no_grad():
            for batch in test_loader:
                if is_pair_loss:
                    img1, img2, labels = batch
                    img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

                    features1 = model(img1)
                    features2 = model(img2)

                    loss = criterion(features1, features2, labels)
                    val_loss += loss.item()

                    features1_norm = F.normalize(features1, p=2, dim=1)
                    features2_norm = F.normalize(features2, p=2, dim=1)
                    cosine_similarity = F.cosine_similarity(features1_norm, features2_norm)
                    cosine_distance = 1 - cosine_similarity

                    for dist, label in zip(cosine_distance, labels):
                        if label == 1.0:
                            pos_distances.append(dist.item())
                        else:
                            neg_distances.append(dist.item())

                    threshold = margin / 2
                    predictions = (cosine_distance < threshold).float()
                    correct += (predictions == labels).sum().item()
                    total += labels.size(0)
                else:
                    anchor, positive, negative = batch
                    anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
                    f_anchor = model(anchor)
                    f_pos = model(positive)
                    f_neg = model(negative)
                    loss = criterion(f_anchor, f_pos, f_neg)
                    val_loss += loss.item()

                    pos_dist = F.pairwise_distance(f_anchor, f_pos)
                    neg_dist = F.pairwise_distance(f_anchor, f_neg)
                    pos_distances.extend(pos_dist.tolist())
                    neg_distances.extend(neg_dist.tolist())
                    correct += (pos_dist < neg_dist).sum().item()
                    total += pos_dist.numel()

        val_loss /= len(test_loader)
        val_acc = 100 * correct / total if total > 0 else 0

        avg_pos_dist = sum(pos_distances) / len(pos_distances) if pos_distances else 0
        avg_neg_dist = sum(neg_distances) / len(neg_distances) if neg_distances else 0

        log(f"Epoch {epoch+1}: Train Loss={avg_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")
        log(f"  正样本距离: {avg_pos_dist:.4f}, 负样本距离: {avg_neg_dist:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_acc = val_acc
            torch.save(
                {
                    "model": model.state_dict(),
                    "feature_dim": feature_dim,
                    "margin": margin,
                    "loss_type": loss_type,
                    "backbone": backbone,
                },
                save_path,
            )
            log(f"✓ 保存最佳模型，验证损失: {best_loss:.4f}, 准确率: {val_acc:.2f}% -> {save_path}")

    log(f"\n训练完成！最佳验证损失: {best_loss:.4f}, 准确率: {best_acc:.2f}%")
    log_f.close()
    return {"best_metric": best_loss, "best_path": save_path, "best_acc": best_acc / 100, "log_file": str(log_file)}
