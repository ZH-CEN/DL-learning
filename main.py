"""
一键运行脚本：顺序完成训练与评估（分类 + 对比认证）。
无命令行参数，默认数据路径 PalmBigDataBase，可按需修改下面的常量。
"""

import torch
from torch.utils.data import DataLoader

from palm.config import load_config, set_seed, get_transform
from palm.datasets import AuthDataset
from palm.models import FeatureNet
from palm.trainer import train_classifier, train_contrastive
from palm.evaluate import evaluate_authentication

# 可修改的运行参数
DATA_ROOT = "PalmBigDataBase"
NUM_WORKERS = 0  # Windows 建议 0
FEATURE_DIM = 128
MARGIN = 0.5
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]
USE_CACHE = False  # 分类数据是否缓存到内存
SEED = 42


def run_all() -> None:
    cfg = load_config()
    set_seed(SEED)

    # 1) 分类训练
    cls_result = train_classifier(
        cfg=cfg,
        data_root=DATA_ROOT,
        cache=USE_CACHE,
        num_workers=NUM_WORKERS,
        save_path="best_classifier.pth",
    )
    print(f"[DONE] 分类最佳 acc={cls_result['best_metric']*100:.2f}% -> {cls_result['best_path']}")

    # 2) 认证训练（对比损失，可在 trainer 里切换 loss_type）
    contrastive_result = train_contrastive(
        cfg=cfg,
        data_root=DATA_ROOT,
        feature_dim=FEATURE_DIM,
        margin=MARGIN,
        num_workers=NUM_WORKERS,
        save_path="best_contrastive_model.pth",
        loss_type="contrastive",
    )
    print(
        f"[DONE] 认证(Contrastive) 最佳 val_loss={contrastive_result['best_metric']:.4f} "
        f"-> {contrastive_result['best_path']}"
    )

    # 3) 认证评估（加载上一步最优权重，输出 FAR/FRR）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = get_transform(cfg)
    gallery = AuthDataset(DATA_ROOT, mode="train", transform=transform)
    query = AuthDataset(DATA_ROOT, mode="test", transform=transform)
    gallery_loader = DataLoader(gallery, batch_size=cfg["train"]["batch_size"], shuffle=False)
    query_loader = DataLoader(query, batch_size=cfg["test"]["batch_size"], shuffle=False)

    model = FeatureNet(feature_dim=FEATURE_DIM).to(device)
    state = torch.load(contrastive_result["best_path"], map_location=device)
    model.load_state_dict(state)

    for thr in THRESHOLDS:
        evaluate_authentication(model, gallery_loader, query_loader, threshold=thr)


if __name__ == "__main__":
    run_all()
