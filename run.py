"""
Palm Recognition - 掌纹识别系统
主程序入口

用法示例：
    # 训练分类模型
    python run.py --mode train_classifier --epochs 30
    
    # 训练对比学习模型
    python run.py --mode train_contrastive --epochs 30 --margin 0.5
    
    # 评估认证性能
    python run.py --mode evaluate --model best_contrastive_model.pth --threshold 0.5
    
    # 完整流程（训练+评估）
    python run.py --mode all
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from palm.config import load_config, get_transform, set_seed
from palm.datasets import AuthDataset, ContrastivePairDataset
from palm.models import INet
from palm.trainer import train_classifier, train_contrastive, train_arcface_contrastive
from palm.evaluate import evaluate_authentication


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='掌纹识别系统')
    
    # 运行模式
    parser.add_argument('--mode', type=str, default='all',
                        choices=['train_classifier', 'train_contrastive', 'train_arcface_contrastive', 'evaluate', 'all'],
                        help='运行模式')
    
    # 数据参数
    parser.add_argument('--data_root', type=str, default='PalmBigDataBase',
                        help='数据集根目录')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='数据加载线程数')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=30,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--cls_loss', type=str, default='ce', choices=['ce', 'focal', 'mse'],
                        help='分类损失类型')
    parser.add_argument('--backbone', type=str, default='inet', choices=['inet', 'mobileone', 'resnet18', 'resnet34'],
                        help='特征提取骨干')
    parser.add_argument('--feature_dim', type=int, default=128,
                        help='特征维度')
    parser.add_argument('--margin', type=float, default=0.5,
                        help='对比损失的margin')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='对比学习训练批次大小（默认读取配置或32）')
    parser.add_argument('--arcface_margin', type=float, default=0.5,
                        help='ArcFace 角度 margin')
    parser.add_argument('--arcface_scale', type=float, default=64.0,
                        help='ArcFace logits 缩放因子')
    parser.add_argument('--contrastive_weight', type=float, default=1.0,
                        help='混合训练时对比损失的权重系数')
    
    # 评估参数
    parser.add_argument('--model', type=str, default=None,
                        help='模型权重文件（不填则按 backbone 使用 best_contrastive_<backbone>.pth）')
    parser.add_argument('--threshold', type=float, nargs='+', default=None,
                        help='认证阈值（留空则使用自动推荐阈值）')
    parser.add_argument('--plot_roc', action='store_true', help='在评估时输出 ROC 曲线图')
    parser.add_argument('--roc_path', type=str, default=None, help='ROC 曲线保存路径（png），未指定则保存在 logs/ 下')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--cache', action='store_true', default=True,
                        help='是否缓存数据到内存（默认缓存）')
    parser.add_argument('--no_cache', action='store_false', dest='cache',
                        help='不缓存数据到内存')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 加载配置
    cfg = load_config()
    set_seed(args.seed)
    
    print("="*60)
    print("掌纹识别系统")
    print("="*60)
    print(f"运行模式: {args.mode}")
    print(f"数据路径: {args.data_root}")
    print(f"设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("="*60)
    
    # 根据模式执行不同操作
    if args.mode == 'train_classifier' or args.mode == 'all':
        print("\n[1/3] 训练分类模型...")
        cls_path = f"best_classifier_{args.backbone}.pth"
        result = train_classifier(
            cfg=cfg,
            data_root=args.data_root,
            epochs=args.epochs,
            lr=args.lr,
            cache=args.cache,
            num_workers=args.num_workers,
            save_path=cls_path,
            loss_type=args.cls_loss,
            backbone=args.backbone,
        )
        print(f"✓ 分类模型训练完成 - 最佳准确率: {result['best_metric']*100:.2f}%\n")
    
    if args.mode == 'train_contrastive' or args.mode == 'all':
        print("\n[2/3] 训练对比学习模型...")
        ct_path = f"best_contrastive_{args.backbone}.pth"
        result = train_contrastive(
            cfg=cfg,
            data_root=args.data_root,
            epochs=args.epochs,
            lr=args.lr,
            margin=args.margin,
            feature_dim=args.feature_dim,
            num_workers=args.num_workers,
            save_path=ct_path,
            backbone=args.backbone,
            batch_size=args.batch_size,
        )
        print(f"✓ 对比学习模型训练完成 - 最佳损失: {result['best_metric']:.4f}\n")
        
        # 如果是all模式，更新模型路径
        if args.mode == 'all':
            args.model = result['best_path']

    if args.mode == 'train_arcface_contrastive':
        print("\n[2/3] ArcFace+对比混合训练...")
        mix_path = f"best_arcface_contrastive_{args.backbone}.pth"
        result = train_arcface_contrastive(
            cfg=cfg,
            data_root=args.data_root,
            epochs=args.epochs,
            lr=args.lr,
            margin=args.margin,
            feature_dim=args.feature_dim,
            num_workers=args.num_workers,
            save_path=mix_path,
            backbone=args.backbone,
            batch_size=args.batch_size,
            arcface_margin=args.arcface_margin,
            arcface_scale=args.arcface_scale,
            contrastive_weight=args.contrastive_weight,
        )
        print(f"✓ 混合模型训练完成 - 最佳损失: {result['best_metric']:.4f}\n")
        args.model = result['best_path']
    
    if args.mode == 'evaluate' or args.mode == 'all':
        print("\n[3/3] 评估认证性能...")
        # 日志
        log_path = Path("logs") / f"evaluate_{datetime.now():%Y%m%d_%H%M%S}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_f = open(log_path, "a", encoding="utf-8")

        def log(msg: str):
            print(msg)
            log_f.write(msg + "\n")
            log_f.flush()
        
        # 加载模型
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = None
        model_path = args.model or f"best_contrastive_{args.backbone}.pth"
        try:
            checkpoint = torch.load(model_path, map_location=device)
            log(f"✓ 成功加载模型: {model_path}")
        except FileNotFoundError:
            log(f"✗ 找不到模型文件: {model_path}")
            log_f.close()
            return

        # 兼容两种存储格式：纯 state_dict 或 新版字典
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            saved_backbone = checkpoint.get("backbone", "inet")
            saved_feature_dim = checkpoint.get("feature_dim", args.feature_dim)
            if saved_backbone == "mobileone":
                from palm.mobileone import MobileOne

                model = MobileOne(feature_dim=saved_feature_dim, normalize=False).to(device)
            elif saved_backbone in {"resnet18", "resnet34"}:
                from palm.models import ResNetBackbone

                model = ResNetBackbone(name=saved_backbone, feature_dim=saved_feature_dim, normalize=False).to(device)
            else:
                model = INet(feature_dim=saved_feature_dim).to(device)
            model.load_state_dict(checkpoint["model"])
        else:
            model = INet(feature_dim=args.feature_dim).to(device)
            model.load_state_dict(checkpoint)
        
        # 准备数据：使用 ContrastivePairDataset 的 test split，评估与训练验证一致（成对评估）
        transform = get_transform(cfg)
        test_set = ContrastivePairDataset(args.data_root, mode='test', transform=transform, cache=args.cache)
        test_loader = DataLoader(
            test_set, batch_size=cfg['test']['batch_size'], shuffle=False, num_workers=args.num_workers
        )
        
        user_thresholds = args.threshold if args.threshold is not None else []
        pos_distances = []
        neg_distances = []

        # 先计算一次所有距离
        with torch.no_grad():
            for img1, img2, labels in test_loader:
                img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
                feats1 = model(img1)
                feats2 = model(img2)
                f1 = torch.nn.functional.normalize(feats1, p=2, dim=1)
                f2 = torch.nn.functional.normalize(feats2, p=2, dim=1)
                dist = 1 - torch.nn.functional.cosine_similarity(f1, f2)
                for d, lbl in zip(dist, labels):
                    if lbl == 1.0:
                        pos_distances.append(d.item())
                    else:
                        neg_distances.append(d.item())

        pos = np.array(pos_distances)
        neg = np.array(neg_distances)

        def compute_metrics(th: float):
            """给定阈值，计算 FAR / FRR / ACC（百分比形式）"""
            if not (pos.size and neg.size):
                return 0.0, 0.0, 0.0
            far = float((neg < th).mean() * 100)
            frr = float((pos > th).mean() * 100)

            tp = float((pos < th).sum())
            fn = float((pos >= th).sum())
            fp = float((neg < th).sum())
            tn = float((neg >= th).sum())
            total = tp + fn + fp + tn
            acc = float((tp + tn) / total * 100) if total > 0 else 0.0
            return far, frr, acc

        # 使用二分搜索在距离范围内寻找 FAR≈FRR 的推荐阈值
        best_th, best_gap = None, float("inf")
        if pos.size and neg.size:
            all_scores = np.concatenate([pos, neg])
            low = float(all_scores.min()) - 1e-6
            high = float(all_scores.max()) + 1e-6

            def gap(th: float) -> float:
                far_tmp, frr_tmp, _ = compute_metrics(th)
                return far_tmp - frr_tmp

            g_low = gap(low)
            g_high = gap(high)
            if g_low > 0 or g_high < 0:
                # 极端情况下退化为线性扫描
                scan = np.linspace(all_scores.min(), all_scores.max(), num=200)
                for th in scan:
                    far_tmp, frr_tmp, _ = compute_metrics(th)
                    gap_val = abs(far_tmp - frr_tmp)
                    if gap_val < best_gap:
                        best_gap = gap_val
                        best_th = float(th)
            else:
                # 二分查找 FAR≈FRR 的阈值
                for _ in range(40):
                    mid = 0.5 * (low + high)
                    gap_mid = gap(mid)
                    gap_abs = abs(gap_mid)
                    if gap_abs < best_gap:
                        best_gap = gap_abs
                        best_th = mid
                    if gap_mid > 0:
                        high = mid
                    else:
                        low = mid

        log(f"样本数: 正={len(pos)}, 负={len(neg)}, 正均值={pos.mean():.4f}, 负均值={neg.mean():.4f}")
        if best_th is not None:
            far_b, frr_b, acc_b = compute_metrics(best_th)
            log(f"推荐阈值≈EER: {best_th:.4f} -> FAR={far_b:.2f}%, FRR={frr_b:.2f}%, ACC={acc_b:.2f}% (gap={best_gap:.2f})")

        # 可选：绘制 ROC 曲线
        if args.plot_roc and pos.size and neg.size:
            # 全范围扫描
            all_scores = np.concatenate([pos, neg])
            scan = np.linspace(all_scores.min(), all_scores.max(), num=400)
            fpr_list = []
            tpr_list = []
            for th in scan:
                far, frr = compute_far_frr(th)
                fpr_list.append(far / 100)
                tpr_list.append(1 - frr / 100)
            auc = float(np.trapz(tpr_list, fpr_list))
            try:
                import matplotlib.pyplot as plt

                roc_path = Path(args.roc_path) if args.roc_path else Path("logs") / f"roc_{datetime.now():%Y%m%d_%H%M%S}.png"
                roc_path.parent.mkdir(parents=True, exist_ok=True)
                plt.figure()
                plt.plot(fpr_list, tpr_list, label=f"AUC={auc:.4f}")
                plt.plot([0, 1], [0, 1], "k--", alpha=0.3)
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title("ROC Curve (pair evaluation)")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(roc_path, dpi=200, bbox_inches="tight")
                plt.close()
                log(f"✓ ROC 曲线已保存: {roc_path} (AUC={auc:.4f})")
            except Exception as e:
                log(f"✗ 绘制 ROC 失败: {e}")

        # 阈值列表：用户未指定则只用推荐阈值
        thresholds = user_thresholds if user_thresholds else ([best_th] if best_th is not None else [])

        # 按阈值输出
        log("\n" + "="*60)
        log("评估结果总结:")
        log("="*60)
        log(f"{'阈值':<10} {'FAR (%)':<10} {'FRR (%)':<10} {'ACC (%)':<10} {'平均真实距离':<15} {'平均冒充距离':<15}")
        log("-"*60)
        for th in thresholds:
            far, frr, acc = compute_metrics(th)
            log(f"{th:<10.4f} {far:<10.2f} {frr:<10.2f} {acc:<10.2f} {pos.mean():<15.4f} {neg.mean():<15.4f}")
        if best_th is not None and not user_thresholds:
            log("-"*60)
            far_b, frr_b, acc_b = compute_metrics(best_th)
            log(f"推荐阈值≈EER: {best_th:.4f} -> FAR={far_b:.2f}%, FRR={frr_b:.2f}%, ACC={acc_b:.2f}% (gap={best_gap:.2f})")
        log("="*60)
        log(f"日志已保存: {log_path}")
        log_f.close()
    
    print("\n✓ 所有任务完成！")


if __name__ == '__main__':
    main()
