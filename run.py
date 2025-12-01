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
import torch
from torch.utils.data import DataLoader

from palm.config import load_config, get_transform, set_seed
from palm.datasets import AuthDataset
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
    parser.add_argument('--threshold', type=float, nargs='+', default=[0.3, 0.4, 0.5, 0.6, 0.7],
                        help='认证阈值（可以指定多个）')
    
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
        
        # 准备数据，共享 id2idx，避免 train/test 标签映射不一致
        from palm.datasets import AuthDataset
        from pathlib import Path as _Path

        transform = get_transform(cfg)
        all_ids = sorted({AuthDataset._get_identity(p.name) for p in _Path(args.data_root).glob("*.bmp")})
        shared_id2idx = {pid: idx for idx, pid in enumerate(all_ids)}

        gallery_set = AuthDataset(args.data_root, mode='train', transform=transform, cache=args.cache, id2idx=shared_id2idx)
        query_set = AuthDataset(args.data_root, mode='test', transform=transform, cache=args.cache, id2idx=shared_id2idx)
        
        gallery_loader = DataLoader(gallery_set, batch_size=cfg['train']['batch_size'], 
                                    shuffle=False, num_workers=args.num_workers)
        query_loader = DataLoader(query_set, batch_size=cfg['test']['batch_size'], 
                                  shuffle=False, num_workers=args.num_workers)
        
        # 多个阈值评估
        thresholds = args.threshold if isinstance(args.threshold, list) else [args.threshold]
        
        results = []
        for threshold in thresholds:
            result = evaluate_authentication(model, gallery_loader, query_loader, threshold=threshold, log_func=log)
            results.append((threshold, result))
        
        # 输出总结
        log("\n" + "="*60)
        log("评估结果总结:")
        log("="*60)
        log(f"{'阈值':<10} {'FAR (%)':<10} {'FRR (%)':<10} {'平均真实距离':<15} {'平均冒充距离':<15}")
        log("-"*60)
        for threshold, result in results:
            log(f"{threshold:<10.2f} {result['FAR']:<10.2f} {result['FRR']:<10.2f} "
                f"{result['genuine_mean']:<15.4f} {result['impostor_mean']:<15.4f}")
        log("="*60)
        log(f"日志已保存: {log_path}")
        log_f.close()
    
    print("\n✓ 所有任务完成！")


if __name__ == '__main__':
    main()
