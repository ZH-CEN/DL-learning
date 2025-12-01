"""
Optuna 自动调参示例：针对 MobileOne + Contrastive。
运行前请确认已安装 optuna：pip install optuna
"""

import argparse
from datetime import datetime
from pathlib import Path

import optuna
import torch

from palm.config import load_config
from palm.trainer import train_contrastive


def objective(trial, args):
    cfg = load_config()

    # 采样超参
    lr = trial.suggest_loguniform("lr", 1e-4, 5e-3)
    margin = trial.suggest_float("margin", 0.2, 0.8)
    feature_dim = trial.suggest_categorical("feature_dim", [64, 96, 128])
    batch_size = trial.suggest_categorical("batch_size", [8, 12, 16, 24])
    epochs = trial.suggest_int("epochs", 8, 16)

    result = train_contrastive(
        cfg=cfg,
        data_root=args.data_root,
        epochs=epochs,
        lr=lr,
        margin=margin,
        feature_dim=feature_dim,
        num_workers=args.num_workers,
        save_path=f"trial_{trial.number}_mobileone_contrastive.pth",
        loss_type="contrastive",
        backbone="mobileone",
        batch_size=batch_size,
        cache=args.cache,
        log_file=Path("logs") / f"optuna_trial_{trial.number}_{datetime.now():%Y%m%d_%H%M%S}.log",
    )

    # 以验证损失为优化目标
    return result["best_metric"]


def parse_args():
    parser = argparse.ArgumentParser(description="Optuna tuning for MobileOne + Contrastive")
    parser.add_argument("--data_root", type=str, default="PalmBigDataBase")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--cache", action="store_true", default=True)
    parser.add_argument("--no_cache", action="store_false", dest="cache")
    return parser.parse_args()


def main():
    args = parse_args()
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, args), n_trials=args.trials)

    print("Best trial:")
    print(study.best_trial)
    print("Best params:")
    print(study.best_trial.params)


if __name__ == "__main__":
    main()
