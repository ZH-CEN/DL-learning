"""
配置管理模块
负责加载配置文件和设置随机种子
"""

import yaml
import torch
import numpy as np
import random
from pathlib import Path
import torchvision.transforms as T


def load_config(config_path="args.yml"):
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        dict: 配置字典
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        # 默认配置
        cfg = {
            "model": {"name": "noise_split_net"},
            "img_basic_info": {
                "img_height": 128,
                "img_width": 128,
                "img_channels": 3
            },
            "train": {
                "batch_size": 512,
                "epochs": 50,
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "lr_step_size": 20,
                "lr_gamma": 0.5
            },
            "test": {
                "batch_size": 32,
                "shuffle": False
            }
        }
    return cfg


def get_transform(cfg):
    """
    获取数据预处理转换
    
    Args:
        cfg: 配置字典
        
    Returns:
        torchvision.transforms.Compose: 转换组合
    """
    img_height = cfg["img_basic_info"]["img_height"]
    img_width = cfg["img_basic_info"]["img_width"]
    
    transform = T.Compose([
        T.Resize((img_height, img_width)),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])
    return transform


def set_seed(seed=42):
    """
    设置随机种子以保证可重复性
    
    Args:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
