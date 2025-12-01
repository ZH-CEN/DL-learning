"""
Palm Recognition Package
掌纹识别系统核心模块
"""

__version__ = "1.0.0"

from .models import INet, FeatureNet, VGG
from .datasets import PalmDataset, AuthDataset, ContrastivePairDataset
from .losses import ContrastiveLoss, TripletLoss
from .config import load_config, get_transform, set_seed
from .train import train_classifier, train_contrastive
from .evaluate import evaluate_authentication, extract_features, build_gallery, authenticate_single

__all__ = [
    # Models
    "INet",
    "FeatureNet",
    "VGG",
    # Datasets
    "PalmDataset",
    "AuthDataset",
    "ContrastivePairDataset",
    # Losses
    "ContrastiveLoss",
    "TripletLoss",
    # Config
    "load_config",
    "get_transform",
    "set_seed",
    # Training
    "train_classifier",
    "train_contrastive",
    # Evaluation
    "evaluate_authentication",
    "extract_features",
    "build_gallery",
    "authenticate_single",
]
