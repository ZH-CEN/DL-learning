"""
Palm Recognition Package
掌纹识别系统核心模块
"""

__version__ = "1.0.0"

from .models import INet, FeatureNet, VGG, PalmClassifier, FeatureExtractor
from .datasets import PalmDataset, AuthDataset, ContrastivePairDataset, TripletDataset
from .losses import ContrastiveLoss, TripletLoss, MarginLoss, FocalLoss, MSELoss
from .config import load_config, load_loss_config, get_transform, set_seed
from .trainer import train_classifier, train_contrastive
from .evaluate import evaluate_authentication, extract_features, build_gallery, authenticate_single

__all__ = [
    # Models
    "INet",
    "FeatureNet",
    "VGG",
    "PalmClassifier",
    "FeatureExtractor",
    # Datasets
    "PalmDataset",
    "AuthDataset",
    "ContrastivePairDataset",
    "TripletDataset",
    # Losses
    "ContrastiveLoss",
    "TripletLoss",
    "MarginLoss",
    "FocalLoss",
    "MSELoss",
    # Config
    "load_config",
    "load_loss_config",
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
