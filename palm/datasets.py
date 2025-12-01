"""
数据集模块
包含所有数据集类的定义
"""

import random
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class PalmDataset(Dataset):
    """
    掌纹分类数据集
    用于训练分类模型
    """
    
    def __init__(self, root, transform=None, target_transform=None, cache=True):
        """
        Args:
            root: 数据集根目录
            transform: 图像转换
            target_transform: 标签转换
            cache: 是否缓存图像到内存
        """
        self.root = Path(root)
        self.samples = sorted(self.root.glob("*.bmp"))
        self.transform = transform
        self.target_transform = target_transform
        self.cache = cache
        self._cache_data = {}

        # 区分左右手，格式 "F_100" 或 "S_100"
        ids = sorted({self._get_identity(p.name) for p in self.samples})
        self.id2idx = {pid: idx for idx, pid in enumerate(ids)}

        # 如果启用缓存，预加载所有图像
        if self.cache:
            print(f"正在缓存 {len(self.samples)} 张图像到内存...")
            for idx in tqdm(range(len(self.samples)), desc="加载图像"):
                path = self.samples[idx]
                with Image.open(path) as img:
                    img = img.convert("L")  # 灰度
                    self._cache_data[idx] = img.copy()
            print("缓存完成！")

    @staticmethod
    def _get_identity(filename):
        """
        从文件名提取身份ID
        例：P_F_100_1.bmp → "F_100" (右手ID=100)
            P_S_100_1.bmp → "S_100" (左手ID=100)
        """
        parts = filename.split("_")
        hand = parts[1]  # F 或 S
        person_id = parts[2]
        return f"{hand}_{person_id}"

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        # 从缓存或磁盘读取图像
        if self.cache and index in self._cache_data:
            img = self._cache_data[index]
        else:
            path = self.samples[index]
            with Image.open(path) as img:
                img = img.convert("L")

        label = self.id2idx[self._get_identity(self.samples[index].name)]
        
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)

        return img, label


class AuthDataset(Dataset):
    """
    认证数据集
    每个身份取5张训练，5张测试
    """
    
    def __init__(self, root, mode='train', transform=None):
        """
        Args:
            root: 数据集根目录
            mode: 'train' 或 'test'
            transform: 图像转换
        """
        self.root = Path(root)
        self.mode = mode
        self.transform = transform
        
        # 按身份ID分组样本（区分左右手）
        all_samples = sorted(self.root.glob("*.bmp"))
        id_groups = {}
        for path in all_samples:
            pid = self._get_identity(path.name)
            if pid not in id_groups:
                id_groups[pid] = []
            id_groups[pid].append(path)
        
        # 每个ID取5张训练，5张测试
        self.samples = []
        for pid, paths in id_groups.items():
            if mode == 'train':
                self.samples.extend(paths[:5])
            else:  # test
                self.samples.extend(paths[5:10])
        
        self.samples = sorted(self.samples)
        ids = sorted({self._get_identity(p.name) for p in self.samples})
        self.id2idx = {pid: idx for idx, pid in enumerate(ids)}
    
    @staticmethod
    def _get_identity(filename):
        """从文件名提取身份ID"""
        parts = filename.split("_")
        hand = parts[1]  # F 或 S
        person_id = parts[2]
        return f"{hand}_{person_id}"
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        path = self.samples[index]
        img = Image.open(path).convert("L")
        label = self.id2idx[self._get_identity(path.name)]
        
        if self.transform:
            img = self.transform(img)
        
        return img, label


class ContrastivePairDataset(Dataset):
    """
    对比学习数据集
    返回样本对（正样本对/负样本对）
    """
    
    def __init__(self, root, mode='train', transform=None):
        """
        Args:
            root: 数据集根目录
            mode: 'train' 或 'test'
            transform: 图像转换
        """
        self.root = Path(root)
        self.mode = mode
        self.transform = transform
        
        # 按身份ID分组样本（区分左右手）
        all_samples = sorted(self.root.glob("*.bmp"))
        id_groups = {}
        for path in all_samples:
            pid = self._get_identity(path.name)
            if pid not in id_groups:
                id_groups[pid] = []
            id_groups[pid].append(path)
        
        # 每个ID取5张训练，5张测试
        self.id_groups = {}
        for pid, paths in id_groups.items():
            if mode == 'train':
                selected = paths[:5]
            else:  # test
                selected = paths[5:10]
            
            # 只保留有样本的身份
            if len(selected) > 0:
                self.id_groups[pid] = selected
        
        self.ids = list(self.id_groups.keys())
    
    @staticmethod
    def _get_identity(filename):
        """从文件名提取身份ID"""
        parts = filename.split("_")
        hand = parts[1]  # F 或 S
        person_id = parts[2]
        return f"{hand}_{person_id}"
    
    def __len__(self):
        return len(self.ids) * 10  # 每个ID生成10对样本
    
    def __getitem__(self, index):
        # 50%正样本对，50%负样本对
        if index % 2 == 0:
            # 正样本对（同一个人同一只手）
            pid = self.ids[index // 10 % len(self.ids)]
            paths = self.id_groups[pid]
            if len(paths) >= 2:
                path1, path2 = random.sample(paths, 2)
            else:
                path1 = path2 = paths[0]
            label = 1.0
        else:
            # 负样本对（不同的人或不同的手）
            idx1 = (index // 10) % len(self.ids)
            idx2 = (idx1 + 1 + random.randint(0, len(self.ids) - 2)) % len(self.ids)
            
            pid1 = self.ids[idx1]
            pid2 = self.ids[idx2]
            
            path1 = random.choice(self.id_groups[pid1])
            path2 = random.choice(self.id_groups[pid2])
            label = 0.0
        
        img1 = Image.open(path1).convert("L")
        img2 = Image.open(path2).convert("L")
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return img1, img2, torch.tensor(label, dtype=torch.float32)
