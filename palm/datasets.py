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
    
    def __init__(self, root, transform=None, target_transform=None, cache=True, samples=None, id2idx=None):
        """
        Args:
            root: 数据集根目录
            transform: 图像转换
            target_transform: 标签转换
            cache: 是否缓存图像到内存
            samples: 可选，预先指定的样本 Path 列表；若为 None，则从 root/*.bmp 自动收集
            id2idx: 可选，固定的身份->类别编号映射，保证跨 train/val 一致
        """
        self.root = Path(root)
        if samples is None:
            self.samples = sorted(self.root.glob("*.bmp"))
        else:
            self.samples = sorted(samples)
        self.transform = transform
        self.target_transform = target_transform
        self.cache = cache
        self._cache_data = {}

        # 区分左右手，格式 "F_100" 或 "S_100"
        if id2idx is None:
            ids = sorted({self._get_identity(p.name) for p in self.samples})
            self.id2idx = {pid: idx for idx, pid in enumerate(ids)}
        else:
            self.id2idx = dict(id2idx)
            missing = {self._get_identity(p.name) for p in self.samples} - set(self.id2idx)
            if missing:
                raise ValueError(f"样本中存在未映射的身份: {missing}")

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
        从文件名提取身份ID（忽略 F/S 光照标记，按 person_id 聚合）
        例：P_F_100_1.bmp / P_S_100_1.bmp → "100"
        """
        parts = filename.split("_")
        person_id = parts[2]
        return person_id

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
    按人聚合身份，训练用 F 手，测试用 S 手，可选缓存。
    """

    def __init__(self, root, mode='train', transform=None, cache: bool = True):
        """
        Args:
            root: 数据集根目录
            mode: 'train' 或 'test'
            transform: 图像转换
            cache: 是否将图像缓存到内存
        """
        self.root = Path(root)
        self.mode = mode
        self.transform = transform
        self.cache = cache
        self._cache_data = {}

        all_samples = sorted(self.root.glob("*.bmp"))
        id_groups = {}
        for path in all_samples:
            pid = self._get_identity(path.name)
            hand = self._get_hand(path.name)
            id_groups.setdefault(pid, {"F": [], "S": []})
            id_groups[pid][hand].append(path)

        # 训练仅用 F，测试仅用 S，避免光照泄漏
        self.samples = []
        for _, hands in id_groups.items():
            if mode == 'train':
                selected = sorted(hands["F"])
            else:
                selected = sorted(hands["S"])
            self.samples.extend(selected)

        self.samples = sorted(self.samples)
        ids = sorted({self._get_identity(p.name) for p in self.samples})
        self.id2idx = {pid: idx for idx, pid in enumerate(ids)}

        if self.cache:
            for idx, path in enumerate(self.samples):
                with Image.open(path) as img:
                    img = img.convert("L")
                    self._cache_data[idx] = img.copy()

    @staticmethod
    def _get_identity(filename):
        """从文件名提取身份ID（忽略 F/S 光照标记）"""
        parts = filename.split("_")
        person_id = parts[2]
        return person_id

    @staticmethod
    def _get_hand(filename):
        parts = filename.split("_")
        return parts[1]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        if self.cache and index in self._cache_data:
            img = self._cache_data[index]
        else:
            path = self.samples[index]
            img = Image.open(path).convert("L")
        label = self.id2idx[self._get_identity(self.samples[index].name)]

        if self.transform:
            img = self.transform(img)

        return img, label


class ContrastivePairDataset(Dataset):
    """
    对比学习数据集
    返回样本对（正样本对/负样本对），可选缓存。
    训练用 F 手，测试用 S 手。
    """

    def __init__(self, root, mode='train', transform=None, cache: bool = True):
        """
        Args:
            root: 数据集根目录
            mode: 'train' 或 'test'
            transform: 图像转换
            cache: 是否缓存图像
        """
        self.root = Path(root)
        self.mode = mode
        self.transform = transform
        self.cache = cache
        self._cache_data = {}

        all_samples = sorted(self.root.glob("*.bmp"))
        id_groups = {}
        for path in all_samples:
            pid = self._get_identity(path.name)
            hand = self._get_hand(path.name)
            id_groups.setdefault(pid, {"F": [], "S": []})
            id_groups[pid][hand].append(path)

        # 训练仅用 F，测试仅用 S
        self.id_groups = {}
        for pid, hands in id_groups.items():
            selected = sorted(hands["F"]) if mode == 'train' else sorted(hands["S"])
            if selected:
                self.id_groups[pid] = selected

        self.ids = list(self.id_groups.keys())

        if self.cache:
            for pid, paths in self.id_groups.items():
                for path in paths:
                    with Image.open(path) as img:
                        img = img.convert("L")
                        self._cache_data[path] = img.copy()

    @staticmethod
    def _get_identity(filename):
        """从文件名提取身份ID（忽略 F/S 光照标记）"""
        parts = filename.split("_")
        person_id = parts[2]
        return person_id

    @staticmethod
    def _get_hand(filename):
        parts = filename.split("_")
        return parts[1]

    def __len__(self):
        return len(self.ids) * 10  # 每个ID生成10对样本

    def _read(self, path: Path):
        if self.cache and path in self._cache_data:
            return self._cache_data[path]
        with Image.open(path) as img:
            return img.convert("L")

    def __getitem__(self, index):
        # 50%正样本对，50%负样本对
        if index % 2 == 0:
            # 正样本对（同一个人）
            pid = self.ids[index // 10 % len(self.ids)]
            paths = self.id_groups[pid]
            if len(paths) >= 2:
                path1, path2 = random.sample(paths, 2)
            else:
                path1 = path2 = paths[0]
            label = 1.0
        else:
            # 负样本对（不同的人）
            idx1 = (index // 10) % len(self.ids)
            idx2 = (idx1 + 1 + random.randint(0, len(self.ids) - 2)) % len(self.ids)

            pid1 = self.ids[idx1]
            pid2 = self.ids[idx2]

            path1 = random.choice(self.id_groups[pid1])
            path2 = random.choice(self.id_groups[pid2])
            label = 0.0

        img1 = self._read(path1)
        img2 = self._read(path2)

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(label, dtype=torch.float32)


class TripletDataset(Dataset):
    """
    三元组数据集
    返回 (anchor, positive, negative)，可选缓存。
    训练用 F 手，测试用 S 手。
    """

    def __init__(self, root, mode="train", transform=None, cache: bool = True):
        assert mode in {"train", "test"}
        self.root = Path(root)
        self.mode = mode
        self.transform = transform
        self.cache = cache
        self._cache_data = {}

        all_samples = sorted(self.root.glob("*.bmp"))
        id_groups = {}
        for path in all_samples:
            pid = self._get_identity(path.name)
            hand = self._get_hand(path.name)
            id_groups.setdefault(pid, {"F": [], "S": []})
            id_groups[pid][hand].append(path)

        # 训练仅用 F，测试仅用 S
        self.id_groups = {}
        for pid, hands in id_groups.items():
            selected = sorted(hands["F"]) if mode == "train" else sorted(hands["S"])
            if selected:
                self.id_groups[pid] = selected
        self.ids = list(self.id_groups.keys())

        if self.cache:
            for pid, paths in self.id_groups.items():
                for path in paths:
                    with Image.open(path) as img:
                        img = img.convert("L")
                        self._cache_data[path] = img.copy()

    @staticmethod
    def _get_identity(filename):
        parts = filename.split("_")
        person_id = parts[2]
        return person_id

    @staticmethod
    def _get_hand(filename):
        parts = filename.split("_")
        return parts[1]

    def __len__(self):
        return max(1, len(self.ids) * 10)

    def _read(self, path: Path):
        if self.cache and path in self._cache_data:
            return self._cache_data[path]
        with Image.open(path) as img:
            return img.convert("L")

    def __getitem__(self, index):
        # 选择一个身份作为 anchor/positive
        pid = self.ids[index % len(self.ids)]
        paths = self.id_groups[pid]
        anchor_path, pos_path = random.sample(paths, 2) if len(paths) >= 2 else (paths[0], paths[0])

        # 选择不同身份作为 negative
        neg_pid = self.ids[(index + 1) % len(self.ids)]
        neg_path = random.choice(self.id_groups[neg_pid])

        anchor = self._read(anchor_path)
        positive = self._read(pos_path)
        negative = self._read(neg_path)

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative
