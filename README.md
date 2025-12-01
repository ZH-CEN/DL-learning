# Palm Recognition System

掌纹识别系统 - 基于深度学习的掌纹认证系统

## 📁 项目结构

```
DL-learning/
├── palm/                          # 核心模块
│   ├── __init__.py               # 包初始化
│   ├── config.py                 # 配置管理
│   ├── datasets.py               # 数据集定义
│   ├── models.py                 # 模型定义
│   ├── losses.py                 # 损失函数
│   ├── trainer.py                # 训练函数
│   └── evaluate.py               # 评估函数
├── config/                       # 按损失函数划分的训练配置
│   ├── contrastive.yml
│   ├── triplet.yml
│   ├── margin.yml
│   ├── focal.yml
│   └── mse.yml
├── PalmBigDataBase/              # 数据集目录
├── args.yml                       # 配置文件
├── run.py                         # 命令行主程序
├── main.py                        # 简化版主程序
├── PalmRecognition.ipynb         # Jupyter Notebook
└── README.md                      # 本文档
```

## 🚀 快速开始

### 安装依赖

```bash
pip install torch torchvision pillow pyyaml tqdm
```

### 数据准备

将数据集放在 `PalmBigDataBase/` 目录下，文件命名格式：
- 右手：`P_F_{ID}_{序号}.bmp`（例如：`P_F_100_1.bmp`）
- 左手：`P_S_{ID}_{序号}.bmp`（例如：`P_S_100_1.bmp`）

每个身份包含10张图像：前5张用于训练，后5张用于测试。

## 💻 使用方法

### 方法1：命令行运行（推荐）

#### 完整流程（训练+评估）
```bash
python run.py --mode all --epochs 30
```

#### 仅训练分类模型
```bash
python run.py --mode train_classifier --epochs 30 --lr 0.001 --cls_loss ce  # 可选 ce/focal/mse
```

#### 仅训练对比学习模型
```bash
python run.py --mode train_contrastive --epochs 30 --margin 0.5 --feature_dim 128
```

#### 仅评估模型
```bash
python run.py --mode evaluate --model best_contrastive_model.pth --threshold 0.5
```

#### 多阈值评估
```bash
python run.py --mode evaluate --threshold 0.3 0.4 0.5 0.6 0.7
```

> 度量学习训练时会自动读取 `config/<loss>.yml`（如 `config/contrastive.yml`、`config/triplet.yml`、`config/margin.yml`），其中可配置 `learning_rate`、`epochs`、`batch_size`、`feature_dim`、`margin`、`weight_decay`、`lr_step_size`、`lr_gamma` 等超参数。未提供对应文件时使用代码内默认值。
>
> 分类训练支持 `--cls_loss ce|focal|mse`，会自动读取 `config/focal.yml` 或 `config/mse.yml` 中的超参（如 `focal_alpha`、`focal_gamma`、`learning_rate`、`epochs`）。未提供配置文件则使用代码默认值。

### 方法2：Jupyter Notebook

打开 `PalmRecognition.ipynb` 交互式运行。

## 📊 模型说明

### INet（特征提取网络）
- **输入**：128×128 灰度图像
- **输出**：128维特征向量
- **结构**：
  - 3个卷积块（Conv + BN + ReLU + MaxPool）
  - 2个全连接层
  - Dropout正则化

### 训练方法

#### 1. 分类训练（Classification）
- **损失函数**：交叉熵损失
- **用途**：学习区分不同身份
- **输出**：`best_classifier.pth`

#### 2. 对比学习（Contrastive Learning）
- **损失函数**：基于余弦相似度的对比损失
- **用途**：学习相似度度量
- **输出**：`best_contrastive_model.pth`
- **参数**：
  - `margin=0.5`：余弦距离边界
  - 正样本对：同一身份同一只手
  - 负样本对：不同身份或不同手

## 📈 评估指标

- **FAR (False Acceptance Rate)**：误识率，冒充者被错误接受的比例
- **FRR (False Rejection Rate)**：误拒率，真实用户被错误拒绝的比例
- **阈值调优**：根据应用场景平衡 FAR 和 FRR

### 典型结果示例

```
阈值      FAR (%)    FRR (%)    平均真实距离    平均冒充距离
0.30      0.50       15.20      0.2850         0.7150
0.40      1.20       8.50       0.2850         0.7150
0.50      3.80       4.20       0.2850         0.7150  ← 推荐
0.60      8.50       1.80       0.2850         0.7150
0.70      15.20      0.50       0.2850         0.7150
```

## ⚙️ 配置说明

编辑 `args.yml` 修改配置：

```yaml
train:
  batch_size: 512          # 批次大小
  epochs: 50               # 训练轮数
  learning_rate: 0.001     # 学习率
  weight_decay: 0.0001     # 权重衰减
  lr_step_size: 20         # 学习率衰减步长
  lr_gamma: 0.5            # 学习率衰减因子

img_basic_info:
  img_height: 128          # 图像高度
  img_width: 128           # 图像宽度
  img_channels: 3          # 通道数
```

## 🔧 常见问题

### Q1: 内存不足
```bash
# 减小批次大小
python run.py --mode all --epochs 30
# 编辑 args.yml，将 batch_size 从 512 改为 128 或更小
```

### Q2: Windows下DataLoader报错
```bash
# 设置 num_workers=0
python run.py --mode all --num_workers 0
```

### Q3: 训练准确率停留在50%
- 检查阈值设置（推荐0.5）
- 增加训练轮数
- 调整margin参数（0.3-0.7）

### Q4: 找不到模型文件
```bash
# 先训练模型
python run.py --mode train_contrastive --epochs 30
# 再评估
python run.py --mode evaluate --model best_contrastive_model.pth
```

## 📝 命令行参数完整列表

```
--mode              运行模式 [train_classifier|train_contrastive|evaluate|all]
--data_root         数据集路径 (默认: PalmBigDataBase)
--num_workers       数据加载线程数 (默认: 0)
--epochs            训练轮数 (默认: 30)
--lr                学习率 (默认: 0.001)
--feature_dim       特征维度 (默认: 128)
--margin            对比损失margin (默认: 0.5)
--model             模型文件路径 (默认: best_contrastive_model.pth)
--threshold         认证阈值，可多个 (默认: 0.3 0.4 0.5 0.6 0.7)
--seed              随机种子 (默认: 42)
--cache             是否缓存数据到内存
```

## 📄 许可证

本项目仅供学习和研究使用。

## 👨‍💻 作者

ZH-CEN

## 🙏 致谢

感谢 PalmBigDataBase 数据集的提供者。
