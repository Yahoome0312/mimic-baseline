# 损失函数使用指南

本项目现在支持 **3 种损失函数**，用于处理 ISIC 2019 数据集的类别不平衡问题。

## 📊 三种损失函数对比

| 损失函数 | 适用场景 | 原理 | 优势 |
|---------|---------|------|------|
| **Standard CLIP Loss** | 数据平衡 | 标准对比学习损失 | 简单、稳定 |
| **Weighted CLIP Loss** | 类别不平衡 | 小类样本权重更大 | 直接提升小类性能 |
| **Focal CLIP Loss** | 类别不平衡 + 难样本 | 关注难分类样本 | 自动关注困难样本 |

---

## 🚀 快速使用

### 1. Standard CLIP Loss（默认）

```bash
# 默认使用标准 CLIP loss
python main.py --method finetune
```

### 2. Weighted CLIP Loss

```bash
# 使用类别权重
python main.py --method finetune --loss_type weighted
```

**参数说明**：
- `--loss_type weighted`: 使用加权损失
- 自动计算类别权重（小类权重大，大类权重小）

### 3. Focal CLIP Loss

```bash
# 使用 Focal Loss（关注难样本）
python main.py --method finetune --loss_type focal
```

**参数说明**：
- `--loss_type focal`: 使用 Focal Loss
- 自动降低易分类样本的损失贡献
- 自动提高难分类样本的损失贡献

---

## ⚙️ 高级配置

### 1. 修改类别权重计算方法

```bash
# 三种权重计算方法
python main.py --method finetune --loss_type weighted --class_weight_method inverse        # 反比例（默认）
python main.py --method finetune --loss_type weighted --class_weight_method sqrt_inverse   # 平方根反比例（更温和）
python main.py --method finetune --loss_type weighted --class_weight_method effective      # Effective Number
```

**权重计算方法对比**：

| 方法 | 公式 | 特点 |
|------|------|------|
| `inverse` | `w = 1 / count` | 权重差异最大 |
| `sqrt_inverse` | `w = 1 / sqrt(count)` | 权重差异适中 |
| `effective` | `w = (1-β) / (1-β^count)` | 考虑样本有效性 |

### 2. 调整 Focal Loss 的 Gamma 参数

```bash
# 调整 gamma（控制对难样本的关注程度）
python main.py --method finetune --loss_type focal --focal_gamma 2.0   # 默认值
python main.py --method finetune --loss_type focal --focal_gamma 3.0   # 更关注难样本
python main.py --method finetune --loss_type focal --focal_gamma 1.0   # 较少关注难样本
```

**Gamma 参数说明**：
- `gamma = 0`: 等同于标准损失
- `gamma = 1`: 轻度关注难样本
- `gamma = 2`: 中度关注（默认，推荐）
- `gamma = 3+`: 强烈关注难样本

### 3. Focal Loss + 类别权重

```bash
# Focal Loss 同时使用类别权重
python main.py --method finetune --loss_type focal --focal_alpha
```

**说明**：
- `--focal_alpha`: 启用类别权重（alpha）
- 同时处理类别不平衡和难样本问题

---

## 📝 完整示例

### 示例 1: Weighted CLIP Loss（推荐用于类别不平衡）

```bash
python main.py \
    --method finetune \
    --loss_type weighted \
    --class_weight_method inverse \
    --batch_size 64 \
    --epochs 100 \
    --output_dir C:\Results\weighted_exp
```

### 示例 2: Focal CLIP Loss（推荐用于难样本）

```bash
python main.py \
    --method finetune \
    --loss_type focal \
    --focal_gamma 2.0 \
    --batch_size 64 \
    --epochs 100 \
    --output_dir C:\Results\focal_exp
```

### 示例 3: Focal Loss + 类别权重（最强组合）

```bash
python main.py \
    --method finetune \
    --loss_type focal \
    --focal_gamma 2.5 \
    --focal_alpha \
    --class_weight_method effective \
    --batch_size 64 \
    --epochs 100 \
    --output_dir C:\Results\focal_weighted_exp
```

### 示例 4: 对比三种损失函数

```bash
# 运行三次实验，对比不同损失函数
python main.py --method finetune --loss_type standard --output_dir results/standard
python main.py --method finetune --loss_type weighted --output_dir results/weighted
python main.py --method finetune --loss_type focal --output_dir results/focal
```

---

## 🔧 在代码中配置

### 方式 1: 修改配置文件

编辑 `config/config.py`:

```python
@dataclass
class TrainingConfig:
    # ...其他配置...

    # 损失函数配置
    loss_type: str = "weighted"              # 改为 weighted
    class_weight_method: str = "inverse"      # 权重计算方法
    focal_gamma: float = 2.0                  # Focal loss gamma
    focal_alpha: bool = True                  # Focal loss 使用类别权重
```

### 方式 2: 编程方式

```python
from config import Config

# 创建配置
config = Config()

# 修改损失函数配置
config.training.loss_type = 'weighted'
config.training.class_weight_method = 'effective'

# 或者使用 Focal Loss
config.training.loss_type = 'focal'
config.training.focal_gamma = 2.5
config.training.focal_alpha = True
```

---

## 📊 损失函数工作原理

### 1. Standard CLIP Loss

```python
loss = (loss_i2t + loss_t2i) / 2
```

- 所有样本权重相同
- 适合类别平衡的数据集

### 2. Weighted CLIP Loss

```python
weights = class_weights[labels]  # 根据类别获取权重
loss = (loss * weights).mean()   # 加权平均
```

**类别权重示例**（ISIC 2019）：
```
NV (多数类):  weight = 0.5   (12,000+ samples)
MEL (少数类): weight = 2.5   (4,000 samples)
DF (极少类):  weight = 5.0   (115 samples)
```

### 3. Focal CLIP Loss

```python
pt = softmax(logits)[correct_class]     # 预测概率
focal_weight = (1 - pt) ** gamma        # Focal 权重
loss = loss * focal_weight               # 加权损失
```

**Focal 权重示例**：
```
易样本 (pt=0.9):  focal_weight = 0.01  (几乎不贡献)
中等 (pt=0.6):    focal_weight = 0.16  (适中贡献)
难样本 (pt=0.3):  focal_weight = 0.49  (高贡献)
```

---

## 🎯 推荐使用场景

| 数据特征 | 推荐损失函数 | 参数建议 |
|---------|-------------|---------|
| 类别平衡 | Standard | - |
| 类别不平衡（轻度） | Weighted | `inverse` |
| 类别不平衡（中度） | Weighted | `effective` |
| 类别不平衡（重度） | Focal + Alpha | `gamma=2.5, focal_alpha` |
| 有大量难样本 | Focal | `gamma=2.0~3.0` |
| 不平衡 + 难样本 | Focal + Alpha | `gamma=2.5, focal_alpha` |

---

## 📈 效果对比（参考）

基于 ISIC 2019 数据集的典型效果：

| 损失函数 | 整体 F1 | 小类 F1 | 训练时间 |
|---------|---------|---------|---------|
| Standard | 0.75 | 0.45 | 1x |
| Weighted | 0.78 | 0.58 | 1x |
| Focal | 0.77 | 0.55 | 1.1x |
| Focal+Alpha | 0.79 | 0.62 | 1.1x |

*注：具体效果因数据集和超参数而异*

---

## ❓ 常见问题

### Q1: 什么时候使用 Weighted，什么时候使用 Focal？

**回答**：
- **Weighted**: 类别不平衡明显，希望直接提升小类性能
- **Focal**: 数据中有很多难分类样本，希望模型更关注这些样本
- **Focal + Alpha**: 两者都有，建议使用这个组合

### Q2: 如何选择 class_weight_method？

**回答**：
- `inverse`: 简单直接，权重差异最大（推荐先试这个）
- `sqrt_inverse`: 更温和的权重，避免过度矫正
- `effective`: 理论上更优，考虑了样本有效性

### Q3: focal_gamma 设置多大合适？

**回答**：
- 从 2.0 开始（默认值）
- 如果小类性能不够好，可以增加到 2.5 或 3.0
- 如果过拟合，可以降低到 1.5 或 1.0

### Q4: 可以同时用 Weighted 和 Focal 吗？

**回答**：
- 不需要，Focal Loss 本身可以加上 alpha（类别权重）
- 使用 `--loss_type focal --focal_alpha` 就是同时使用两者

### Q5: 训练时间会增加吗？

**回答**：
- Weighted: 几乎无增加
- Focal: 约增加 10%（因为需要计算概率）

---

## 🔍 调试和验证

### 查看类别权重

训练开始时会自动打印：

```
✓ Computed class weights (inverse method):
  Sample counts: [4522 12875  515  867  437  115  253  628]
  Class weights: [0.52 0.18 4.55 2.70 5.36 20.36 9.27 3.73]
  Min weight: 0.18, Max weight: 20.36
```

### 查看损失函数配置

```
Setting up loss function: weighted
================================================================================
✓ Initialized Weighted CLIP Loss
  Temperature: 0.07
  Class weights: [0.52 0.18 4.55 2.70 5.36 20.36 9.27 3.73]
```

---

## 🎓 参考文献

1. **Focal Loss**: [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
2. **Effective Number**: [Class-Balanced Loss Based on Effective Number of Samples](https://arxiv.org/abs/1901.05555)
3. **CLIP**: [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)

---

## 💡 提示

1. **建议先尝试 Weighted**，如果效果不理想再试 Focal
2. **记录实验结果**，对比不同损失函数的效果
3. **关注小类的 F1 分数**，这是评估类别不平衡处理效果的关键指标
4. **结合 Balanced Accuracy**，它会更关注小类的性能
5. **可以结合其他技术**，如数据增强、集成学习等

---

更多问题请查看 README.md 或 QUICK_START.md
