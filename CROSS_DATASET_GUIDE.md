# 跨数据集测试指南

## 概述

本项目现在支持**跨数据集迁移学习**：
- **训练**: 使用 MIMIC-CXR 数据集 (377K+ 图像)
- **测试**: 使用 ChestXray14 数据集 (25.6K 测试图像)

这种设置可以评估模型的**泛化能力**和**迁移能力**。

---

## 数据集对比

### MIMIC-CXR (训练数据)
- **图像数量**: 377,095 张
- **训练集**: 368,945 张
- **类别数**: 14 个 CheXpert 标签
- **来源**: 波士顿贝斯以色列女执事医疗中心
- **特点**: 高质量临床标注

### ChestXray14 (测试数据)
- **图像数量**: 25,596 张 (测试集)
- **类别数**: 14 个病理类别
- **来源**: 美国国立卫生研究院 (NIH)
- **特点**: 大规模公开数据集

---

## 类别对应关系

虽然两个数据集都有 14 个类别，但名称略有不同：

| MIMIC-CXR | ChestXray14 | 说明 |
|-----------|-------------|------|
| Atelectasis | Atelectasis | ✓ 相同 |
| Cardiomegaly | Cardiomegaly | ✓ 相同 |
| Consolidation | Consolidation | ✓ 相同 |
| Edema | Edema | ✓ 相同 |
| **Pleural Effusion** | **Effusion** | 名称不同但含义相同 |
| Pneumonia | Pneumonia | ✓ 相同 |
| Pneumothorax | Pneumothorax | ✓ 相同 |
| Enlarged Cardiomediastinum | - | MIMIC 独有 |
| Fracture | - | MIMIC 独有 |
| Lung Lesion | - | MIMIC 独有 |
| Lung Opacity | - | MIMIC 独有 |
| No Finding | - | MIMIC 独有 |
| Pleural Other | - | MIMIC 独有 |
| Support Devices | - | MIMIC 独有 |
| - | Emphysema | ChestXray14 独有 |
| - | Fibrosis | ChestXray14 独有 |
| - | Hernia | ChestXray14 独有 |
| - | Infiltration | ChestXray14 独有 |
| - | Mass | ChestXray14 独有 |
| - | Nodule | ChestXray14 独有 |
| - | Pleural_Thickening | ChestXray14 独有 |

**重要提示**: 模型使用 zero-shot 方法，通过文本提示自动适配不同类别。

---

## 使用方法

### 1️⃣ 测试数据加载

首先验证 ChestXray14 数据是否正确加载：

```bash
python test_chestxray14_data.py
```

预期输出：
```
[OK] Data loaded successfully!
  Total test images: 25596
  Label shape: (25596, 14)
```

### 2️⃣ 跨数据集训练和测试

#### 方法 A: 使用批处理脚本（推荐）

```bash
run_cross_dataset_test.bat
```

#### 方法 B: 命令行

**仅 Zero-shot 测试**（无需训练）：
```bash
python main.py --method zeroshot --test_chestxray14
```

**完整流程**（训练 + 测试）：
```bash
python main.py --method finetune --test_chestxray14
```

**自定义参数**：
```bash
python main.py --method finetune --test_chestxray14 --batch_size 16 --epochs 30
```

### 3️⃣ 仅在 MIMIC 上测试（对比基线）

```bash
# 在 MIMIC 测试集上评估
python main.py --method finetune
```

---

## 参数说明

### 新增参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `--test_chestxray14` | flag | 使用 ChestXray14 数据集进行测试 |
| `--chestxray14_path` | str | 自定义 ChestXray14 路径（默认: `D:\Data\ChestXray14\CXR8`） |

### 完整示例

```bash
python main.py \
    --method finetune \
    --test_chestxray14 \
    --chestxray14_path "D:\Data\ChestXray14\CXR8" \
    --batch_size 32 \
    --epochs 50 \
    --lr 1e-5 \
    --experiment_name "mimic_to_chestxray14"
```

---

## 输出结果

### 文件命名规则

当使用 `--test_chestxray14` 时，结果文件名会自动添加后缀：

**MIMIC 测试**:
- `zeroshot_results.json`
- `finetune_results.json`

**ChestXray14 测试**:
- `zeroshot_on_ChestXray14_results.json`
- `finetune_on_ChestXray14_results.json`

### 结果目录

所有结果保存在：`C:\Users\admin\Desktop\mimic-baseline\results\mimic_clip\`

包含：
- 📊 评估指标 (JSON)
- 📈 性能可视化 (PNG)
- 💾 训练曲线
- 🎯 最佳模型权重

---

## 评估指标

### 多标签评估指标

对于两个数据集，都使用以下指标：

1. **Subset Accuracy**: 完全匹配的样本比例
2. **Hamming Loss**: 平均标签错误率
3. **Jaccard Score**: 样本级 IoU
4. **Per-class Metrics**:
   - Precision (精确率)
   - Recall (召回率)
   - F1-Score
   - AUC-ROC

5. **Macro/Micro 平均**:
   - Macro: 每个类别同等重要
   - Micro: 每个样本同等重要

---

## 实验设计建议

### 实验 1: 基线性能
```bash
# 在 MIMIC 上训练和测试
python main.py --method finetune
```

### 实验 2: 跨数据集泛化
```bash
# 在 MIMIC 训练，在 ChestXray14 测试
python main.py --method finetune --test_chestxray14
```

### 实验 3: Zero-shot 能力
```bash
# 不训练，直接在 ChestXray14 zero-shot
python main.py --method zeroshot --test_chestxray14
```

---

## 预期结果分析

### 性能预期

1. **MIMIC 测试集**（同分布）:
   - 预期性能最高
   - F1-score: 0.60 - 0.75

2. **ChestXray14 测试集**（跨分布）:
   - 性能会有所下降
   - F1-score: 0.40 - 0.60
   - 性能下降幅度反映模型泛化能力

### 类别性能差异

- **共有类别**（如 Pneumonia, Atelectasis）: 性能较好
- **独有类别**（如 Infiltration, Mass）: 依赖 zero-shot 能力

---

## 故障排查

### 问题 1: ChestXray14 数据未找到

**错误**:
```
FileNotFoundError: D:\Data\ChestXray14\CXR8\Data_Entry_2017_v2020.csv
```

**解决**:
```bash
# 指定正确路径
python main.py --test_chestxray14 --chestxray14_path "你的路径"
```

### 问题 2: 内存不足

**错误**:
```
RuntimeError: CUDA out of memory
```

**解决**:
```bash
# 减小 batch size
python main.py --test_chestxray14 --batch_size 16
```

### 问题 3: 图像路径错误

**症状**: 部分图像显示 [MISSING]

**解决**:
1. 检查图像是否在 `images/images/` 子目录
2. 确认图像文件扩展名为 `.png`

---

## 技术细节

### Zero-Shot 推理机制

1. **文本提示生成**:
   ```python
   # ChestXray14
   "chest x-ray showing atelectasis"
   "chest x-ray showing cardiomegaly"
   ...
   ```

2. **相似度计算**:
   - 图像特征 × 文本特征
   - Sigmoid 激活（多标签）
   - 阈值 0.5 二值化

3. **无需标签映射**:
   - 直接使用文本语义
   - 自动适配不同类别名称

### 数据流程

```
MIMIC-CXR (训练)
    ↓
训练 BiomedCLIP
    ↓
保存最佳模型
    ↓
加载 ChestXray14
    ↓
Zero-shot 推理
    ↓
评估和保存结果
```

---

## 引用

如果使用跨数据集功能，请引用：

**MIMIC-CXR**:
```bibtex
@article{johnson2019mimic,
  title={MIMIC-CXR, a de-identified publicly available database of chest radiographs with free-text reports},
  author={Johnson, Alistair EW and others},
  journal={Scientific data},
  year={2019}
}
```

**ChestXray14**:
```bibtex
@inproceedings{wang2017chestx,
  title={ChestX-ray8: Hospital-scale chest x-ray database and benchmarks on weakly-supervised classification and localization of common thorax diseases},
  author={Wang, Xiaosong and others},
  booktitle={CVPR},
  year={2017}
}
```

---

## 高级用法

### 多数据集对比实验

```bash
# 1. MIMIC 测试
python main.py --method finetune --experiment_name "mimic_baseline"

# 2. ChestXray14 测试
python main.py --method finetune --test_chestxray14 --experiment_name "chestxray14_transfer"

# 3. 对比结果
# 查看 results/mimic_clip/ 目录下的 JSON 文件
```

### 自定义类别

如需测试特定类别，可以修改 `chestxray14_dataset.py` 中的 `CHESTXRAY14_CLASSES` 列表。

---

**创建日期**: 2025-12-15
**项目路径**: C:\Users\admin\Desktop\mimic-baseline
**状态**: ✓ 跨数据集测试功能已实现并测试通过
