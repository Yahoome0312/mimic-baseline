# 保存路径自定义指南

## 📁 概述

项目现在支持完全自定义保存路径和文件名，让你可以更好地组织实验结果。

## 🎯 两个关键参数

### 1. `--output_dir` - 输出目录
控制所有文件保存的**根目录**

### 2. `--experiment_name` - 实验名称
控制保存文件的**文件名前缀**

---

## 🚀 快速使用

### 基础用法

```bash
# 默认保存（使用默认名称）
python main.py --method finetune

# 自定义输出目录
python main.py --method finetune --output_dir results/my_experiment

# 自定义实验名称
python main.py --method finetune --experiment_name weighted_exp1

# 同时自定义两者
python main.py --method finetune \
    --output_dir results/weighted_experiments \
    --experiment_name weighted_lr5e6
```

---

## 📂 保存文件详解

### 默认情况（不指定 experiment_name）

运行：
```bash
python main.py --method finetune --output_dir results/default
```

生成文件：
```
results/default/
├── class_distribution.png                    # 数据集类别分布
├── training_curves.png                       # 训练曲线
├── method2_best_model.pth                    # 最佳模型
├── method2_full_finetune_results.json        # 评估结果
├── method2_full_finetune_confusion_matrix.png   # 混淆矩阵
└── method2_full_finetune_per_class_recall.png   # 各类召回率
```

### 自定义实验名称

运行：
```bash
python main.py --method finetune \
    --output_dir results/weighted_exp \
    --experiment_name weighted_inv_lr5e6
```

生成文件：
```
results/weighted_exp/
├── class_distribution.png                         # 数据集类别分布
├── weighted_inv_lr5e6_training_curves.png         # 训练曲线
├── weighted_inv_lr5e6_best_model.pth              # 最佳模型
├── weighted_inv_lr5e6_results.json                # 评估结果
├── weighted_inv_lr5e6_confusion_matrix.png        # 混淆矩阵
└── weighted_inv_lr5e6_per_class_recall.png        # 各类召回率
```

---

## 💡 实用示例

### 示例 1: 组织不同损失函数的实验

```bash
# Standard CLIP Loss
python main.py --method finetune \
    --loss_type standard \
    --output_dir results/loss_comparison \
    --experiment_name standard_baseline

# Weighted CLIP Loss
python main.py --method finetune \
    --loss_type weighted \
    --output_dir results/loss_comparison \
    --experiment_name weighted_inverse

# Focal CLIP Loss
python main.py --method finetune \
    --loss_type focal \
    --focal_alpha \
    --output_dir results/loss_comparison \
    --experiment_name focal_gamma2.5
```

结果目录：
```
results/loss_comparison/
├── standard_baseline_best_model.pth
├── standard_baseline_results.json
├── weighted_inverse_best_model.pth
├── weighted_inverse_results.json
├── focal_gamma2.5_best_model.pth
└── focal_gamma2.5_results.json
```

### 示例 2: 学习率搜索实验

```bash
# 实验 1: lr_image=5e-6
python main.py --method finetune \
    --lr_image 5e-6 \
    --output_dir results/lr_search \
    --experiment_name lr_img5e6_txt1e4

# 实验 2: lr_image=1e-5
python main.py --method finetune \
    --lr_image 1e-5 \
    --output_dir results/lr_search \
    --experiment_name lr_img1e5_txt1e4

# 实验 3: lr_image=2e-5
python main.py --method finetune \
    --lr_image 2e-5 \
    --output_dir results/lr_search \
    --experiment_name lr_img2e5_txt1e4
```

### 示例 3: 带时间戳的实验

```bash
# Linux/Mac
python main.py --method finetune \
    --experiment_name "exp_$(date +%Y%m%d_%H%M%S)"

# Windows (PowerShell)
python main.py --method finetune `
    --experiment_name exp_$(Get-Date -Format "yyyyMMdd_HHmmss")

# Windows (批处理)
# 创建 run_with_timestamp.bat:
# set timestamp=%date:~0,4%%date:~5,2%%date:~8,2%_%time:~0,2%%time:~3,2%%time:~6,2%
# python main.py --method finetune --experiment_name exp_%timestamp%
```

### 示例 4: 按日期组织实验

```bash
# 2024年12月10日的实验
python main.py --method finetune \
    --output_dir results/2024-12-10 \
    --experiment_name weighted_v1

python main.py --method finetune \
    --output_dir results/2024-12-10 \
    --experiment_name weighted_v2_tuned
```

### 示例 5: 完整的实验命名规范

```bash
# 命名格式: {loss}_{method}_{lr}_{date}
python main.py --method finetune \
    --loss_type weighted \
    --class_weight_method effective \
    --lr_image 5e-6 \
    --lr_text 1e-4 \
    --output_dir results/production \
    --experiment_name weighted_eff_lr5e6-1e4_20241210
```

---

## 🗂️ 推荐目录结构

### 方案 1: 按实验类型组织

```
results/
├── baseline/                    # 基准实验
│   ├── standard_default/
│   └── zeroshot/
├── loss_comparison/             # 损失函数对比
│   ├── standard/
│   ├── weighted_inverse/
│   ├── weighted_effective/
│   └── focal_gamma2.5/
├── lr_tuning/                   # 学习率调参
│   ├── lr_5e6/
│   ├── lr_1e5/
│   └── lr_2e5/
└── final_models/                # 最终模型
    └── best_weighted/
```

### 方案 2: 按日期组织

```
results/
├── 2024-12-10/
│   ├── exp1_weighted/
│   ├── exp2_focal/
│   └── exp3_tuned/
├── 2024-12-11/
│   └── exp1_production/
└── archive/
    └── old_experiments/
```

---

## 🔧 高级技巧

### 1. 批量实验脚本

创建 `run_experiments.bat`:
```batch
@echo off
set BASE_DIR=results/batch_exp_%date:~0,4%%date:~5,2%%date:~8,2%

python main.py --method finetune --loss_type standard --output_dir %BASE_DIR% --experiment_name standard
python main.py --method finetune --loss_type weighted --output_dir %BASE_DIR% --experiment_name weighted
python main.py --method finetune --loss_type focal --focal_alpha --output_dir %BASE_DIR% --experiment_name focal

echo All experiments completed! Results in %BASE_DIR%
pause
```

### 2. 条件命名

```bash
# 根据损失函数类型自动命名
LOSS_TYPE="weighted"
LR_IMAGE="5e-6"
EXP_NAME="${LOSS_TYPE}_lr${LR_IMAGE}_$(date +%m%d)"

python main.py --method finetune \
    --loss_type $LOSS_TYPE \
    --lr_image $LR_IMAGE \
    --experiment_name $EXP_NAME
```

### 3. 实验记录

创建 `experiments.log`:
```bash
echo "Experiment: weighted_v1" >> experiments.log
echo "Date: $(date)" >> experiments.log
echo "Command: python main.py --method finetune --loss_type weighted ..." >> experiments.log
echo "---" >> experiments.log

python main.py --method finetune \
    --loss_type weighted \
    --experiment_name weighted_v1
```

---

## 📊 文件命名规则

### 训练相关文件
- 模型: `{experiment_name}_best_model.pth`
- 训练曲线: `{experiment_name}_training_curves.png`

### 评估相关文件
- 结果JSON: `{experiment_name}_results.json`
- 混淆矩阵: `{experiment_name}_confusion_matrix.png`
- 各类召回率: `{experiment_name}_per_class_recall.png`

### 数据相关文件（不受 experiment_name 影响）
- 类别分布: `class_distribution.png`

---

## ⚠️ 注意事项

### 1. 文件覆盖
如果使用相同的 `output_dir` 和 `experiment_name`，**会覆盖**之前的文件！

```bash
# 第一次运行
python main.py --experiment_name test_exp  # 创建文件

# 第二次运行（会覆盖！）
python main.py --experiment_name test_exp  # 覆盖之前的文件
```

**解决方案**：使用不同的实验名称或添加版本号/时间戳

### 2. 文件名限制
- Windows: 不能包含 `< > : " / \ | ? *`
- 推荐使用: `字母、数字、下划线、连字符`

```bash
# ✅ 正确
--experiment_name weighted_v1
--experiment_name focal_gamma2.5_20241210
--experiment_name exp-lr5e6

# ❌ 错误
--experiment_name "weighted/v1"      # 包含 /
--experiment_name "focal:gamma2.5"   # 包含 :
```

### 3. 路径长度
Windows 路径总长度限制为 260 字符，注意不要设置太长的路径。

---

## 🎓 最佳实践

### 1. 使用有意义的命名
```bash
# ❌ 不好
--experiment_name exp1
--experiment_name test

# ✅ 好
--experiment_name weighted_inverse_lr5e6
--experiment_name focal_gamma2.5_effective
```

### 2. 包含关键参数
```bash
# 包含损失函数、学习率、日期
--experiment_name weighted_lr5e6_1210

# 包含损失函数、权重方法、gamma
--experiment_name focal_eff_g2.5
```

### 3. 使用版本控制
```bash
--experiment_name baseline_v1
--experiment_name baseline_v2_tuned
--experiment_name baseline_v3_final
```

### 4. 分类存储
```bash
# 基准实验
--output_dir results/baseline --experiment_name standard_default

# 调参实验
--output_dir results/tuning --experiment_name lr_search_v1

# 生产模型
--output_dir results/production --experiment_name final_model
```

---

## 📝 快速参考

| 需求 | 命令示例 |
|------|---------|
| 更改保存目录 | `--output_dir results/my_exp` |
| 自定义文件名 | `--experiment_name my_exp_v1` |
| 两者都改 | `--output_dir results/test --experiment_name exp1` |
| 默认设置 | 不加任何参数 |

---

## 🔗 相关文档

- **命令行参数完整列表**: 运行 `python main.py --help`
- **损失函数使用**: `LOSS_FUNCTIONS_GUIDE.md`
- **快速开始**: `QUICK_START.md`

---

现在你可以完全控制实验结果的保存位置和命名了！ 🎉
