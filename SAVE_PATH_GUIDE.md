# 保存路径自定义指南

## 概述

项目支持自定义保存路径和文件名前缀，方便管理训练结果。

## 两个关键参数

### 1. `--output_dir`
控制所有文件保存的根目录。

### 2. `--experiment_name`
控制保存文件的文件名前缀。

---

## 快速使用

```bash
# 默认保存（使用默认名称）
python main.py --method finetune

# 自定义输出目录
python main.py --method finetune --output_dir results/my_experiment

# 自定义实验名称
python main.py --method finetune --experiment_name exp1

# 同时自定义两者
python main.py --method finetune     --output_dir results/my_experiments     --experiment_name lr5e6
```

---

## 保存文件详解

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
├── finetune_best_model.pth                   # 最佳模型
├── finetune_results.json                     # 评估结果
├── finetune_confusion_matrix.png             # 混淆矩阵
└── finetune_per_class_recall.png             # 各类召回率
```

### 自定义实验名称

运行：
```bash
python main.py --method finetune     --output_dir results/my_experiment     --experiment_name lr5e6
```

生成文件：
```
results/my_experiment/
├── class_distribution.png                    # 数据集类别分布
├── lr5e6_training_curves.png                 # 训练曲线
├── lr5e6_best_model.pth                      # 最佳模型
├── lr5e6_results.json                        # 评估结果
├── lr5e6_confusion_matrix.png                # 混淆矩阵
└── lr5e6_per_class_recall.png                # 各类召回率
```

---

## 实用示例

### 示例 1: 学习率搜索

```bash
python main.py --method finetune     --lr 5e-6     --output_dir results/lr_search     --experiment_name lr5e6

python main.py --method finetune     --lr 1e-5     --output_dir results/lr_search     --experiment_name lr1e5
```

### 示例 2: 带时间戳的实验

```bash
# Linux/Mac
python main.py --method finetune     --experiment_name "exp_$(date +%Y%m%d_%H%M%S)"

# Windows (PowerShell)
python main.py --method finetune `
    --experiment_name exp_$(Get-Date -Format "yyyyMMdd_HHmmss")
```

### 示例 3: 按日期组织实验

```bash
python main.py --method finetune     --output_dir results/2024-12-10     --experiment_name exp_v1

python main.py --method finetune     --output_dir results/2024-12-10     --experiment_name exp_v2_tuned
```

---

## 推荐目录结构

### 方案 1: 按实验类型组织

```
results/
├── baseline/                    # 基准实验
│   ├── finetune_default/
│   └── zeroshot/
├── lr_tuning/                   # 学习率调参
│   ├── lr_5e6/
│   └── lr_1e5/
└── final_models/                # 最终模型
    └── best_model/
```

### 方案 2: 按日期组织

```
results/
├── 2024-12-10/
│   ├── exp1/
│   └── exp2/
├── 2024-12-11/
│   └── exp1_production/
└── archive/
    └── old_experiments/
```

---

## 高级技巧

### 1. 批量实验脚本

创建 `run_experiments.bat`:
```batch
@echo off
set BASE_DIR=results/batch_exp_%date:~0,4%%date:~5,2%%date:~8,2%

python main.py --method finetune --lr 5e-6 --output_dir %BASE_DIR% --experiment_name lr5e6
python main.py --method finetune --lr 1e-5 --output_dir %BASE_DIR% --experiment_name lr1e5
python main.py --method finetune --lr 2e-5 --output_dir %BASE_DIR% --experiment_name lr2e5

echo All experiments completed! Results in %BASE_DIR%
pause
```

### 2. 条件命名

```bash
# 根据学习率自动命名
LR_IMAGE="5e-6"
EXP_NAME="lr${LR_IMAGE}_$(date +%m%d)"

python main.py --method finetune     --lr $LR_IMAGE     --experiment_name $EXP_NAME
```

### 3. 实验记录

创建 `experiments.log`:
```bash
echo "Experiment: exp_v1" >> experiments.log
echo "Date: $(date)" >> experiments.log
echo "Command: python main.py --method finetune --lr 5e-6 ..." >> experiments.log
echo "---" >> experiments.log

python main.py --method finetune     --lr 5e-6     --experiment_name exp_v1
```

---

## 文件命名规则

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

## 注意事项

### 1. 文件覆盖
如果使用相同的 `output_dir` 和 `experiment_name`，会覆盖之前的文件。

### 2. 文件名限制
- Windows: 不能包含 `< > : " / \ | ? *`
- 推荐使用: `字母、数字、下划线、连字符`

```bash
# 正确
--experiment_name exp_v1
--experiment_name lr5e6_20241210

# 错误
--experiment_name "exp/v1"      # 包含 /
--experiment_name "lr:5e-6"     # 包含 :
```

### 3. 路径长度
Windows 路径总长度限制为 260 字符，注意不要设置太长的路径。

---

## 最佳实践

### 1. 使用有意义的命名
```bash
# 不好
--experiment_name exp1
--experiment_name test

# 好
--experiment_name lr5e6_v1
--experiment_name lr1e5_tuned
```

### 2. 包含关键参数
```bash
# 包含学习率、日期
--experiment_name lr5e6_1210
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

## 相关文档

- **命令行参数完整列表**: 运行 `python main.py --help`
- **损失函数说明**: `LOSS_FUNCTIONS_GUIDE.md`
- **快速开始**: `QUICK_START.md`

---

现在你可以完全控制实验结果的保存位置和命名了。
