# 实验名称自定义功能说明

## 功能概述

使用 `--experiment_name` 可以自定义保存文件的名称前缀，方便区分不同实验。

### 之前（固定文件名）
```bash
python main.py --method finetune
```
生成文件：
- `finetune_best_model.pth`
- `finetune_results.json`
- `training_curves.png`

### 现在（自定义文件名）
```bash
python main.py --method finetune --experiment_name my_exp_v1
```
生成文件：
- `my_exp_v1_best_model.pth`
- `my_exp_v1_results.json`
- `my_exp_v1_training_curves.png`

---

## 快速使用

```bash
# 自定义实验名称
python main.py --method finetune --experiment_name exp1

# 同时自定义输出目录和实验名称
python main.py --method finetune     --output_dir results/my_experiments     --experiment_name lr5e6
```

---

## 实用示例

### 示例 1: 学习率搜索
```bash
python main.py --method finetune --lr 5e-6 --experiment_name lr5e6
python main.py --method finetune --lr 1e-5 --experiment_name lr1e5
```

### 示例 2: 完整命名
```bash
python main.py --method finetune     --lr 5e-6     --output_dir results/production     --experiment_name lr5e6_final
```

---

## 影响的文件

| 原文件名 | 使用 `--experiment_name my_exp` 后 |
|---------|-----------------------------------|
| `finetune_best_model.pth` | `my_exp_best_model.pth` |
| `training_curves.png` | `my_exp_training_curves.png` |
| `finetune_results.json` | `my_exp_results.json` |
| `finetune_confusion_matrix.png` | `my_exp_confusion_matrix.png` |
| `finetune_per_class_recall.png` | `my_exp_per_class_recall.png` |

不受影响的文件：
- `class_distribution.png`

---

## 命名建议

```bash
# 格式: {key_param}_{version}
--experiment_name lr5e6_v1
--experiment_name lr1e5_v2
--experiment_name standard_baseline
```

---

## 批量实验

```batch
python main.py --method finetune --lr 5e-6 --experiment_name lr5e6
python main.py --method finetune --lr 1e-5 --experiment_name lr1e5
python main.py --method finetune --lr 2e-5 --experiment_name lr2e5
```

---

## 完整示例

```bash
for lr in 5e-6 1e-5 2e-5; do
    python main.py --method finetune         --lr $lr         --output_dir results/lr_search         --experiment_name lr_${lr/e/-}  # 5e-6 -> lr_5-6
done
```

---

## 相关文档

- `SAVE_PATH_GUIDE.md`
- `QUICK_START.md`
