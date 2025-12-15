# 实验名称自定义功能说明

## ✅ 新增功能

已成功添加 `--experiment_name` 参数，让你可以自定义所有保存文件的名称！

## 🎯 功能概述

### 之前（固定文件名）
```bash
python main.py --method finetune
```
生成文件：
- ❌ `method2_best_model.pth` （名称固定）
- ❌ `method2_full_finetune_results.json` （名称固定）
- ❌ `training_curves.png` （名称固定）

### 现在（自定义文件名）
```bash
python main.py --method finetune --experiment_name my_exp_v1
```
生成文件：
- ✅ `my_exp_v1_best_model.pth` （可自定义）
- ✅ `my_exp_v1_results.json` （可自定义）
- ✅ `my_exp_v1_training_curves.png` （可自定义）

## 🚀 快速使用

### 基础用法
```bash
# 自定义实验名称
python main.py --method finetune --experiment_name weighted_exp1

# 同时自定义输出目录和实验名称
python main.py --method finetune \
    --output_dir results/my_experiments \
    --experiment_name weighted_lr5e6
```

### 实用示例
```bash
# 示例 1: 损失函数对比实验
python main.py --method finetune --loss_type weighted --experiment_name weighted_inv
python main.py --method finetune --loss_type focal --experiment_name focal_g2.5

# 示例 2: 学习率搜索
python main.py --method finetune --lr_image 5e-6 --experiment_name lr5e6
python main.py --method finetune --lr_image 1e-5 --experiment_name lr1e5

# 示例 3: 完整命名
python main.py --method finetune \
    --loss_type weighted \
    --lr_image 5e-6 \
    --output_dir results/production \
    --experiment_name weighted_eff_lr5e6_final
```

## 📂 影响的文件

### 所有受影响的保存文件

| 原文件名 | 使用 `--experiment_name my_exp` 后 |
|---------|-----------------------------------|
| `method2_best_model.pth` | `my_exp_best_model.pth` |
| `training_curves.png` | `my_exp_training_curves.png` |
| `method2_full_finetune_results.json` | `my_exp_results.json` |
| `method2_full_finetune_confusion_matrix.png` | `my_exp_confusion_matrix.png` |
| `method2_full_finetune_per_class_recall.png` | `my_exp_per_class_recall.png` |

### 不受影响的文件
- `class_distribution.png` （数据集统计，不随实验改变）

## 🔧 技术实现

### 修改的文件
1. **main.py**
   - ✅ 添加 `--experiment_name` 参数
   - ✅ 修改 `run_zeroshot()` 和 `run_finetune()` 接受实验名称
   - ✅ 传递实验名称给 evaluator 和 trainer

2. **trainers/clip_trainer.py**
   - ✅ `CLIPTrainer.__init__()` 接受 `experiment_name` 参数
   - ✅ `_plot_training_curves()` 使用自定义文件名

3. **README.md**
   - ✅ 添加参数说明

### 向后兼容性
✅ 完全向后兼容！如果不指定 `--experiment_name`，使用默认名称：
- Zero-shot: `method1_zeroshot`
- Fine-tuning: `method2_full_finetune`

## 💡 使用建议

### 1. 命名规范
推荐使用有意义的名称，包含关键信息：
```bash
# 格式: {loss_type}_{key_param}_{version}
--experiment_name weighted_lr5e6_v1
--experiment_name focal_g2.5_effective
--experiment_name standard_baseline
```

### 2. 实验组织
按类型组织实验：
```bash
# 基准实验
--output_dir results/baseline --experiment_name standard_default

# 调参实验
--output_dir results/tuning --experiment_name lr_search_exp1

# 最终模型
--output_dir results/final --experiment_name production_v1
```

### 3. 批量实验
创建脚本批量运行：
```batch
python main.py --method finetune --loss_type standard --experiment_name standard
python main.py --method finetune --loss_type weighted --experiment_name weighted
python main.py --method finetune --loss_type focal --experiment_name focal
```

## 📊 完整示例

### 示例 1: 对比不同损失函数
```bash
# 创建对比实验目录
mkdir results\loss_comparison

# 运行三个实验
python main.py --method finetune --loss_type standard \
    --output_dir results/loss_comparison \
    --experiment_name 1_standard_baseline

python main.py --method finetune --loss_type weighted \
    --output_dir results/loss_comparison \
    --experiment_name 2_weighted_inverse

python main.py --method finetune --loss_type focal --focal_alpha \
    --output_dir results/loss_comparison \
    --experiment_name 3_focal_alpha_g2.5

# 结果会整齐地保存在同一目录下，文件名清晰区分
```

### 示例 2: 学习率调参
```bash
for lr in 5e-6 1e-5 2e-5; do
    python main.py --method finetune \
        --lr_image $lr \
        --output_dir results/lr_search \
        --experiment_name lr_${lr/e/-}  # 5e-6 -> lr_5-6
done
```

## 📖 相关文档

- **详细指南**: `SAVE_PATH_GUIDE.md` - 完整的保存路径自定义指南
- **快速开始**: `QUICK_START.md` - 快速上手指南
- **命令行参数**: 运行 `python main.py --help` 查看所有参数

## ⚠️ 注意事项

1. **文件覆盖**: 相同的 `output_dir` + `experiment_name` 会覆盖之前的文件
2. **文件名限制**: Windows 下不能使用 `< > : " / \ | ? *` 等字符
3. **路径长度**: 注意 Windows 路径总长度限制（260字符）

## 🎉 总结

现在你可以：
- ✅ 自定义所有保存文件的名称
- ✅ 更好地组织实验结果
- ✅ 避免文件名混乱
- ✅ 批量运行实验更方便

享受更灵活的实验管理！ 🚀
