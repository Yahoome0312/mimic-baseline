# 快速开始指南

## 一、环境准备

### 1. 安装依赖
```bash
cd C:\Users\admin\Desktop\baseline\isic2019\isic_clip_project
pip install -r requirements.txt
```

### 2. 检查数据路径
确保以下文件/文件夹存在：
- 数据集：`D:\Data\isic2019\ISIC_2019_Training_Input\` (图像文件夹)
- 标签：`D:\Data\isic2019\ISIC_2019_Training_GroundTruth.csv`
- 模型：`C:\Users\admin\Desktop\baseline\bimedclip-zs\checkpoints\` (BiomedCLIP 权重)

## 二、运行方式

### 方式1：双击批处理文件（推荐）

#### 运行所有方法（Zero-shot + Fine-tuning）
双击 `run_all.bat`

#### 仅运行 Zero-shot
双击 `run_zeroshot.bat`

#### 仅运行 Fine-tuning
双击 `run_finetune.bat`

#### 自定义参数运行
编辑 `run_custom_example.bat` 修改参数后运行

### 方式2：命令行运行

#### 基础运行
```bash
python main.py
```

#### 仅运行 Zero-shot
```bash
python main.py --method zeroshot
```

#### 仅运行 Fine-tuning
```bash
python main.py --method finetune
```

#### 自定义数据和输出路径
```bash
python main.py --data_path D:\Data\isic2019 --output_dir C:\Results\my_exp
```

#### 修改训练参数
```bash
python main.py --batch_size 32 --epochs 50 --lr_image 5e-6
```

## 三、常用参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--method` | 运行方法 (all/zeroshot/finetune) | all |
| `--data_path` | 数据集路径 | D:\Data\isic2019 |
| `--output_dir` | 结果输出路径 | results\isic_clip_loss |
| `--checkpoint_dir` | 模型权重路径 | bimedclip-zs\checkpoints |
| `--batch_size` | 批次大小 | 64 |
| `--epochs` | 训练轮数 | 100 |
| `--lr_image` | 图像编码器学习率 | 1e-5 |
| `--lr_text` | 文本嵌入学习率 | 1e-4 |
| `--patience` | 早停耐心值 | 10 |

## 四、输出文件说明

运行完成后，在输出目录下会生成：

### 可视化文件
- `class_distribution.png` - 数据集类别分布图
- `training_curves.png` - 训练曲线（loss 和 accuracy）
- `method1_zeroshot_confusion_matrix.png` - Zero-shot 混淆矩阵
- `method2_full_finetune_confusion_matrix.png` - Fine-tuning 混淆矩阵
- `method1_zeroshot_per_class_recall.png` - Zero-shot 各类别召回率
- `method2_full_finetune_per_class_recall.png` - Fine-tuning 各类别召回率
- `methods_comparison.png` - 方法对比图

### 结果文件
- `method1_zeroshot_results.json` - Zero-shot 详细结果
- `method2_full_finetune_results.json` - Fine-tuning 详细结果
- `methods_comparison.csv` - 方法对比表格

### 模型文件
- `method2_best_model.pth` - 最佳微调模型权重

## 五、完整示例

```bash
# 示例1：快速测试（仅 Zero-shot）
python main.py --method zeroshot

# 示例2：完整训练（默认参数）
python main.py --method all

# 示例3：自定义实验
python main.py \
    --method finetune \
    --data_path D:\Data\isic2019 \
    --output_dir C:\Results\experiment_001 \
    --batch_size 32 \
    --epochs 100 \
    --lr_image 1e-5 \
    --lr_text 1e-4 \
    --patience 10 \
    --seed 42

# 示例4：小批量快速测试
python main.py \
    --method finetune \
    --batch_size 16 \
    --epochs 10 \
    --patience 3
```

## 六、常见问题

### Q1: CUDA Out of Memory
**解决方案**：减小 batch_size
```bash
python main.py --batch_size 16
```

### Q2: 找不到数据文件
**解决方案**：检查数据路径，或使用 `--data_path` 指定
```bash
python main.py --data_path YOUR_DATA_PATH
```

### Q3: 训练速度慢
**解决方案**：
1. 确保使用 GPU：检查 CUDA 是否可用
2. 减少 num_workers：
```bash
python main.py --num_workers 2
```

### Q4: 修改默认配置
**解决方案**：
1. 编辑 `config/config.py` 修改默认值
2. 或使用命令行参数覆盖
3. 参考 `config_example.py` 的示例

## 七、评估指标说明

- **Accuracy**: 整体准确率
- **Balanced Accuracy**: 平衡准确率（各类召回率的平均值）
- **F1-score (Macro)**: 各类 F1 值的算术平均（适合不平衡数据）
- **F1-score (Weighted)**: 按样本数加权的 F1 平均值
- **Per-class F1**: 每个类别单独的 F1 分数
- **Per-class Recall**: 每个类别的召回率 (TP/(TP+FN))

## 八、获取帮助

查看所有可用参数：
```bash
python main.py --help
```

## 九、下一步

1. 查看 `README.md` 了解详细文档
2. 参考 `config_example.py` 学习如何编程配置
3. 修改 `config/config.py` 自定义默认配置
4. 根据实验结果调整超参数
