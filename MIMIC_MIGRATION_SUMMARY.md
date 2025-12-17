# MIMIC-CXR 数据集迁移总结

## 概述
已成功将项目从 **ISIC 2019 皮肤病变分类**（单标签）迁移到 **MIMIC-CXR 胸部X光分类**（多标签）。

## 数据集信息

### MIMIC-CXR 数据集
- **类型**: 多标签分类
- **类别数**: 14个 CheXpert 病理类别
- **总样本数**: 377,095 张胸部X光图像
- **数据分割**:
  - 训练集: 368,945 (97.8%)
  - 验证集: 2,991 (0.8%)
  - 测试集: 5,159 (1.4%)

### 类别列表
1. Atelectasis (肺不张)
2. Cardiomegaly (心脏肥大)
3. Consolidation (肺实变)
4. Edema (水肿)
5. Enlarged Cardiomediastinum (心纵隔增大)
6. Fracture (骨折)
7. Lung Lesion (肺部病变)
8. Lung Opacity (肺部混浊)
9. No Finding (未见异常)
10. Pleural Effusion (胸腔积液)
11. Pleural Other (其他胸膜异常)
12. Pneumonia (肺炎)
13. Pneumothorax (气胸)
14. Support Devices (支持设备)

## 主要修改

### 1. 新建文件

#### datasets/mimic_dataset.py
- **MIMICCXRDataset**: PyTorch 数据集类，支持多标签加载
- **MIMICCXRDataLoader**: 数据加载器
  - 自动构建图像路径（`files/p{XX}/p{subject_id}/s{study_id}/{dicom_id}.jpg`）
  - 处理不确定标签（-1.0）的策略
  - 支持官方训练/验证/测试分割
  - 可视化类别分布

#### test_mimic_data.py
- 测试 MIMIC-CXR 数据加载
- 验证图像路径是否正确
- 显示数据集统计信息

#### run_mimic_finetune.bat
- 一键启动 MIMIC-CXR 微调训练的批处理脚本

### 2. 修改的文件

#### config/config.py
**PathConfig**:
- `base_data_path`: 改为 `D:\Data\MIMIC`
- `output_dir`: 改为 `C:\Users\admin\Desktop\mimic-baseline\results\mimic_clip`

**DataConfig**:
- `batch_size`: 64 → 32（MIMIC图像更大）
- 新增 `use_provided_split`: 使用官方数据分割
- 新增 `label_policy`: 不确定标签处理策略

**ClassConfig**:
- `class_descriptions`: 改为14个 CheXpert 类别
- `task_type`: 新增，设为 `'multi-label'`
- `get_text_prompts()`: 默认模板改为 `"chest x-ray showing {description}"`

#### trainers/clip_trainer.py
**ZeroShotCLIPInference.predict()**:
- 支持多标签预测（使用 sigmoid 而非 softmax）
- 阈值设为 0.5

**CLIPTrainer**:
- `train_epoch()`:
  - 多标签使用 `BCEWithLogitsLoss`
  - 准确率计算改为多标签方式

- `validate()`:
  - 支持多标签评估
  - 返回预测概率用于 AUC 计算

#### evaluators/evaluator.py
**新增方法**:
- `_evaluate_multilabel()`: 多标签评估
  - Subset Accuracy（精确匹配率）
  - Hamming Loss
  - Jaccard Score
  - Per-class Precision/Recall/F1
  - AUC-ROC（如果提供预测分数）

- `_plot_multilabel_metrics()`: 多标签可视化
  - 每个类别的 Precision/Recall/F1 横向条形图

**修改方法**:
- `evaluate()`: 自动检测任务类型，分别调用单标签或多标签评估

#### main.py
- 导入改为 `MIMICCXRDataLoader`
- `run_zeroshot()`: 传递 `y_scores` 参数给评估器
- `run_finetune()`: 注释更新为支持多标签
- 数据加载改为调用 `MIMICCXRDataLoader`

#### datasets/__init__.py
- 导出 `MIMICCXRDataset` 和 `MIMICCXRDataLoader`

## 测试结果

✓ 数据加载成功：377,095 张图像
✓ 标签形状正确：(377,095, 14)
✓ 图像路径正确且存在
✓ 数据分割正常：
  - 训练集: 368,945 张
  - 验证集: 2,991 张
  - 测试集: 5,159 张

## 使用方法

### 测试数据加载
```bash
python test_mimic_data.py
```

### 运行微调训练
```bash
# 方法1：使用批处理脚本
run_mimic_finetune.bat

# 方法2：命令行
python main.py --method finetune

# 方法3：仅运行 Zero-shot
python main.py --method zeroshot

# 方法4：自定义参数
python main.py --method finetune --batch_size 16 --epochs 50 --lr 5e-6
```

### 查看结果
训练结果将保存在：`C:\Users\admin\Desktop\mimic-baseline\results\mimic_clip\`

包含：
- 类别分布图
- 训练曲线
- 每个类别的性能指标
- 评估结果 JSON 文件
- 最佳模型权重

## 关键特性

### 多标签分类支持
- ✓ 每个图像可以有多个标签
- ✓ 使用 BCEWithLogitsLoss 而非交叉熵
- ✓ Sigmoid 激活而非 Softmax
- ✓ 支持样本级和类别级的评估指标

### 不确定标签处理
MIMIC-CXR 的标签有三种值：
- `1.0`: 阳性
- `0.0`: 阴性
- `-1.0`: 不确定
- `NaN`: 未提及

支持三种处理策略（`label_policy`）：
- `'ignore_uncertain'`: -1.0 和 NaN 都视为 0（默认）
- `'as_positive'`: -1.0 视为 1，NaN 视为 0
- `'as_negative'`: -1.0 和 NaN 都视为 0

### 官方数据分割
使用 MIMIC-CXR 官方的训练/验证/测试分割，保持可复现性。

## 评估指标

### 多标签专用指标
- **Subset Accuracy**: 完全正确预测的样本比例
- **Hamming Loss**: 平均错误标签比例
- **Jaccard Score**: 样本级的 IoU

### 传统指标（每个类别）
- Precision（精确率）
- Recall（召回率）
- F1-Score
- AUC-ROC

### 汇总指标
- Macro 平均：每个类别同等重要
- Micro 平均：每个样本同等重要
- Samples 平均：每个样本的平均性能

## 注意事项

1. **内存占用**: MIMIC-CXR 数据集很大，建议：
   - 减小 batch size（已设为32）
   - 使用充足的系统内存（建议 32GB+）

2. **训练时间**: 由于数据集规模大，训练时间会较长：
   - 使用 GPU 加速
   - 考虑使用早停机制

3. **类别不平衡**: 某些类别样本很少（如 Pleural Other 只有0.92%）：
   - 可以使用 `--loss_type weighted` 处理
   - 或使用 `--loss_type focal` 关注难例

## 下一步建议

1. **数据采样**：对于快速实验，可以先用部分数据测试
2. **超参数调优**：调整学习率、batch size、epochs
3. **损失函数实验**：尝试 weighted 或 focal loss 处理类别不平衡
4. **数据增强**：添加胸部X光特定的数据增强方法
5. **模型集成**：训练多个模型并进行集成

## 兼容性说明

原有的 ISIC 2019 代码仍然保留在：
- `datasets/isic_dataset.py`

如果需要切换回 ISIC 2019，只需：
1. 修改 `config.py` 中的路径和类别配置
2. 修改 `main.py` 的导入语句
3. 将 `task_type` 改为 `'single-label'`

---

**创建日期**: 2025-12-15
**项目路径**: C:\Users\admin\Desktop\mimic-baseline
**状态**: ✓ 测试通过，可以开始训练
