# 损失函数集成总结

## ✅ 集成完成

已成功将 **Weighted CLIP Loss** 和 **Focal CLIP Loss** 集成到项目中！

## 📝 修改文件清单

### 修改了 3 个核心文件：

1. **models/clip_model.py**
   - ✅ 添加 `WeightedCLIPLoss` 类
   - ✅ 添加 `FocalCLIPLoss` 类
   - ✅ 添加 `compute_class_weights()` 函数
   - ✅ 修改 `CLIPLoss.forward()` 支持 labels 参数

2. **config/config.py**
   - ✅ 在 `TrainingConfig` 中添加损失函数配置：
     - `loss_type`: 损失函数类型
     - `class_weight_method`: 权重计算方法
     - `focal_gamma`: Focal loss gamma 参数
     - `focal_alpha`: 是否使用类别权重

3. **trainers/clip_trainer.py**
   - ✅ 更新导入语句
   - ✅ 修改 `CLIPTrainer.__init__()` 接受 `train_labels`
   - ✅ 添加 `_setup_loss_function()` 方法
   - ✅ 修改 `train_epoch()` 传递 labels
   - ✅ 修改 `validate()` 传递 labels

4. **main.py**
   - ✅ 添加命令行参数（`--loss_type`, `--focal_gamma` 等）
   - ✅ 修改 `update_config_from_args()` 处理新参数
   - ✅ 修改 `run_finetune()` 传递 `train_labels`

### 更新了 2 个模块文件：

5. **models/__init__.py** - 导出新的损失函数类
6. **trainers/__init__.py** - (无需修改)

### 新增了 5 个文件：

7. **LOSS_FUNCTIONS_GUIDE.md** - 完整使用指南
8. **run_weighted_loss.bat** - Weighted Loss 快速运行
9. **run_focal_loss.bat** - Focal Loss 快速运行
10. **run_focal_weighted_loss.bat** - 组合方式运行
11. **run_compare_losses.bat** - 对比实验脚本

### 更新了 1 个文档：

12. **README.md** - 添加损失函数说明

---

## 🎯 新增功能

### 1. 三种损失函数

| 损失函数 | 类名 | 用途 |
|---------|------|------|
| Standard | `CLIPLoss` | 标准 CLIP 损失（原有） |
| Weighted | `WeightedCLIPLoss` | 类别加权（新增） |
| Focal | `FocalCLIPLoss` | 难样本关注（新增） |

### 2. 类别权重计算

- `inverse`: 反比例权重
- `sqrt_inverse`: 平方根反比例
- `effective`: Effective Number 方法

### 3. 命令行参数

```bash
--loss_type {standard,weighted,focal}
--class_weight_method {inverse,sqrt_inverse,effective}
--focal_gamma FLOAT
--focal_alpha
```

---

## 🚀 使用方式

### 方式 1: 命令行

```bash
# Weighted Loss
python main.py --method finetune --loss_type weighted

# Focal Loss
python main.py --method finetune --loss_type focal

# Focal + Weights
python main.py --method finetune --loss_type focal --focal_alpha
```

### 方式 2: 双击批处理文件

- `run_weighted_loss.bat`
- `run_focal_loss.bat`
- `run_focal_weighted_loss.bat`
- `run_compare_losses.bat`

### 方式 3: 修改配置文件

编辑 `config/config.py`:
```python
loss_type: str = "weighted"
class_weight_method: str = "inverse"
```

---

## 🔍 技术细节

### 损失函数调用链

```
main.py
  └─> run_finetune(train_labels=y_train)
       └─> CLIPTrainer(train_labels=y_train)
            └─> _setup_loss_function(train_labels)
                 ├─> compute_class_weights(train_labels)  # 如果需要
                 └─> CLIPLoss / WeightedCLIPLoss / FocalCLIPLoss
                      └─> forward(image_features, text_features, labels)
```

### 关键设计

1. **向后兼容**：`CLIPLoss.forward()` 的 `labels` 参数是可选的
2. **自动权重计算**：在 `_setup_loss_function()` 中自动计算类别权重
3. **灵活配置**：通过配置文件或命令行参数控制

---

## ⚠️ 重要说明

### 1. 兼容性
- ✅ 不影响原有的 Standard CLIP Loss
- ✅ 不影响 Zero-shot 方法
- ✅ 所有原有功能正常运行

### 2. 默认行为
- 默认使用 `standard` CLIP Loss（与之前一致）
- 需要显式指定 `--loss_type` 才会使用新损失函数

### 3. 训练标签
- Weighted 和 Focal Loss 需要训练标签来计算权重
- 已自动从 `data_splits['train']` 获取

---

## 📊 预期效果

使用 Weighted 或 Focal Loss 后：
- ✅ 小类别（如 DF, VASC）的 F1 分数提升
- ✅ Balanced Accuracy 提高
- ✅ 整体性能略有提升或持平
- ⚠️ 训练时间几乎不增加（Focal Loss 约增加 10%）

---

## 🧪 建议实验

### 对比实验
运行以下命令对比三种损失函数：

```bash
python run_compare_losses.bat
```

### 推荐顺序
1. 先试 Weighted Loss（最简单）
2. 如果效果不理想，试 Focal Loss
3. 对于严重不平衡，使用 Focal + Alpha

---

## 📚 参考文档

- **完整使用指南**: `LOSS_FUNCTIONS_GUIDE.md`
- **快速开始**: `QUICK_START.md`
- **项目结构**: `PROJECT_STRUCTURE.md`
- **主文档**: `README.md`

---

## ✅ 验证清单

- [x] Weighted CLIP Loss 实现
- [x] Focal CLIP Loss 实现
- [x] 类别权重计算函数
- [x] 配置文件支持
- [x] 命令行参数支持
- [x] 训练器集成
- [x] 向后兼容性测试
- [x] 文档完善
- [x] 批处理脚本
- [x] 使用示例

---

## 🎉 集成完成！

所有功能已完整集成，可以立即使用：

```bash
# 快速测试
python main.py --method finetune --loss_type weighted --epochs 10

# 完整训练
python main.py --method finetune --loss_type focal --focal_alpha
```

如有问题，请查看 `LOSS_FUNCTIONS_GUIDE.md` 获取详细帮助！
