# 项目结构说明

## 目录树

```
isic_clip_project/
│
├── config/                          # 配置模块
│   ├── __init__.py                 # 模块初始化
│   └── config.py                   # 配置类定义（路径、模型、数据、训练参数）
│
├── datasets/                        # 数据集模块
│   ├── __init__.py                 # 模块初始化
│   └── isic_dataset.py             # ISIC 2019 数据集类和数据加载器
│
├── models/                          # 模型模块
│   ├── __init__.py                 # 模块初始化
│   └── clip_model.py               # CLIP 模型、损失函数、模型加载器
│
├── trainers/                        # 训练器模块
│   ├── __init__.py                 # 模块初始化
│   └── clip_trainer.py             # Zero-shot 推理和微调训练器
│
├── evaluators/                      # 评估器模块
│   ├── __init__.py                 # 模块初始化
│   └── evaluator.py                # 模型评估和结果对比
│
├── utils/                           # 工具模块
│   ├── __init__.py                 # 模块初始化
│   └── helpers.py                  # 辅助函数（早停、随机种子等）
│
├── main.py                          # 主入口文件
├── config_example.py                # 配置示例
├── __init__.py                      # 项目初始化
│
├── requirements.txt                 # Python 依赖
├── README.md                        # 详细文档（英文）
├── QUICK_START.md                   # 快速开始指南（中文）
├── PROJECT_STRUCTURE.md             # 本文件
├── .gitignore                       # Git 忽略文件
│
├── run_all.bat                      # 运行所有方法
├── run_zeroshot.bat                 # 仅运行 Zero-shot
├── run_finetune.bat                 # 仅运行 Fine-tuning
└── run_custom_example.bat           # 自定义参数示例
```

## 模块详解

### 1. config/ - 配置模块

#### config.py
定义了所有配置类：

- **PathConfig**: 路径配置
  - `local_checkpoints_dir`: BiomedCLIP 模型权重路径
  - `base_data_path`: ISIC 2019 数据集路径
  - `output_dir`: 结果输出路径

- **ModelConfig**: 模型配置
  - `model_name`: 模型名称
  - `context_length`: 文本上下文长度
  - `temperature`: CLIP 损失温度参数

- **DataConfig**: 数据配置
  - `batch_size`: 批次大小
  - `num_workers`: 数据加载线程数
  - `test_size`: 测试集比例
  - `val_size`: 验证集比例
  - `random_state`: 随机种子

- **TrainingConfig**: 训练配置
  - `learning_rate_image`: 图像编码器学习率
  - `learning_rate_text`: 文本嵌入学习率
  - `epochs`: 训练轮数
  - `weight_decay`: 权重衰减
  - `early_stopping_patience`: 早停耐心值
  - `use_scheduler`: 是否使用学习率调度器
  - `scheduler_type`: 调度器类型

- **ClassConfig**: 类别配置
  - `class_descriptions`: 8 个皮肤病变类别的描述
  - `class_names`: 类别名称列表
  - `num_classes`: 类别数量
  - `get_text_prompts()`: 生成文本提示

- **Config**: 主配置类
  - 整合所有子配置
  - 提供配置更新、打印、转换等方法

### 2. datasets/ - 数据集模块

#### isic_dataset.py

- **ISIC2019Dataset**: PyTorch 数据集类
  - 继承自 `torch.utils.data.Dataset`
  - 加载图像和标签
  - 应用图像变换
  - 处理文本提示

- **ISIC2019DataLoader**: 数据加载和预处理
  - `load_data()`: 加载 ISIC 2019 数据集
  - `split_dataset()`: 分割训练/验证/测试集
  - `create_dataloaders()`: 创建 DataLoader 实例
  - `_print_class_distribution()`: 打印类别分布
  - `_plot_class_distribution()`: 绘制类别分布图

### 3. models/ - 模型模块

#### clip_model.py

- **CLIPLoss**: CLIP 对比学习损失函数
  - 实现图像-文本对比损失
  - 双向交叉熵损失（image→text + text→image）
  - 可调节温度参数

- **CLIPFineTune**: 微调 CLIP 模型
  - 封装 CLIP 图像编码器
  - 可学习的类别文本嵌入
  - 特征归一化
  - 返回图像和文本特征

- **BiomedCLIPLoader**: BiomedCLIP 模型加载器
  - 从本地加载预训练模型
  - 加载分词器和预处理器
  - 注册模型配置
  - 创建分词器包装函数

### 4. trainers/ - 训练器模块

#### clip_trainer.py

- **ZeroShotCLIPInference**: Zero-shot 推理
  - 使用预训练 CLIP 进行零样本分类
  - 编码文本提示
  - 计算图像-文本相似度
  - 返回预测、标签和置信度

- **CLIPTrainer**: CLIP 微调训练器
  - 完整的训练循环
  - 早停机制
  - 学习率调度
  - 最佳模型保存
  - 训练曲线可视化
  - 支持分层学习率（图像编码器 vs 文本嵌入）

### 5. evaluators/ - 评估器模块

#### evaluator.py

- **ModelEvaluator**: 模型评估类
  - 计算多种评估指标
    - Accuracy
    - Balanced Accuracy
    - F1-score (Macro, Weighted, Per-class)
    - Recall (Per-class)
  - 生成混淆矩阵
  - 绘制评估可视化
    - 混淆矩阵热图
    - 各类别召回率柱状图
  - 保存评估结果（JSON）
  - 方法对比功能

### 6. utils/ - 工具模块

#### helpers.py

- **EarlyStopping**: 早停机制
  - 监控验证指标
  - 自动保存最佳模型
  - 可配置耐心值和改进阈值
  - 支持最大化或最小化指标

- **set_seed()**: 设置随机种子
  - Python random
  - NumPy
  - PyTorch (CPU + CUDA)
  - 确保结果可复现

- **count_parameters()**: 统计模型参数数量

- **get_device()**: 获取计算设备

- **print_device_info()**: 打印设备信息
  - PyTorch 版本
  - CUDA 可用性
  - GPU 信息

### 7. main.py - 主入口

- 命令行参数解析
- 配置初始化和更新
- 模型加载
- 数据加载和分割
- 运行 Zero-shot 和/或 Fine-tuning
- 结果评估和对比
- 完整的实验流程控制

## 数据流程

```
1. 数据加载
   ISIC2019DataLoader.load_data()
   → 读取 CSV 标签
   → 构建图像路径和标签列表
   → 绘制类别分布

2. 数据分割
   ISIC2019DataLoader.split_dataset()
   → 分层采样分割
   → 返回训练/验证/测试集

3. 创建 DataLoader
   ISIC2019DataLoader.create_dataloaders()
   → 创建 ISIC2019Dataset 实例
   → 应用图像预处理
   → 返回 PyTorch DataLoader

4. 模型加载
   BiomedCLIPLoader.load_model()
   → 读取配置文件
   → 加载预训练权重
   → 返回模型、分词器、预处理器

5. 训练/推理
   ZeroShotCLIPInference.predict()  # Zero-shot
   或
   CLIPTrainer.train()              # Fine-tuning
   → 训练循环
   → 早停检查
   → 保存最佳模型

6. 评估
   ModelEvaluator.evaluate()
   → 计算评估指标
   → 生成可视化
   → 保存结果
```

## 配置流程

```
1. 默认配置
   Config() 创建默认配置实例

2. 命令行参数
   parse_args() 解析命令行参数

3. 配置更新
   update_config_from_args() 用命令行参数更新配置

4. 配置使用
   各模块接收 config 对象
   通过 config.paths.xxx, config.training.xxx 等访问配置
```

## 模块依赖关系

```
main.py
├── config (配置)
├── models (加载 BiomedCLIP)
├── datasets (加载数据)
├── trainers (训练/推理)
│   ├── models (使用模型和损失)
│   └── utils (使用早停)
└── evaluators (评估)
    └── config (使用类别信息)
```

## 扩展指南

### 添加新的数据集
1. 在 `datasets/` 创建新的数据集类
2. 继承 `torch.utils.data.Dataset`
3. 实现 `__getitem__` 和 `__len__`
4. 在 `main.py` 中替换数据加载逻辑

### 添加新的训练方法
1. 在 `trainers/` 创建新的训练器类
2. 实现 `train()` 和 `predict()` 方法
3. 在 `main.py` 中添加新的运行函数

### 修改评估指标
1. 编辑 `evaluators/evaluator.py`
2. 在 `evaluate()` 方法中添加新指标
3. 更新可视化函数

### 自定义损失函数
1. 在 `models/clip_model.py` 添加新的损失类
2. 继承 `nn.Module`
3. 实现 `forward()` 方法
4. 在训练器中使用新损失

## 文件命名规范

- Python 文件：小写 + 下划线（例如：`clip_model.py`）
- 类名：大驼峰（例如：`CLIPFineTune`）
- 函数名：小写 + 下划线（例如：`load_model()`）
- 常量：大写 + 下划线（例如：`NUM_CLASSES`）
- 私有方法：前缀下划线（例如：`_plot_confusion_matrix()`）

## 代码风格

- 遵循 PEP 8 规范
- 使用类型提示（type hints）
- 详细的文档字符串（docstrings）
- 清晰的注释
- 模块化设计
- 单一职责原则
