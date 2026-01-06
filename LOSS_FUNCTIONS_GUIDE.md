# 损失函数使用指南

当前仅支持 **Standard CLIP Loss**（对比学习损失）。

## 使用方法

```bash
python main.py --method finetune
```

## 配置位置

温度参数在 `config/config.py` 的 `ModelConfig.temperature`。

