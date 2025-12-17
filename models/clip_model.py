"""
CLIP Model Module
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from open_clip import create_model_and_transforms, get_tokenizer
from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS


class CLIPLoss(nn.Module):
    """CLIP contrastive learning loss function"""

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, image_features, text_features, labels=None):
        """
        Args:
            image_features: (batch_size, feature_dim) normalized image features
            text_features: (batch_size, feature_dim) normalized text features
            labels: (optional) for compatibility with weighted/focal losses

        Returns:
            loss: contrastive learning loss
        """
        # Calculate similarity matrix
        logits = (image_features @ text_features.T) / self.temperature  # (batch_size, batch_size)

        # Labels: positive pairs on the diagonal
        batch_labels = torch.arange(len(image_features), device=image_features.device)

        # Calculate cross-entropy loss in both directions
        loss_i2t = F.cross_entropy(logits, batch_labels)  # image to text
        loss_t2i = F.cross_entropy(logits.T, batch_labels)  # text to image

        # Total loss is the average of both directions
        loss = (loss_i2t + loss_t2i) / 2

        return loss


class WeightedCLIPLoss(nn.Module):
    """
    Weighted CLIP Loss - 处理类别不平衡

    原理：
    - 小类样本的 loss 贡献更大（通过类别权重）
    - 公式：weighted_loss = class_weight[label] * original_loss
    """

    def __init__(self, class_weights=None, temperature=0.07):
        """
        Args:
            class_weights: (num_classes,) tensor，每个类别的权重
            temperature: CLIP温度参数
        """
        super().__init__()
        self.temperature = temperature
        self.class_weights = class_weights

        if class_weights is not None:
            print(f"\n✓ Initialized Weighted CLIP Loss")
            print(f"  Temperature: {temperature}")
            print(f"  Class weights: {class_weights.cpu().numpy()}")
        else:
            print(f"\n✓ Initialized Standard CLIP Loss (no weights)")

    def forward(self, image_features, text_features, labels):
        """
        Args:
            image_features: (batch_size, feature_dim)
            text_features: (batch_size, feature_dim)
            labels: (batch_size,) 类别标签，用于权重

        Returns:
            loss: 加权后的对比学习损失
        """
        # 计算相似度矩阵
        logits = (image_features @ text_features.T) / self.temperature

        # 正样本在对角线上
        batch_labels = torch.arange(len(image_features), device=image_features.device)

        # 计算交叉熵损失（不求平均，保留每个样本的 loss）
        loss_i2t = F.cross_entropy(logits, batch_labels, reduction='none')
        loss_t2i = F.cross_entropy(logits.T, batch_labels, reduction='none')

        # 应用类别权重
        if self.class_weights is not None:
            weights = self.class_weights[labels]  # (batch_size,)
            loss_i2t = (loss_i2t * weights).mean()
            loss_t2i = (loss_t2i * weights).mean()
        else:
            loss_i2t = loss_i2t.mean()
            loss_t2i = loss_t2i.mean()

        loss = (loss_i2t + loss_t2i) / 2
        return loss


class FocalCLIPLoss(nn.Module):
    """
    Focal CLIP Loss - 关注难样本

    原理：
    - 难样本（低置信度）的 loss 贡献更大
    - 易样本（高置信度）的 loss 贡献被降低
    - Focal weight: (1 - pt)^gamma，其中 pt 是预测概率
    """

    def __init__(self, alpha=None, gamma=2.0, temperature=0.07):
        """
        Args:
            alpha: (num_classes,) tensor，类别权重
            gamma: focusing 参数，越大越关注难样本（默认 2.0）
            temperature: CLIP温度参数
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.gamma = gamma

        print(f"\n✓ Initialized Focal CLIP Loss")
        print(f"  Temperature: {temperature}")
        print(f"  Gamma (focusing): {gamma}")
        if alpha is not None:
            print(f"  Alpha (class weights): {alpha.cpu().numpy()}")

    def forward(self, image_features, text_features, labels):
        """
        Args:
            image_features: (batch_size, feature_dim)
            text_features: (batch_size, feature_dim)
            labels: (batch_size,) 类别标签

        Returns:
            loss: focal weighted loss
        """
        # 计算相似度矩阵
        logits = (image_features @ text_features.T) / self.temperature
        batch_labels = torch.arange(len(image_features), device=image_features.device)

        # 计算概率（用于 focal weighting）
        probs_i2t = F.softmax(logits, dim=1)
        probs_t2i = F.softmax(logits.T, dim=1)

        # 获取正确匹配的概率（对角线元素）
        pt_i2t = probs_i2t[torch.arange(len(batch_labels)), batch_labels]
        pt_t2i = probs_t2i[torch.arange(len(batch_labels)), batch_labels]

        # Focal weighting: (1-pt)^gamma
        # 置信度低的样本 -> focal_weight 大 -> loss 贡献大
        focal_weight_i2t = (1 - pt_i2t) ** self.gamma
        focal_weight_t2i = (1 - pt_t2i) ** self.gamma

        # 计算基础交叉熵损失
        loss_i2t = F.cross_entropy(logits, batch_labels, reduction='none')
        loss_t2i = F.cross_entropy(logits.T, batch_labels, reduction='none')

        # 应用 focal weighting
        loss_i2t = loss_i2t * focal_weight_i2t
        loss_t2i = loss_t2i * focal_weight_t2i

        # 应用类别权重（如果提供）
        if self.alpha is not None:
            alpha_weights = self.alpha[labels]
            loss_i2t = loss_i2t * alpha_weights
            loss_t2i = loss_t2i * alpha_weights

        loss = (loss_i2t.mean() + loss_t2i.mean()) / 2
        return loss


class CLIPFineTune(nn.Module):
    """Fully fine-tuned CLIP model (using CLIP loss)"""

    def __init__(self, clip_model, num_classes):
        super().__init__()
        self.clip_model = clip_model

        # Get feature dimension
        device = next(clip_model.parameters()).device
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224).to(device)
            dummy_features = clip_model.encode_image(dummy_input)
            self.feature_dim = dummy_features.shape[-1]

        print(f"Feature dimension: {self.feature_dim}")

        # Learnable class text embeddings
        self.text_embeddings = nn.Parameter(torch.randn(num_classes, self.feature_dim))
        nn.init.normal_(self.text_embeddings, std=0.02)

    def forward(self, images, texts=None):
        # Extract image features (fine-tuned)
        image_features = self.clip_model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # If texts are provided (e.g., radiology reports), encode them
        if texts is not None:
            text_features = self.clip_model.encode_text(texts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        else:
            # Otherwise, use learnable class text embeddings (original behavior)
            text_features = self.text_embeddings
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return image_features, text_features


class BiomedCLIPLoader:
    """BiomedCLIP model loader"""

    def __init__(self, config):
        """
        Args:
            config: Configuration object
        """
        self.config = config
        self.local_checkpoints_dir = config.paths.local_checkpoints_dir
        self.model_name = config.model.model_name
        self.context_length = config.model.context_length

    def load_model(self):
        """
        Load BiomedCLIP model from local checkpoint

        Returns:
            model: CLIP model
            tokenizer: Text tokenizer
            preprocess: Image preprocessing function
        """
        print("=" * 80)
        print("Loading BiomedCLIP model...")
        print("=" * 80)

        # Read configuration file
        try:
            with open(f"{self.local_checkpoints_dir}/open_clip_config.json", "r") as f:
                config = json.load(f)
                model_cfg = config["model_cfg"]
                preprocess_cfg = config["preprocess_cfg"]
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Cannot find config file '{self.local_checkpoints_dir}/open_clip_config.json'"
            )

        # Register configuration
        if (not self.model_name.startswith(HF_HUB_PREFIX)
            and self.model_name not in _MODEL_CONFIGS
            and config is not None):
            _MODEL_CONFIGS[self.model_name] = model_cfg

        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = get_tokenizer(self.model_name)

        # Create model
        print("Creating model...")
        model, _, preprocess = create_model_and_transforms(
            model_name=self.model_name,
            pretrained=f"{self.local_checkpoints_dir}/open_clip_pytorch_model.bin",
            **{f"image_{k}": v for k, v in preprocess_cfg.items()},
        )

        print("Model loaded successfully!")
        return model, tokenizer, preprocess

    def create_tokenizer_wrapper(self, tokenizer):
        """
        Create a tokenizer wrapper with context_length

        Args:
            tokenizer: Original tokenizer

        Returns:
            Wrapped tokenizer function
        """
        def tokenize(texts):
            return tokenizer(texts, context_length=self.context_length)

        return tokenize


# ==============================================================================
# Helper Functions for Computing Class Weights
# ==============================================================================

def compute_class_weights(labels, method='inverse', device='cpu'):
    """
    计算类别权重（用于处理类别不平衡）

    Args:
        labels: list或array，训练集的所有标签
        method: 权重计算方法
            - 'inverse': 1 / count (反比例)
            - 'sqrt_inverse': 1 / sqrt(count) (平方根反比例，更温和)
            - 'effective': 基于Effective Number的权重
        device: torch设备

    Returns:
        class_weights: torch.Tensor (num_classes,)
    """
    import numpy as np
    from collections import Counter

    # 统计每个类别的样本数
    label_counts = Counter(labels)
    num_classes = len(label_counts)
    counts = np.array([label_counts[i] for i in range(num_classes)])

    if method == 'inverse':
        # 反比例权重: weight = 1 / count
        weights = 1.0 / counts
    elif method == 'sqrt_inverse':
        # 平方根反比例权重: weight = 1 / sqrt(count)
        weights = 1.0 / np.sqrt(counts)
    elif method == 'effective':
        # Effective Number of Samples
        # 论文: https://arxiv.org/abs/1901.05555
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / effective_num
    else:
        raise ValueError(f"Unknown method: {method}")

    # 归一化：让所有权重的平均值 = 1
    weights = weights / weights.mean()

    # 转换为 torch tensor
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    print(f"\n✓ Computed class weights ({method} method):")
    print(f"  Sample counts: {counts}")
    print(f"  Class weights: {class_weights.cpu().numpy()}")
    print(f"  Min weight: {class_weights.min():.4f}, Max weight: {class_weights.max():.4f}")

    return class_weights
