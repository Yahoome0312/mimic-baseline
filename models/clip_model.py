"""
CLIP Model Module
"""

import json
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
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
            labels: (optional) unused

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


class CLIPFineTune(nn.Module):
    """
    Fully fine-tuned CLIP model (using CLIP loss)

    This model fine-tunes both image and text encoders using radiology reports.
    No fixed text embeddings - uses dynamic text encoding for flexibility.
    """

    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model

    def forward(self, images, texts):
        """
        Forward pass with dynamic text encoding

        Args:
            images: (batch, 3, 224, 224) - Input images
            texts: Tokenized text (training: reports, inference: prompts)

        Returns:
            image_features: Normalized image features (batch, 512)
            text_features: Normalized text features (batch, 512)
        """
        if texts is None:
            raise ValueError(
                "texts is required for MIMIC training with radiology reports. "
                "This model does not support fixed text_embeddings."
            )

        # Extract image features (fine-tuned)
        image_features = self.clip_model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Extract text features (dynamic encoding, any number of classes)
        text_features = self.clip_model.encode_text(texts)
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
        # When using 'local-dir:' schema, model weights are loaded automatically from that directory
        # No need to specify 'pretrained' parameter explicitly
        if self.model_name.startswith('local-dir:'):
            model, _, preprocess = create_model_and_transforms(
                model_name=self.model_name,
                **{f"image_{k}": v for k, v in preprocess_cfg.items()},
            )
        else:
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


class MixClsHead(nn.Module):
    """Lightweight classification head for bag-of-words prediction."""
    # 轻量级分类头，MLP 残差结构，用于将视觉特征映射到词汇表维度，得到logits（尺寸V*1），V为词汇表大小
    def __init__(
        self,
        width: int,
        layers: int,
        mlp_ratio: float = 4.0, # MLP扩展比例
        output_dim: int = 512,
    ):
        super().__init__()
        self.width = width
        mlp_width = int(width * mlp_ratio) # MLP隐藏层宽度

        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(width),
                nn.Linear(width, mlp_width),
                nn.GELU(),
                nn.Linear(mlp_width, width),
            )
            for _ in range(layers)
        ])

        self.ln_mlp = nn.LayerNorm(width)
        self.text_projection = nn.Linear(width, output_dim) # 将 width 维特征映射到output_dim维，即投影到文本特征空间

        self._init_parameters()

    def _init_parameters(self):
        proj_std = (self.width ** -0.5) * (2 ** -0.5) # 标准差计算
        fc_std = (2 * self.width) ** -0.5 # 标准差计算
         # 初始化MLP层的权重和偏置
        for block in self.mlps:
            nn.init.normal_(block[1].weight, std=fc_std)
            nn.init.normal_(block[3].weight, std=proj_std)
        nn.init.normal_(self.text_projection.weight, std=self.width ** -0.5)
        nn.init.zeros_(self.text_projection.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for mlp in self.mlps:
            x = x + mlp(x) # 残差连接
        x = self.ln_mlp(x)
        return self.text_projection(x)


class SuperCLIPLoss(nn.Module):
    """SuperCLIP loss = cls bag-of-words loss + CLIP contrastive loss."""
    # SuperCLIP损失函数，结合了分类的bag-of-words损失和CLIP对比损失
    def __init__(
        self,
        temperature: float = 0.07,
        cls_loss_weight: float = 1.0,
        clip_loss_weight: float = 1.0,
        pad_id: Optional[int] = None,
        cls_id: Optional[int] = None,
        sep_id: Optional[int] = None,
        unk_id: Optional[int] = None,
        mask_id: Optional[int] = None,
    ):
        super().__init__()
        self.temperature = temperature
        self.cls_loss_weight = cls_loss_weight
        self.clip_loss_weight = clip_loss_weight
        self.pad_id = pad_id
        self.cls_id = cls_id
        self.sep_id = sep_id
        self.unk_id = unk_id
        self.mask_id = mask_id

    # 重新加权目标函数，基于类别频率调整目标分布
    def _reweight_targets(self, cap_fq: torch.Tensor, num_samples: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        cap_fq += targets.sum(dim=0, keepdim=True) / targets.shape[0]
        num_samples += 1

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(cap_fq, op=dist.ReduceOp.AVG)
            dist.all_reduce(num_samples, op=dist.ReduceOp.AVG)
            world_size = dist.get_world_size()
        else:
            world_size = 1

        all_batch_size = world_size * targets.shape[0]
        targets = targets * torch.log(
            (num_samples + 1.0 / all_batch_size) / (cap_fq + 1.0 / all_batch_size)
        ).to(dtype=targets.dtype)
        return targets

    def _class_loss(
        self,
        cap_fq: torch.Tensor,
        num_samples: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor: # 计算分类损失，使用重新加权的目标分布
        batch_size, vocab_size = logits.shape
        targets = torch.zeros(batch_size, vocab_size, dtype=torch.float32, device=logits.device)
        targets.scatter_(dim=1, index=labels.long(), value=1.0)
        for token_id in (self.pad_id, self.cls_id, self.sep_id, self.unk_id, self.mask_id):
            if token_id is None:
                continue
            if 0 <= token_id < vocab_size:
                targets[:, token_id] = 0
        targets = self._reweight_targets(cap_fq, num_samples, targets)
        row_sums = targets.sum(dim=1, keepdim=True)
        targets = targets / row_sums.clamp_min(1.0)
        return -(F.log_softmax(logits, dim=1) * targets).sum(dim=1).mean()

    def _clip_loss(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        logits = (image_features @ text_features.T) / self.temperature
        batch_labels = torch.arange(len(image_features), device=image_features.device)
        loss_i2t = F.cross_entropy(logits, batch_labels)
        loss_t2i = F.cross_entropy(logits.T, batch_labels)
        return (loss_i2t + loss_t2i) / 2

    def forward(
        self,
        cap_fq: torch.Tensor,
        num_samples: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        output_dict: bool = False,
        **unused,
    ):
        class_loss = self._class_loss(cap_fq, num_samples, logits, labels) * self.cls_loss_weight
        clip_loss = self._clip_loss(image_features, text_features) * self.clip_loss_weight
        if output_dict:
            return {"class_loss": class_loss, "contrastive_loss": clip_loss}
        return class_loss + clip_loss


def _get_embed_dim(model: nn.Module) -> Optional[int]:
    proj = getattr(model, "text_projection", None)
    if isinstance(proj, nn.Linear):
        return proj.out_features
    if isinstance(proj, torch.Tensor):
        return proj.shape[1]
    return None


def _get_visual_dim(model: nn.Module) -> Optional[int]:
    visual = getattr(model, "visual", None)
    if visual is None:
        return None
    trunk = getattr(visual, "trunk", None)
    if trunk is not None and hasattr(trunk, "num_features"):
        return trunk.num_features
    if hasattr(visual, "width"):
        return visual.width
    if hasattr(visual, "num_features"):
        return visual.num_features
    return None


class SuperCLIPFineTune(nn.Module):
    """
    SuperCLIP-style fine-tuning with an extra image->vocab classifier head.
    """

    def __init__(
        self,
        clip_model: nn.Module,
        vocab_size: int,
        cls_head_layers: int = 1,
        cls_head_mlp_ratio: float = 4.0,
        use_patch_tokens: bool = True,
    ):
        super().__init__()
        self.clip_model = clip_model
        self.use_patch_tokens = use_patch_tokens
        self._warned_no_tokens = False
    

        cls_feature_dim = _get_visual_dim(clip_model)
        if cls_feature_dim is None:
            raise ValueError("Unable to infer visual feature dim for SuperCLIP head.")
        self.cls_feature_dim = cls_feature_dim

        embed_dim = _get_embed_dim(clip_model)
        if embed_dim is not None and embed_dim != cls_feature_dim:
            self.cls_input_proj = nn.Linear(embed_dim, cls_feature_dim)
        else:
            self.cls_input_proj = None

        self.text_decoder = MixClsHead(
            width=cls_feature_dim,
            layers=cls_head_layers,
            mlp_ratio=cls_head_mlp_ratio,
            output_dim=vocab_size,
        )

        self.register_buffer("cap_fq", torch.zeros([1, vocab_size], dtype=torch.float64))# 
        self.register_buffer("num_samples", torch.zeros([1, 1], dtype=torch.float64))

    def _get_patch_tokens(self, images: torch.Tensor):
        if not self.use_patch_tokens:
            return None, None

        if hasattr(self.clip_model, "forward_intermediates"):
            try:
                out = self.clip_model.forward_intermediates(
                    image=images,
                    image_indices=1,
                    image_output_fmt="NLC",
                    image_output_extra_tokens=False,
                    normalize_intermediates=True,
                )
                tokens = out.get("image_intermediates")
                if isinstance(tokens, list) and tokens:
                    tokens = tokens[-1]
                image_features = out.get("image_features")
                return tokens, image_features
            except Exception:
                return None, None

        return None, None

    def forward(self, images: torch.Tensor, texts: torch.Tensor):
        tokens, image_features = self._get_patch_tokens(images)
        if image_features is None:
            image_features = self.clip_model.encode_image(images)

        if tokens is not None:
            if tokens.dim() == 4:
                tokens = tokens.flatten(2).transpose(1, 2)
            class_features = tokens.mean(dim=1)
        else:
            if self.use_patch_tokens:
                raise RuntimeError(
                    "Patch tokens unavailable for SuperCLIP cls head. "
                    "Set cls_use_patch_tokens=False or use a vision tower that exposes patch tokens."
                )
            class_features = image_features

        if self.cls_input_proj is not None and class_features.shape[-1] != self.cls_feature_dim:
            class_features = self.cls_input_proj(class_features)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = self.clip_model.encode_text(texts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logits = self.text_decoder(class_features)

        return {
            "cap_fq": self.cap_fq,
            "num_samples": self.num_samples,
            "logits": logits,
            "labels": texts,
            "image_features": image_features,
            "text_features": text_features,
        }
