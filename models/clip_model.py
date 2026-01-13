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
