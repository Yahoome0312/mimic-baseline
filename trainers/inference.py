"""
CLIP Inference Module

Unified inference logic for all CLIP-based models (zero-shot, fine-tuned, LoRA, etc.)
"""

import torch
from tqdm import tqdm


def clip_inference(clip_model, test_loader, class_names, tokenizer, config, threshold=0.0, device=None):
    """
    Universal inference function for CLIP models

    This function ensures all training methods use identical inference logic.
    Works with zero-shot, fine-tuned, LoRA, Adapter, and any future methods.

    Args:
        clip_model: CLIP model (can be original or fine-tuned)
        test_loader: Test data loader
        class_names: List of class names for generating text prompts
        tokenizer: Text tokenizer
        config: Configuration object
        threshold: Similarity threshold for multi-label classification (default: 0.0)
        device: Device to run on (default: auto-detect)

    Returns:
        predictions: Array of predictions (N, num_classes)
        labels: Array of true labels (N, num_classes)
        scores: Array of raw similarity scores (N, num_classes) for AUC calculation
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    clip_model.to(device)
    clip_model.eval()

    # Generate text prompts from class names
    text_prompts = [f"There is {cls.lower().replace('_', ' ')}." for cls in class_names]

    print(f"\n[Inference] Using {len(class_names)} classes")
    print(f"Threshold: {threshold}")

    # Encode class text features
    texts = tokenizer(text_prompts, context_length=config.model.context_length).to(device)

    with torch.no_grad():
        # Get text features for all classes
        text_features = clip_model.encode_text(texts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        all_predictions = []
        all_labels = []
        all_scores = []

        for images, _, labels in tqdm(test_loader, desc="Inference"):
            images = images.to(device)

            # Get image features
            image_features = clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Compute similarity with all class texts
            similarity = image_features @ text_features.T  # (batch, num_classes)

            # Multi-label prediction using threshold
            predictions = (similarity > threshold).float()

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(similarity.cpu().numpy())

    return all_predictions, all_labels, all_scores
