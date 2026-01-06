"""
CLIP Inference Module

Unified inference logic for all CLIP-based models (zero-shot, fine-tuned, LoRA, etc.)
"""

import torch
from tqdm import tqdm


def clip_inference(clip_model, test_loader, class_names, tokenizer, config, threshold=0.0, device=None,
                   text_prompts=None, prompt_mode='single', task_type='multi-label'):
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
        text_prompts: Optional pre-generated prompts (from generate_text_prompts)
                     - For 'binary' mode: dict with {'positive': [...], 'negative': [...]}
                     - For 'single' mode: list of prompts [...]
                     - If None, uses legacy format: "There is {class}."
        prompt_mode: 'binary' or 'single' (default: 'single')
        task_type: 'multi-label' or 'single-label' (default: 'multi-label')
                  - 'multi-label': Use threshold for prediction (can predict multiple classes)
                  - 'single-label': Use argmax for prediction (predict only one class)

    Returns:
        predictions: Array of predictions
                    - For multi-label: (N, num_classes) binary matrix
                    - For single-label: (N,) class indices
        labels: Array of true labels (N, num_classes)
        scores: Array of raw similarity scores (N, num_classes) for AUC calculation

    Note:
        For binary mode:
        - Computes similarity with both positive and negative prompts
        - Final score = positive_similarity - negative_similarity
        - Prediction = (final_score > threshold) for multi-label, argmax for single-label
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    clip_model.to(device)
    clip_model.eval()

    # Generate text prompts if not provided (legacy behavior)
    if text_prompts is None:
        text_prompts = [f"There is {cls.lower().replace('_', ' ')}." for cls in class_names]
        prompt_mode = 'single'

    print(f"\n[Inference] Using {len(class_names)} classes")
    print(f"Prompt mode: {prompt_mode}")
    print(f"Threshold: {threshold}")

    if prompt_mode == 'binary':
        # Binary mode: use both positive and negative prompts
        positive_prompts = text_prompts['positive']
        negative_prompts = text_prompts['negative']

        # Encode positive prompts
        positive_texts = tokenizer(positive_prompts, context_length=config.model.context_length).to(device)
        # Encode negative prompts
        negative_texts = tokenizer(negative_prompts, context_length=config.model.context_length).to(device)

        with torch.no_grad():
            # Get text features for positive and negative prompts
            positive_features = clip_model.encode_text(positive_texts)
            positive_features = positive_features / positive_features.norm(dim=-1, keepdim=True)

            negative_features = clip_model.encode_text(negative_texts)
            negative_features = negative_features / negative_features.norm(dim=-1, keepdim=True)

            all_predictions = []
            all_labels = []
            all_scores = []

            for images, _, labels in tqdm(test_loader, desc="Inference (Binary)"):
                images = images.to(device)

                # Get image features
                image_features = clip_model.encode_image(images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                # Compute similarity with positive and negative prompts
                positive_similarity = image_features @ positive_features.T  # (batch, num_classes)
                negative_similarity = image_features @ negative_features.T  # (batch, num_classes)

                # Final score: difference between positive and negative
                similarity = positive_similarity - negative_similarity

                # Prediction based on task_type
                if task_type == 'single-label':
                    # Single-label: use argmax to select one class
                    # Return 1D indices (N,) for compatibility with single-label evaluators
                    predictions = torch.argmax(similarity, dim=1)  # (batch,)
                else:
                    # Multi-label: use threshold
                    # Return 2D binary matrix (N, num_classes)
                    predictions = (similarity > threshold).float()

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_scores.extend(similarity.cpu().numpy())

    else:
        # Single mode: use one prompt per class (legacy behavior)
        texts = tokenizer(text_prompts, context_length=config.model.context_length).to(device)

        with torch.no_grad():
            # Get text features for all classes
            text_features = clip_model.encode_text(texts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            all_predictions = []
            all_labels = []
            all_scores = []

            for images, _, labels in tqdm(test_loader, desc="Inference (Single)"):
                images = images.to(device)

                # Get image features
                image_features = clip_model.encode_image(images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                # Compute similarity with all class texts
                similarity = image_features @ text_features.T  # (batch, num_classes)

                # Prediction based on task_type
                if task_type == 'single-label':
                    # Single-label: use argmax to select one class
                    # Return 1D indices (N,) for compatibility with single-label evaluators
                    predictions = torch.argmax(similarity, dim=1)  # (batch,)
                else:
                    # Multi-label: use threshold
                    # Return 2D binary matrix (N, num_classes)
                    predictions = (similarity > threshold).float()

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_scores.extend(similarity.cpu().numpy())

    return all_predictions, all_labels, all_scores
