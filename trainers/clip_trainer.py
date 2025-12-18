"""
CLIP Trainer Module
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from models import CLIPLoss, WeightedCLIPLoss, FocalCLIPLoss, CLIPFineTune, compute_class_weights
from utils import EarlyStopping


class ZeroShotCLIPInference:
    """Zero-shot CLIP inference"""

    def __init__(self, model, tokenizer, config):
        """
        Args:
            model: CLIP model
            tokenizer: Text tokenizer
            config: Configuration object
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def predict(self, test_loader, threshold=0.0):
        """
        Perform zero-shot prediction

        Args:
            test_loader: Test data loader
            threshold: Z-Score threshold for multi-label classification
                      (0=mean, >0=conservative, <0=aggressive, default: 0.0)

        Returns:
            all_predictions: List of predictions (multi-label: N x num_classes)
            all_labels: List of true labels (multi-label: N x num_classes)
            all_scores: List of prediction scores
        """
        print("\n" + "=" * 80)
        print("Zero-shot CLIP")
        print("=" * 80)

        self.model.to(self.device)
        self.model.eval()

        # Prepare text prompts
        text_prompts = self.config.classes.get_text_prompts()
        print("\nText prompts used:")
        for i, (cls, prompt) in enumerate(zip(self.config.classes.class_names, text_prompts)):
            print(f"  {cls}: {prompt}")

        # Encode text
        print("\nEncoding text features...")
        texts = self.tokenizer(text_prompts, context_length=self.config.model.context_length).to(self.device)

        # Zero-shot prediction
        all_predictions = []
        all_labels = []
        all_scores = []

        print("\nPerforming Zero-shot classification...")
        is_multilabel = self.config.classes.task_type == 'multi-label'

        # Track similarity statistics for debugging
        all_similarities = []

        with torch.no_grad():
            # Encode text features once (shared across all images)
            text_features = self.model.encode_text(texts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            for images, _, labels in tqdm(test_loader, desc="Zero-shot inference"):
                images = images.to(self.device)

                # Encode image features
                image_features = self.model.encode_image(images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                # Calculate similarity (cosine similarity, normalized features)
                similarity = (image_features @ text_features.t()).detach()  # Range: [-1, 1]

                # Collect statistics (first batch only to save memory)
                if len(all_similarities) == 0:
                    all_similarities.append(similarity.cpu().numpy())

                if is_multilabel:
                    # Multi-label: use Z-Score normalized similarity scores
                    # Z-Score normalization: (x - mean) / std
                    # After Z-Score: mean=0, std=1, range≈[-3, 3]
                    # Threshold interpretation:
                    #   - threshold=0: use mean as cutoff (balanced)
                    #   - threshold>0: more conservative (higher precision)
                    #   - threshold<0: more aggressive (higher recall)

                    # Apply Z-Score normalization
                    mean = similarity.mean()
                    std = similarity.std() + 1e-8  # Add epsilon to avoid division by zero
                    z_scores = (similarity - mean) / std

                    predictions = (z_scores > threshold).float()

                    # For AUC calculation, convert Z-scores to probabilities using sigmoid
                    # This maps (-∞, +∞) -> (0, 1)
                    probs = torch.sigmoid(z_scores)
                    all_scores.extend(probs.cpu().numpy())
                else:
                    # Single-label: use softmax and argmax
                    # Use temperature scaling (typical CLIP uses ~100)
                    temperature = 100.0
                    logits = similarity * temperature
                    probs = logits.softmax(dim=-1)
                    scores, predictions = probs.max(dim=-1)
                    all_scores.extend(scores.cpu().numpy())

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Print similarity statistics for debugging
        if len(all_similarities) > 0:
            import numpy as np
            sim_stats = all_similarities[0]
            print("\n" + "=" * 80)
            print("Similarity Score Statistics (first batch):")
            print("=" * 80)
            print(f"Min:    {sim_stats.min():.4f}")
            print(f"Max:    {sim_stats.max():.4f}")
            print(f"Mean:   {sim_stats.mean():.4f}")
            print(f"Median: {np.median(sim_stats):.4f}")
            print(f"Std:    {sim_stats.std():.4f}")
            print(f"Threshold used: {threshold}")
            print("=" * 80)

        return all_predictions, all_labels, all_scores


class CLIPTrainer:
    """CLIP fine-tuning trainer"""

    def __init__(self, model, config, output_dir, train_labels=None, experiment_name=None):
        """
        Args:
            model: CLIPFineTune model
            config: Configuration object
            output_dir: Output directory for saving models and plots
            train_labels: (optional) 训练集标签，用于计算类别权重
            experiment_name: (optional) 实验名称，用于自定义保存文件名
        """
        self.model = model
        self.config = config
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Move model to device
        self.model.to(self.device)

        # Initialize loss function based on config
        self.criterion = self._setup_loss_function(train_labels)

        # Setup optimizer
        self._setup_optimizer()

        # Setup scheduler
        self._setup_scheduler()

        # Setup early stopping
        self.early_stopping = EarlyStopping(
            patience=config.training.early_stopping_patience,
            mode='max',
            verbose=True
        )

        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.val_f1s = []
        self.best_val_f1 = 0
        self.best_model_state = None

    def _setup_loss_function(self, train_labels=None):
        """Setup loss function based on configuration"""
        loss_type = self.config.training.loss_type
        temperature = self.config.model.temperature

        print(f"\n{'='*80}")
        print(f"Setting up loss function: {loss_type}")
        print(f"{'='*80}")

        # Compute class weights if needed
        class_weights = None
        if (loss_type in ['weighted', 'focal']) and train_labels is not None:
            class_weights = compute_class_weights(
                train_labels,
                method=self.config.training.class_weight_method,
                device=self.device
            )

        # Select loss function
        if loss_type == 'standard':
            criterion = CLIPLoss(temperature=temperature)
            print("✓ Using standard CLIP loss")

        elif loss_type == 'weighted':
            criterion = WeightedCLIPLoss(
                class_weights=class_weights,
                temperature=temperature
            )

        elif loss_type == 'focal':
            # Focal loss can optionally use class weights
            alpha = class_weights if self.config.training.focal_alpha else None
            criterion = FocalCLIPLoss(
                alpha=alpha,
                gamma=self.config.training.focal_gamma,
                temperature=temperature
            )

        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        print(f"{'='*80}\n")
        return criterion

    def _setup_optimizer(self):
        """Setup optimizer with layered learning rates"""
        # Get all trainable parameters
        # This includes both image encoder and text encoder in clip_model
        all_params = list(self.model.parameters())

        # Use a single learning rate for simplicity
        # (Both image and text encoders will use the same LR)
        self.optimizer = optim.AdamW(
            all_params,
            lr=self.config.training.learning_rate_image,
            weight_decay=self.config.training.weight_decay
        )

    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        if self.config.training.use_scheduler:
            if self.config.training.scheduler_type == 'cosine':
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=self.config.training.epochs
                )
            elif self.config.training.scheduler_type == 'step':
                self.scheduler = optim.lr_scheduler.StepLR(
                    self.optimizer, step_size=30, gamma=0.1
                )
            else:
                self.scheduler = None
        else:
            self.scheduler = None

    def train_epoch(self, train_loader):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        is_multilabel = self.config.classes.task_type == 'multi-label'

        # For multi-label accuracy tracking
        if is_multilabel:
            all_preds = []
            all_labels = []

        correct = 0
        total = 0

        for images, texts, labels in tqdm(train_loader, desc="Training"):
            images, labels = images.to(self.device), labels.to(self.device)

            # Move texts to device if provided
            if texts is not None:
                texts = texts.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            image_features, text_features = self.model(images, texts)

            if is_multilabel:
                # Multi-label classification
                if texts is not None:
                    # When using reports: Use contrastive loss (image-report pairs)
                    # Create diagonal labels for contrastive learning
                    batch_size = images.size(0)
                    contrastive_labels = torch.arange(batch_size, device=self.device)

                    # Compute similarity matrix
                    logits_per_image = image_features @ text_features.T / 0.07  # temperature
                    logits_per_text = text_features @ image_features.T / 0.07

                    # Contrastive loss
                    loss_i2t = nn.CrossEntropyLoss()(logits_per_image, contrastive_labels)
                    loss_t2i = nn.CrossEntropyLoss()(logits_per_text, contrastive_labels)
                    loss = (loss_i2t + loss_t2i) / 2

                    # For tracking: use image features to predict labels (zero-shot style)
                    with torch.no_grad():
                        # We can't compute class predictions without class embeddings
                        # Just set dummy values for now
                        preds = torch.zeros_like(labels)
                        all_preds.append(preds.cpu())
                        all_labels.append(labels.cpu())
                else:
                    # Original: compute similarity for all classes
                    logits = image_features @ text_features.T  # (batch, num_classes)

                    # Use BCE loss for multi-label
                    bce_loss = nn.BCEWithLogitsLoss()(logits, labels)
                    loss = bce_loss

                    # Track predictions for accuracy
                    with torch.no_grad():
                        preds = (torch.sigmoid(logits) > 0.5).float()
                        all_preds.append(preds.cpu())
                        all_labels.append(labels.cpu())
            else:
                # Single-label: original logic
                batch_text_features = text_features[labels]
                loss = self.criterion(image_features, batch_text_features, labels)

                # Calculate accuracy
                with torch.no_grad():
                    logits = image_features @ text_features.T
                    _, predicted = logits.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        if is_multilabel:
            # Multi-label accuracy: percentage of correctly predicted labels
            all_preds = torch.cat(all_preds, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            accuracy = (all_preds == all_labels).float().mean().item() * 100
        else:
            accuracy = 100. * correct / total

        return avg_loss, accuracy

    def validate(self, val_loader):
        """
        Validate model using contrastive learning metrics
        Uses image-text retrieval Recall@K instead of classification metrics
        """
        self.model.eval()
        total_loss = 0

        # Collect all features for retrieval evaluation
        all_image_features = []
        all_text_features = []

        with torch.no_grad():
            for images, texts, labels in tqdm(val_loader, desc="Validation"):
                images = images.to(self.device)

                # Move texts to device if provided
                if texts is not None:
                    texts = texts.to(self.device)

                # Extract features (using reports, consistent with training)
                image_features, text_features = self.model(images, texts)

                # Compute contrastive loss (same as training)
                batch_size = images.size(0)
                contrastive_labels = torch.arange(batch_size, device=self.device)

                # Similarity matrix
                logits_per_image = image_features @ text_features.T / 0.07
                logits_per_text = text_features @ image_features.T / 0.07

                # Contrastive loss
                loss_i2t = nn.CrossEntropyLoss()(logits_per_image, contrastive_labels)
                loss_t2i = nn.CrossEntropyLoss()(logits_per_text, contrastive_labels)
                loss = (loss_i2t + loss_t2i) / 2

                total_loss += loss.item()

                # Collect features for retrieval evaluation
                all_image_features.append(image_features.cpu())
                all_text_features.append(text_features.cpu())

        avg_loss = total_loss / len(val_loader)

        # Concatenate all features
        all_image_features = torch.cat(all_image_features, dim=0)  # (N, 512)
        all_text_features = torch.cat(all_text_features, dim=0)    # (N, 512)

        # Compute retrieval metrics (Recall@K)
        recall_at_1, recall_at_5, recall_at_10 = self._compute_retrieval_metrics(
            all_image_features, all_text_features
        )

        # Use Recall@5 as the main metric for early stopping
        main_metric = recall_at_5

        return avg_loss, recall_at_1, main_metric, recall_at_10, None, None

    def _compute_retrieval_metrics(self, image_features, text_features, k_values=[1, 5, 10]):
        """
        Compute image-to-text retrieval Recall@K

        Args:
            image_features: (N, D) tensor of image features
            text_features: (N, D) tensor of text features
            k_values: list of k values for Recall@K

        Returns:
            recall@1, recall@5, recall@10
        """
        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Compute similarity matrix: (N, N)
        similarity = image_features @ text_features.T

        N = similarity.size(0)
        recalls = []

        for k in k_values:
            # For each image, find top-k most similar texts
            _, top_k_indices = similarity.topk(k, dim=1)  # (N, k)

            # Check if the correct text (diagonal) is in top-k
            correct_indices = torch.arange(N).unsqueeze(1)  # (N, 1)
            hits = (top_k_indices == correct_indices).any(dim=1).float()

            # Recall@K = percentage of correct retrievals
            recall = hits.mean().item() * 100
            recalls.append(recall)

        return recalls[0], recalls[1], recalls[2]

    def train(self, train_loader, val_loader):
        """
        Full training loop

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
        """
        print(f"\nStarting training (Full Fine-Tuning with CLIP Loss)...")
        print(f"Epochs: {self.config.training.epochs}")
        print(f"Learning Rate: {self.config.training.learning_rate_image}")
        print(f"Validation Metric: Image-to-Text Retrieval Recall@5")
        print(f"Model saving strategy: Best Recall@5")

        for epoch in range(self.config.training.epochs):
            print(f"\nEpoch {epoch+1}/{self.config.training.epochs}")

            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)

            # Validate (returns: loss, R@1, R@5, R@10, None, None)
            val_loss, recall_at_1, recall_at_5, recall_at_10, _, _ = self.validate(val_loader)
            self.val_losses.append(val_loss)
            self.val_accs.append(recall_at_1)  # Store R@1 in acc slot
            self.val_f1s.append(recall_at_5)   # Store R@5 in f1 slot

            # Update learning rate
            if self.scheduler:
                self.scheduler.step()

            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | R@1: {recall_at_1:.2f}% | R@5: {recall_at_5:.2f}% | R@10: {recall_at_10:.2f}%")
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")

            # Early stopping check (based on Recall@5)
            is_best = self.early_stopping(recall_at_5, epoch+1)
            if is_best:
                self.best_val_f1 = recall_at_5  # Store best R@5
                self.best_model_state = self.model.state_dict().copy()
                print(f"✓ Best model updated! (R@5: {recall_at_5:.2f}%)")

            if self.early_stopping.early_stop:
                print(f"\nEarly stopping triggered! Training stopped at epoch {epoch+1}")
                break

        # Plot training curves
        self._plot_training_curves()

    def _plot_training_curves(self):
        """Plot training curves"""
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')

        plt.subplot(1, 2, 2)
        plt.plot(self.train_accs, label='Train Acc')
        plt.plot(self.val_accs, label='Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.title('Training and Validation Accuracy')

        plt.tight_layout()

        # Generate filename based on experiment name
        if self.experiment_name:
            filename = f'{self.experiment_name}_training_curves.png'
        else:
            filename = 'training_curves.png'

        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()

    def save_best_model(self, filename='best_model.pth'):
        """Save best model"""
        if self.best_model_state is not None:
            model_path = os.path.join(self.output_dir, filename)
            torch.save(self.best_model_state, model_path)
            print(f"\nBest model saved to: {model_path}")
            print(f"Model selected based on: Best Recall@5 = {self.best_val_f1:.2f}%")
            return model_path
        else:
            print("No best model state found!")
            return None

    def load_best_model(self):
        """Load best model state"""
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"Loaded best model (Best Val Recall@5: {self.best_val_f1:.2f}%)")
        else:
            print("No best model state found!")

    def predict(self, test_loader):
        """
        Predict on test set
        Note: For contrastive learning, we don't have direct predictions
        This method needs to be called only if doing downstream classification

        Args:
            test_loader: Test data loader

        Returns:
            None, None (placeholder)
        """
        self.load_best_model()
        # For contrastive learning with reports, we don't generate class predictions
        # Return None to indicate this is not applicable
        print("\n[Note] Contrastive learning model - skipping classification predictions")
        return None, None
