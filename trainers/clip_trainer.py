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

from ..models import CLIPLoss, WeightedCLIPLoss, FocalCLIPLoss, CLIPFineTune, compute_class_weights
from ..utils import EarlyStopping


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

    def predict(self, test_loader):
        """
        Perform zero-shot prediction

        Args:
            test_loader: Test data loader

        Returns:
            all_predictions: List of predictions (multi-label: N x num_classes)
            all_labels: List of true labels (multi-label: N x num_classes)
            all_scores: List of prediction scores
        """
        print("\n" + "=" * 80)
        print("Method 1: Zero-shot CLIP")
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

        with torch.no_grad():
            for images, _, labels in tqdm(test_loader, desc="Zero-shot inference"):
                images = images.to(self.device)

                # Get features and logits
                image_features, text_features, logit_scale = self.model(images, texts)

                # Calculate similarity
                logits = (logit_scale * image_features @ text_features.t()).detach()

                if is_multilabel:
                    # Multi-label: use sigmoid and threshold at 0.5
                    probs = torch.sigmoid(logits)
                    predictions = (probs > 0.5).float()
                    all_scores.extend(probs.cpu().numpy())
                else:
                    # Single-label: use softmax and argmax
                    probs = logits.softmax(dim=-1)
                    scores, predictions = probs.max(dim=-1)
                    all_scores.extend(scores.cpu().numpy())

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

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
        image_params = []
        text_params = []

        for name, param in self.model.named_parameters():
            if 'text_embeddings' in name:
                text_params.append(param)
            else:
                image_params.append(param)

        self.optimizer = optim.AdamW([
            {'params': image_params, 'lr': self.config.training.learning_rate_image},
            {'params': text_params, 'lr': self.config.training.learning_rate_text}
        ], weight_decay=self.config.training.weight_decay)

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

            # Forward pass
            self.optimizer.zero_grad()
            image_features, text_features = self.model(images)

            if is_multilabel:
                # Multi-label: compute similarity for all classes
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
        """Validate model"""
        self.model.eval()
        total_loss = 0
        is_multilabel = self.config.classes.task_type == 'multi-label'

        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for images, texts, labels in tqdm(val_loader, desc="Validation"):
                images, labels = images.to(self.device), labels.to(self.device)

                image_features, text_features = self.model(images)
                logits = image_features @ text_features.T

                if is_multilabel:
                    # Multi-label classification
                    bce_loss = nn.BCEWithLogitsLoss()(logits, labels)
                    total_loss += bce_loss.item()

                    # Get predictions
                    probs = torch.sigmoid(logits)
                    predicted = (probs > 0.5).float()

                    all_preds.append(predicted.cpu().numpy())
                    all_labels.append(labels.cpu().numpy())
                    all_probs.append(probs.cpu().numpy())
                else:
                    # Single-label classification
                    batch_text_features = text_features[labels]
                    loss = self.criterion(image_features, batch_text_features, labels)
                    total_loss += loss.item()

                    _, predicted = logits.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(val_loader)

        if is_multilabel:
            # Concatenate all predictions and labels
            all_preds = np.vstack(all_preds)
            all_labels = np.vstack(all_labels)
            all_probs = np.vstack(all_probs)

            # Multi-label accuracy
            accuracy = (all_preds == all_labels).mean() * 100

            # Multi-label F1 score (samples average)
            f1_macro = f1_score(all_labels, all_preds, average='samples', zero_division=0)
        else:
            accuracy = 100. * correct / total
            f1_macro = f1_score(all_labels, all_preds, average='macro')

        return avg_loss, accuracy, f1_macro, all_preds, all_labels

    def train(self, train_loader, val_loader):
        """
        Full training loop

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
        """
        print(f"\nStarting training (Full Fine-Tuning with CLIP Loss)...")
        print(f"Epochs: {self.config.training.epochs}")
        print(f"Image Encoder LR: {self.config.training.learning_rate_image}")
        print(f"Text Embeddings LR: {self.config.training.learning_rate_text}")
        print(f"Model saving strategy: Best F1-score (Macro)")

        for epoch in range(self.config.training.epochs):
            print(f"\nEpoch {epoch+1}/{self.config.training.epochs}")

            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)

            # Validate
            val_loss, val_acc, val_f1, _, _ = self.validate(val_loader)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            self.val_f1s.append(val_f1)

            # Update learning rate
            if self.scheduler:
                self.scheduler.step()

            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val F1: {val_f1:.4f}")
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")

            # Early stopping check (based on F1 score)
            is_best = self.early_stopping(val_f1, epoch+1)
            if is_best:
                self.best_val_f1 = val_f1
                self.best_model_state = self.model.state_dict().copy()
                print(f"✓ Best model updated! (F1: {val_f1:.4f})")

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
            print(f"Model selected based on: Best F1-score (Macro) = {self.best_val_f1:.4f}")
            return model_path
        else:
            print("No best model state found!")
            return None

    def load_best_model(self):
        """Load best model state"""
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"Loaded best model (Best Val F1: {self.best_val_f1:.4f})")
        else:
            print("No best model state found!")

    def predict(self, test_loader):
        """
        Predict on test set

        Args:
            test_loader: Test data loader

        Returns:
            test_preds: Predictions
            test_labels: True labels
        """
        self.load_best_model()
        _, _, _, test_preds, test_labels = self.validate(test_loader)
        return test_preds, test_labels
