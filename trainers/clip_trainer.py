"""
CLIP Trainer Module
"""

import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from models import CLIPLoss, CLIPFineTune, SuperCLIPLoss
from utils import EarlyStopping
from .inference import clip_inference


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

    def predict(self, test_loader, threshold=0.0, class_names=None, text_prompts=None):
        """
        Perform zero-shot prediction

        Args:
            test_loader: Test data loader
            threshold: Cosine similarity threshold for multi-label classification
                      (range: [-1, 1], higher=more conservative, default: 0.0)
            class_names: Optional list of class names (overrides config.classes.class_names)

        Returns:
            all_predictions: List of predictions (multi-label: N x num_classes)
            all_labels: List of true labels (multi-label: N x num_classes)
            all_scores: List of prediction scores
        """
        print("\n" + "=" * 80)
        print("Zero-shot CLIP")
        print("=" * 80)
        print(f"Using cosine similarity threshold: {threshold} (range: [-1, 1], higher=more conservative)")

        # Class names must be provided
        if class_names is None:
            raise ValueError("class_names parameter is required (load from JSON using utils.load_class_names)")

        # Use provided text prompts or generate default
        if text_prompts is None:
            text_prompts = [f"There is {cls.lower().replace('_', ' ')}." for cls in class_names]

        # Use universal inference function
        all_predictions, all_labels, all_scores = clip_inference(
            clip_model=self.model,
            test_loader=test_loader,
            class_names=class_names,
            tokenizer=self.tokenizer,
            config=self.config,
            threshold=threshold,
            device=self.device,
            text_prompts=text_prompts
        )

        # Print similarity statistics for debugging (using first few scores)
        import numpy as np
        all_scores_np = np.array(all_scores)
        if len(all_scores_np) > 0:
            # Sample first batch for statistics
            sample_size = min(len(all_scores_np), 32)
            sample_scores = all_scores_np[:sample_size]

            print("\n" + "=" * 80)
            print("Similarity Score Statistics (first batch):")
            print("=" * 80)
            print(f"Min:    {sample_scores.min():.4f}")
            print(f"Max:    {sample_scores.max():.4f}")
            print(f"Mean:   {sample_scores.mean():.4f}")
            print(f"Median: {np.median(sample_scores):.4f}")
            print(f"Std:    {sample_scores.std():.4f}")
            print(f"Threshold used: {threshold}")
            print("=" * 80)

        return all_predictions, all_labels, all_scores


class CLIPTrainer:
    """CLIP fine-tuning trainer"""

    def __init__(self, model, config, output_dir, experiment_name=None, tokenizer=None):
        """
        Args:
            model: CLIPFineTune model
            config: Configuration object
            output_dir: Output directory for saving models and plots
            experiment_name: (optional) 实验名称，用于自定义保存文件名
            tokenizer: Text tokenizer for report tokenization (optional)
        """
        self.model = model
        self.config = config
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        self.tokenizer = tokenizer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.best_model_path = os.path.join(
            self.output_dir,
            f"{self.experiment_name}_best_model.pth" if self.experiment_name else "best_model.pth"
        )
        self.use_superclip = getattr(config.model, "use_superclip", False)

        # Move model to device
        self.model.to(self.device)

        # All datasets use multi-label classification
        print("Task type: multi-label")

        # Initialize loss function based on config
        self.criterion = self._setup_loss_function()

        # Setup optimizer
        self._setup_optimizer()

        # Setup scheduler
        self._setup_scheduler()

        # Setup early stopping (monitor validation loss)
        self.early_stopping = EarlyStopping(
            patience=config.training.early_stopping_patience,
            mode='min',  # Lower loss is better
            verbose=True
        )

        # Setup AMP (Automatic Mixed Precision) - always enabled for GPU training
        self.scaler = torch.amp.GradScaler('cuda')
        print("✓ AMP (Automatic Mixed Precision) enabled")

        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_accs = []
        self.val_f1s = []
        self.best_val_loss = float('inf')  # Initialize with infinity
        self.best_model_state = None


    def _setup_loss_function(self):
        """Setup loss function based on configuration"""
        temperature = self.config.model.temperature

        print("\n" + "=" * 80)
        if self.use_superclip:
            print("Setting up loss function: superclip (cls + clip)")
        else:
            print("Setting up loss function: standard")
        print("=" * 80)

        if self.use_superclip:
            criterion = SuperCLIPLoss(
                temperature=temperature,
                cls_loss_weight=self.config.model.cls_loss_weight,
            )
            print("[OK] Using SuperCLIP loss")
        else:
            criterion = CLIPLoss(temperature=temperature)
            print("[OK] Using standard CLIP loss")
        print("=" * 80 + "\n")
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
        """Setup learning rate scheduler with optional warmup"""
        if self.config.training.use_scheduler:
            # Check if warmup is enabled
            if self.config.training.use_warmup and self.config.training.warmup_epochs > 0:
                warmup_epochs = self.config.training.warmup_epochs
                total_epochs = self.config.training.epochs

                # Create warmup scheduler (linear warmup)
                warmup_scheduler = optim.lr_scheduler.LinearLR(
                    self.optimizer,
                    start_factor=self.config.training.warmup_start_factor,
                    end_factor=self.config.training.warmup_end_factor,
                    total_iters=warmup_epochs
                )

                # Create main scheduler
                if self.config.training.scheduler_type == 'cosine':
                    main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                        self.optimizer, T_max=total_epochs - warmup_epochs
                    )
                elif self.config.training.scheduler_type == 'step':
                    main_scheduler = optim.lr_scheduler.StepLR(
                        self.optimizer, step_size=30, gamma=0.1
                    )
                else:
                    main_scheduler = optim.lr_scheduler.ConstantLR(
                        self.optimizer, factor=1.0, total_iters=total_epochs - warmup_epochs
                    )

                # Combine warmup + main scheduler
                self.scheduler = optim.lr_scheduler.SequentialLR(
                    self.optimizer,
                    schedulers=[warmup_scheduler, main_scheduler],
                    milestones=[warmup_epochs]
                )
                print(f"✓ Learning rate scheduler: {self.config.training.scheduler_type} with {warmup_epochs} epochs warmup")
            else:
                # No warmup, use standard scheduler
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
                print(f"✓ Learning rate scheduler: {self.config.training.scheduler_type} (no warmup)")
        else:
            self.scheduler = None
            print("✓ No learning rate scheduler")

    def train_epoch(self, train_loader):
        """Train one epoch"""
        self.model.train()
        total_loss = 0

        progress = tqdm(train_loader, desc="Training")
        for step, (images, texts, labels) in enumerate(progress, start=1):
            images = images.to(self.device)
            texts = texts.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass with mixed precision
            with torch.amp.autocast('cuda'):
                model_out = self.model(images, texts)
                if isinstance(model_out, dict):
                    loss_out = self.criterion(**model_out, output_dict=True)
                    loss = sum(loss_out.values())
                else:
                    loss_out = None
                    image_features, text_features = model_out
                    loss = self.criterion(image_features, text_features, labels)

            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            loss_value = loss.item()
            total_loss += loss_value
            avg_loss = total_loss / step

            if loss_out is not None:
                cls_loss_val = loss_out.get("class_loss")
                clip_loss_val = loss_out.get("contrastive_loss")
                progress.set_postfix(
                    loss=f"{avg_loss:.4f}",
                    cls=f"{cls_loss_val.item():.4f}" if cls_loss_val is not None else "n/a",
                    clip=f"{clip_loss_val.item():.4f}" if clip_loss_val is not None else "n/a",
                )
            else:
                progress.set_postfix(loss=f"{avg_loss:.4f}")

        avg_loss = total_loss / len(train_loader)

        return avg_loss

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

        with torch.inference_mode():
            progress = tqdm(val_loader, desc="Validation")
            for step, (images, texts, labels) in enumerate(progress, start=1):
                images = images.to(self.device)
                texts = texts.to(self.device)
                labels = labels.to(self.device)

                # Forward pass with mixed precision
                with torch.amp.autocast('cuda'):
                    model_out = self.model(images, texts)
                    if isinstance(model_out, dict): #兼容两种模型输出格式的：原来的 CLIPFineTune 返回的是 tuple (image_features, text_features)。新的 SuperCLIPFineTune 返回的是 dict，里面包含 logits/labels/cap_fq/num_samples 等给 SuperCLIPLoss 用的字段
                        image_features = model_out["image_features"]
                        text_features = model_out["text_features"]
                        # 验证阶段计算总损失，但用克隆缓冲避免更新词频统计
                        class_loss = self.criterion._class_loss(
                            model_out["cap_fq"].clone(),
                            model_out["num_samples"].clone(),
                            model_out["logits"],
                            model_out["labels"],
                        )
                        clip_loss = self.criterion._clip_loss(image_features, text_features)
                        loss_out = {"class_loss": class_loss, "contrastive_loss": clip_loss}
                        loss = class_loss + clip_loss
                    else:
                        loss_out = None
                        image_features, text_features = model_out
                        loss = self.criterion(image_features, text_features, labels)

                loss_value = loss.item()
                total_loss += loss_value
                avg_loss = total_loss / step

                if loss_out is not None:
                    cls_loss_val = loss_out.get("class_loss")
                    clip_loss_val = loss_out.get("contrastive_loss")
                    progress.set_postfix(
                        loss=f"{avg_loss:.4f}",
                        cls=f"{cls_loss_val.item():.4f}" if cls_loss_val is not None else "n/a",
                        clip=f"{clip_loss_val.item():.4f}" if clip_loss_val is not None else "n/a",
                    )
                else:
                    progress.set_postfix(loss=f"{avg_loss:.4f}")

                # Collect features for retrieval evaluation (keep on GPU for faster computation)
                all_image_features.append(image_features)
                all_text_features.append(text_features)

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
            # Ensure k does not exceed number of samples
            k_actual = min(k, N)
            _, top_k_indices = similarity.topk(k_actual, dim=1)  # (N, k_actual)

            # Check if the correct text (diagonal) is in top-k
            correct_indices = torch.arange(N, device=similarity.device).unsqueeze(1)  # (N, 1)
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
        print(f"Validation Metric: Validation Loss")
        print(f"Model saving strategy: Best Validation Loss (lowest)")

        for epoch in range(self.config.training.epochs):
            print(f"\nEpoch {epoch+1}/{self.config.training.epochs}")

            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)

            # Validate (returns: loss, R@1, R@5, R@10, None, None)
            val_loss, recall_at_1, recall_at_5, recall_at_10, _, _ = self.validate(val_loader)
            self.val_losses.append(val_loss)
            self.val_accs.append(recall_at_1)  # Store R@1 in acc slot
            self.val_f1s.append(recall_at_5)   # Store R@5 in f1 slot

            # Update learning rate
            if self.scheduler:
                self.scheduler.step()

            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f} | R@1: {recall_at_1:.2f}% | R@5: {recall_at_5:.2f}% | R@10: {recall_at_10:.2f}%")
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")

            # Early stopping check (based on validation loss)
            is_best = self.early_stopping(val_loss, epoch+1)
            if is_best:
                self.best_val_loss = val_loss  # Store best val loss
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"[BEST] Model updated! (Val Loss: {val_loss:.4f})")

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
        plt.plot(self.val_accs, label='Val R@1')
        plt.plot(self.val_f1s, label='Val R@5')
        plt.xlabel('Epoch')
        plt.ylabel('Recall (%)')
        plt.legend()
        plt.title('Validation Retrieval Recall')

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
        if os.path.exists(self.best_model_path):
            model_path = os.path.join(self.output_dir, filename)
            if os.path.abspath(self.best_model_path) != os.path.abspath(model_path):
                shutil.copyfile(self.best_model_path, model_path)
            print(f"\nBest model saved to: {model_path}")
            print(f"Model selected based on: Best Validation Loss = {self.best_val_loss:.4f}")
            return model_path

        print("No best model file found!")
        return None

    def load_best_model(self):
        """Load best model state"""
        if os.path.exists(self.best_model_path):
            self.model.load_state_dict(torch.load(self.best_model_path, map_location=self.device))
            print(f"Loaded best model (Best Val Loss: {self.best_val_loss:.4f})")
        else:
            print("No best model file found!")

    def predict(self, test_loader, class_names=None, threshold=0.0, text_prompts=None):
        """
        Predict on test set

        Args:
            test_loader: Test data loader
            class_names: List of class names for generating text prompts
            threshold: Similarity threshold for multi-label classification (default: 0.0)

        Returns:
            predictions: Array of predictions (N, num_classes)
            labels: Array of true labels (N, num_classes)
            scores: Array of raw similarity scores (N, num_classes) for AUC calculation
        """
        self.load_best_model()
        self.model.eval()

        # Class names must be provided
        if class_names is None:
            raise ValueError("class_names parameter is required (load from JSON using utils.load_class_names)")

        print(f"\n[Classification] Using zero-shot style prediction with {len(class_names)} classes")
        print(f"Threshold: {threshold}")

        # Use provided text prompts or generate default
        if text_prompts is None:
            text_prompts = [f"There is {cls.lower().replace('_', ' ')}." for cls in class_names]

        # Reuse tokenizer from initialization if available, otherwise load
        if self.tokenizer is not None:
            tokenizer = self.tokenizer
        else:
            # Load tokenizer only if not already available (e.g., testing without training)
            from models import BiomedCLIPLoader
            model_loader = BiomedCLIPLoader(self.config)
            _, tokenizer, _ = model_loader.load_model()

        # Use universal inference function
        all_predictions, all_labels, all_scores = clip_inference(
            clip_model=self.model.clip_model,  # Use wrapped CLIP model
            test_loader=test_loader,
            class_names=class_names,
            tokenizer=tokenizer,
            config=self.config,
            threshold=threshold,
            device=self.device,
            text_prompts=text_prompts
        )

        return all_predictions, all_labels, all_scores
