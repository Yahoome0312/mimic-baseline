"""
ISIC 2019 Dataset Module
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class ISIC2019Dataset(Dataset):
    """ISIC 2019 skin lesion dataset"""

    def __init__(self, image_paths: List[str], labels: List[int],
                 transform=None, tokenizer=None, text_prompts: Optional[List[str]] = None):
        """
        Args:
            image_paths: List of image file paths
            labels: List of labels (class indices)
            transform: Image transformation function
            tokenizer: Text tokenizer
            text_prompts: List of text prompts for each class
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.tokenizer = tokenizer
        self.text_prompts = text_prompts

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Apply transform
        if self.transform:
            image = self.transform(image)

        # Get corresponding text
        if self.text_prompts is not None:
            text = self.text_prompts[label]
            if self.tokenizer is not None:
                text = self.tokenizer([text])[0]
        else:
            text = label  # If no text, return label

        return image, text, label


class ISIC2019DataLoader:
    """Data loader and preprocessing for ISIC 2019 dataset"""

    def __init__(self, config):
        """
        Args:
            config: Configuration object
        """
        self.config = config
        self.class_names = config.classes.class_names
        self.base_data_path = config.paths.base_data_path
        self.output_dir = config.paths.output_dir

    def load_data(self) -> Tuple[List[str], List[int]]:
        """
        Load ISIC 2019 dataset

        Returns:
            image_paths: List of image file paths
            labels: List of labels (class indices)
        """
        print("\n" + "=" * 80)
        print("Loading ISIC 2019 dataset...")
        print("=" * 80)

        # Read label file (one-hot encoding format)
        labels_path = os.path.join(self.base_data_path, "ISIC_2019_Training_GroundTruth.csv")
        try:
            labels_df = pd.read_csv(labels_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Cannot find label file '{labels_path}'")

        print(f"Label data contains {len(labels_df)} rows")
        print(f"Class columns: {self.class_names}")

        # Image folder
        images_dir = os.path.join(self.base_data_path, "ISIC_2019_Training_Input")
        if not os.path.exists(images_dir):
            raise FileNotFoundError(f"Cannot find image folder '{images_dir}'")

        print(f"Image folder: {images_dir}")

        # Build image paths and labels
        image_paths = []
        labels = []

        print("\nBuilding dataset...")
        for idx, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc="Processing data"):
            image_id = row['image']

            # Find corresponding class (one-hot decoding)
            class_idx = None
            for i, cls in enumerate(self.class_names):
                if row[cls] == 1.0:
                    class_idx = i
                    break

            # Skip UNK class or unlabeled
            if class_idx is None:
                continue

            # Build image path
            img_path = os.path.join(images_dir, f"{image_id}.jpg")

            if os.path.exists(img_path):
                image_paths.append(img_path)
                labels.append(class_idx)

        print(f"\nDataset preparation completed, found {len(image_paths)} valid records")

        # Display class distribution
        self._print_class_distribution(labels)

        # Plot class distribution
        self._plot_class_distribution(labels)

        return image_paths, labels

    def _print_class_distribution(self, labels: List[int]):
        """Print class distribution"""
        print("\nClass distribution:")
        for i, cls in enumerate(self.class_names):
            count = labels.count(i)
            print(f"  {cls}: {count} images ({100*count/len(labels):.2f}%)")

    def _plot_class_distribution(self, labels: List[int]):
        """Plot class distribution bar chart"""
        print("\nGenerating class distribution visualization...")

        class_counts = [labels.count(i) for i in range(len(self.class_names))]

        # Sort by sample count
        class_data = list(zip(self.class_names, class_counts))
        class_data_sorted = sorted(class_data, key=lambda x: x[1], reverse=True)
        sorted_names = [item[0] for item in class_data_sorted]
        sorted_counts = [item[1] for item in class_data_sorted]

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 7))

        # Use different color gradients
        colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(sorted_names)))

        # Bar chart
        bars = ax.bar(range(len(sorted_names)), sorted_counts,
                       color=colors, alpha=0.9, edgecolor='black', linewidth=1.5)

        # Add value labels on bars
        for i, (bar, count, name) in enumerate(zip(bars, sorted_counts, sorted_names)):
            height = bar.get_height()
            # Count label
            ax.text(bar.get_x() + bar.get_width()/2., height + 50,
                    f'{count}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
            # Percentage label
            ax.text(bar.get_x() + bar.get_width()/2., height/2,
                    f'{100*count/len(labels):.1f}%',
                    ha='center', va='center', fontsize=10, fontweight='bold', color='white')

        # Set x-axis labels
        ax.set_xticks(range(len(sorted_names)))
        ax.set_xticklabels(sorted_names, rotation=45, ha='right', fontsize=11)

        # Set labels and title
        ax.set_xlabel('Class (Sorted by Sample Count)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Number of Samples', fontsize=13, fontweight='bold')
        ax.set_title('ISIC 2019 Dataset Class Distribution', fontsize=16, fontweight='bold', pad=20)

        # Add grid
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
        ax.set_axisbelow(True)

        # Set borders
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2)
            spine.set_edgecolor('black')

        # Add total count info box
        total_text = f'Total Samples: {len(labels)}\nNumber of Classes: {len(self.class_names)}'
        ax.text(0.98, 0.97, total_text, transform=ax.transAxes,
                fontsize=11, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black', linewidth=2))

        plt.tight_layout()

        # Save figure
        class_dist_path = os.path.join(self.output_dir, 'class_distribution.png')
        plt.savefig(class_dist_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Class distribution plot saved to: {class_dist_path}")

    def split_dataset(self, image_paths: List[str], labels: List[int]) -> dict:
        """
        Split dataset into train, validation, and test sets

        Args:
            image_paths: List of image file paths
            labels: List of labels

        Returns:
            Dictionary containing train/val/test splits
        """
        print("\n" + "=" * 80)
        print("Splitting dataset...")
        print("=" * 80)

        # First split out test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            image_paths, labels,
            test_size=self.config.data.test_size,
            random_state=self.config.data.random_state,
            stratify=labels
        )

        # Then split validation set from remaining data
        val_size_adjusted = self.config.data.val_size / (1 - self.config.data.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=self.config.data.random_state,
            stratify=y_temp
        )

        print(f"Training set: {len(X_train)} images")
        print(f"Validation set: {len(X_val)} images")
        print(f"Test set: {len(X_test)} images")

        return {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }

    def create_dataloaders(self, data_splits: dict, preprocess, tokenizer=None,
                          text_prompts: Optional[List[str]] = None) -> dict:
        """
        Create DataLoader instances for train, val, and test sets

        Args:
            data_splits: Dictionary containing train/val/test splits
            preprocess: Image preprocessing function
            tokenizer: Text tokenizer
            text_prompts: List of text prompts for each class

        Returns:
            Dictionary containing DataLoader instances
        """
        dataloaders = {}

        for split_name, (image_paths, labels) in data_splits.items():
            dataset = ISIC2019Dataset(
                image_paths, labels,
                transform=preprocess,
                tokenizer=tokenizer,
                text_prompts=text_prompts
            )

            dataloaders[split_name] = DataLoader(
                dataset,
                batch_size=self.config.data.batch_size,
                shuffle=(split_name == 'train'),
                num_workers=self.config.data.num_workers
            )

        return dataloaders
