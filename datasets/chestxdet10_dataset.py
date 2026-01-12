"""
ChestXDet10 Dataset Module

This module handles loading the ChestX-Det10 dataset for external testing.
Used for zero-shot evaluation of models trained on MIMIC-CXR.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class ChestXDet10Dataset(Dataset):
    """ChestXDet10 Dataset for multi-label classification"""

    def __init__(self, image_paths, labels, transform=None, tokenizer=None, text_prompts=None):
        """
        Args:
            image_paths: List of image file paths
            labels: Array of multi-label annotations (N, num_classes)
            transform: Image transformations
            tokenizer: Text tokenizer (unused; tokenization handled in inference)
            text_prompts: List of text prompts for each class (unused)
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Get labels
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        # Return empty tensor instead of None for text field (for collate compatibility)
        return image, torch.tensor([]), label


class ChestXDet10DataLoader:
    """ChestXDet10 data loader for external testing"""

    def __init__(self, config, data_path=None):
        """
        Args:
            config: Configuration object
            data_path: Path to ChestXDet10 dataset (overrides default)
        """
        self.config = config
        self.base_path = data_path if data_path else r"D:\Data\ChestXDet10"
        self.annotation_dir = self._resolve_annotation_dir(self.base_path)
        self.test_image_dir = self._resolve_test_image_dir(self.base_path)

        # Load class names from JSON configuration
        from utils import load_class_names
        self.class_names = load_class_names('chestxdet10')
        self.num_classes = len(self.class_names)

    def _resolve_annotation_dir(self, base_path):
        """Resolve the annotation directory containing train.json/test.json"""
        if os.path.exists(os.path.join(base_path, 'train.json')) and os.path.exists(os.path.join(base_path, 'test.json')):
            return base_path

        candidate = os.path.join(base_path, 'ChestX-Det10-Dataset-master')
        if os.path.exists(os.path.join(candidate, 'train.json')) and os.path.exists(os.path.join(candidate, 'test.json')):
            return candidate

        return base_path

    def _resolve_test_image_dir(self, base_path):
        """Resolve the test image directory"""
        if os.path.basename(base_path).lower() == 'test_data' and os.path.isdir(base_path):
            return base_path

        candidate = os.path.join(base_path, 'test_data')
        if os.path.isdir(candidate):
            return candidate

        fallback = os.path.join(base_path, 'images')
        if os.path.isdir(fallback):
            return fallback

        return base_path

    def load_test_data(self):
        """
        Load ChestXDet10 test set

        Returns:
            image_paths: List of image paths
            labels: Array of multi-label annotations
        """
        print("\n" + "=" * 80)
        print("Loading ChestXDet10 Test Dataset")
        print("=" * 80)

        annotation_path = os.path.join(self.annotation_dir, 'test.json')
        print(f"Loading annotations from: {annotation_path}")

        with open(annotation_path, 'r', encoding='utf-8') as f:
            annotations = json.load(f)

        print(f"\nTest images loaded: {len(annotations)}")

        # Build image paths
        image_paths = []
        for entry in annotations:
            img_path = os.path.join(self.test_image_dir, entry['file_name'])
            image_paths.append(img_path)

        # Process labels
        labels = self._process_labels(annotations)

        # Print label statistics
        self._print_label_statistics(labels)

        return image_paths, labels

    def _process_labels(self, annotations):
        """
        Process annotation labels to multi-label format

        Args:
            annotations: List of annotation dicts with "syms"

        Returns:
            labels: Array of shape (N, num_classes) with binary labels
        """
        labels = np.zeros((len(annotations), self.num_classes), dtype=np.float32)
        class_to_idx = {name: i for i, name in enumerate(self.class_names)}

        for i, entry in enumerate(annotations):
            for sym in entry.get('syms', []):
                class_idx = class_to_idx.get(sym)
                if class_idx is not None:
                    labels[i, class_idx] = 1.0

        return labels

    def create_dataloader(self, image_paths, labels, preprocess, tokenizer, text_prompts):
        """
        Create PyTorch DataLoader

        Args:
            image_paths: List of image paths
            labels: Label array
            preprocess: Image preprocessing transform
            tokenizer: Text tokenizer
            text_prompts: List of text prompts

        Returns:
            DataLoader
        """
        dataset = ChestXDet10Dataset(
            image_paths=image_paths,
            labels=labels,
            transform=preprocess,
            tokenizer=tokenizer,
            text_prompts=text_prompts
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.data.batch_size,
            shuffle=False,  # Don't shuffle test set
            num_workers=self.config.data.num_workers,
            pin_memory=True
        )
        print(f"\nCreated ChestXDet10 test dataloader: {len(dataset)} samples, "
              f"{len(dataloader)} batches")

        return dataloader

    def _print_label_statistics(self, labels):
        """Print label statistics"""
        print("\n" + "-" * 80)
        print("ChestXDet10 Test Set Label Statistics")
        print("-" * 80)

        total_samples = len(labels)
        positive_counts = labels.sum(axis=0)
        positive_ratios = positive_counts / total_samples * 100

        print(f"{'Class':<30} {'Positive':<10} {'Ratio':<10}")
        print("-" * 80)
        for i, class_name in enumerate(self.class_names):
            print(f"{class_name:<30} {int(positive_counts[i]):<10} {positive_ratios[i]:>6.2f}%")

        # Labels per image statistics
        labels_per_image = labels.sum(axis=1)
        print("\n" + "-" * 80)
        print("Labels per Image Statistics")
        print("-" * 80)
        print(f"  Mean:   {labels_per_image.mean():.2f}")
        print(f"  Median: {np.median(labels_per_image):.2f}")
        print(f"  Min:    {labels_per_image.min():.0f}")
        print(f"  Max:    {labels_per_image.max():.0f}")


if __name__ == "__main__":
    # Test the data loader
    from config import Config

    config = Config()

    loader = ChestXDet10DataLoader(config)
    image_paths, labels = loader.load_test_data()

    print(f"\nLoaded {len(image_paths)} test images")
    print(f"Label shape: {labels.shape}")

    # Check if images exist
    print("\nChecking first 3 image paths:")
    for i in range(min(3, len(image_paths))):
        exists = "OK" if os.path.exists(image_paths[i]) else "MISSING"
        print(f"  [{exists}] {image_paths[i]}")

    print("\nChestXDet10 test set loaded successfully!")
