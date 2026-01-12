"""
CheXpert Dataset Module

This module handles loading the CheXpert dataset for external testing.
Used for zero-shot evaluation of models trained on MIMIC-CXR.
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class CheXpertDataset(Dataset):
    """CheXpert Dataset for multi-label classification"""

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


class CheXpertDataLoader:
    """CheXpert data loader for external testing"""

    def __init__(self, config, data_path=None):
        """
        Args:
            config: Configuration object
            data_path: Path to CheXpert dataset (overrides config)
        """
        self.config = config
        self.base_path = data_path if data_path else r"D:\\Data\\CheXpert\\CheXpert-v1.0-small"

        # Load class names from JSON configuration
        from utils import load_class_names
        self.class_names = load_class_names('chexpert_5class')
        self.num_classes = len(self.class_names)

    def load_test_data(self):
        """
        Load CheXpert test set

        Returns:
            image_paths: List of image paths
            labels: Array of multi-label annotations
        """
        print("\n" + "=" * 80)
        print("Loading CheXpert Test Dataset")
        print("=" * 80)

        # Load metadata
        metadata_path = os.path.join(self.base_path, 'test_labels.csv')

        print(f"Loading metadata from: {metadata_path}")
        metadata = pd.read_csv(metadata_path)

        print(f"\nTest images loaded: {len(metadata)}")

        # Build image paths (prepend base_path to relative paths)
        image_paths = []
        for rel_path in metadata['Path']:
            # Path in CSV is like: test/patient64741/study1/view1_frontal.jpg
            # We need: D:\Data\CheXpert\CheXpert-v1.0-small\test\patient64741\study1\view1_frontal.jpg
            full_path = os.path.join(self.base_path, rel_path)
            full_path = full_path.replace('/', '\\')  # Windows path fix
            image_paths.append(full_path)

        # Process labels
        labels = self._process_labels(metadata)

        # Print label statistics
        self._print_label_statistics(labels)

        return image_paths, labels

    def _process_labels(self, metadata):
        """
        Process CheXpert labels to multi-label format

        CheXpert labels:
          1.0 = Positive
          0.0 = Negative
         -1.0 = Uncertain
          NaN = Not mentioned

        We treat:
          1.0 → 1 (positive)
          0.0, -1.0, NaN → 0 (negative or uncertain)

        Args:
            metadata: DataFrame with CheXpert labels

        Returns:
            labels: Array of shape (N, num_classes) with binary labels
        """
        labels = np.zeros((len(metadata), self.num_classes), dtype=np.float32)

        for i, class_name in enumerate(self.class_names):
            if class_name in metadata.columns:
                # Get label column
                label_col = metadata[class_name].values

                # Convert to binary: 1.0 → 1, everything else → 0
                labels[:, i] = (label_col == 1.0).astype(np.float32)

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
        dataset = CheXpertDataset(
            image_paths=image_paths,
            labels=labels,
            transform=preprocess,
            tokenizer=tokenizer,
            text_prompts=text_prompts
        )

        workers = self.config.data.num_workers
        loader_kwargs = dict(
            dataset=dataset,
            batch_size=self.config.data.batch_size,
            shuffle=False,  # Don't shuffle test set
            num_workers=workers,
            pin_memory=True
        )
        if workers > 0:
            loader_kwargs.update(
                persistent_workers=True,
                prefetch_factor=2
            )

        dataloader = DataLoader(**loader_kwargs)

        print(f"\nCreated CheXpert test dataloader: {len(dataset)} samples, "
              f"{len(dataloader)} batches")

        return dataloader

    def _print_label_statistics(self, labels):
        """Print label statistics"""
        print("\n" + "-" * 80)
        print("CheXpert Test Set Label Statistics")
        print("-" * 80)

        total_samples = len(labels)
        positive_counts = labels.sum(axis=0)
        positive_ratios = positive_counts / total_samples * 100

        print(f"{'Class':<35} {'Positive':<10} {'Ratio':<10}")
        print("-" * 80)
        for i, class_name in enumerate(self.class_names):
            print(f"{class_name:<35} {int(positive_counts[i]):<10} {positive_ratios[i]:>6.2f}%")

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

    loader = CheXpertDataLoader(config)
    image_paths, labels = loader.load_test_data()

    print(f"\nLoaded {len(image_paths)} test images")
    print(f"Label shape: {labels.shape}")

    # Check if images exist
    print("\nChecking first 3 image paths:")
    for i in range(min(3, len(image_paths))):
        exists = "OK" if os.path.exists(image_paths[i]) else "MISSING"
        print(f"  [{exists}] {image_paths[i]}")

    print("\nCheXpert test set loaded successfully!")
