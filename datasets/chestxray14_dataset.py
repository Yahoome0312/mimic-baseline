"""
ChestXray14 (NIH CXR8) Dataset Module

This module handles loading the ChestXray14 dataset for external testing.
Used for zero-shot evaluation of models trained on MIMIC-CXR.
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class ChestXray14Dataset(Dataset):
    """ChestXray14 Dataset for multi-label classification"""

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


class ChestXray14DataLoader:
    """ChestXray14 data loader for external testing"""

    def __init__(self, config, data_path=None):
        """
        Args:
            config: Configuration object
            data_path: Path to ChestXray14 dataset (overrides config)
        """
        self.config = config
        self.base_path = data_path if data_path else r"D:\Data\ChestXray14\CXR8"
        self.image_dir = os.path.join(self.base_path, 'images', 'images')

        # Load class names from JSON configuration
        from utils import load_class_names
        self.class_names = load_class_names('chestxray14')
        self.num_classes = len(self.class_names)

    def load_test_data(self):
        """
        Load ChestXray14 test set

        Returns:
            image_paths: List of image paths
            labels: Array of multi-label annotations
        """
        print("\n" + "=" * 80)
        print("Loading ChestXray14 Test Dataset")
        print("=" * 80)

        # Load metadata
        metadata_path = os.path.join(self.base_path, 'Data_Entry_2017_v2020.csv')
        test_list_path = os.path.join(self.base_path, 'test_list.txt')

        print(f"Loading metadata from: {metadata_path}")
        metadata = pd.read_csv(metadata_path)

        print(f"Loading test list from: {test_list_path}")
        with open(test_list_path, 'r') as f:
            test_images = set([line.strip() for line in f.readlines()])

        # Filter test images
        test_data = metadata[metadata['Image Index'].isin(test_images)]

        print(f"\nTest images loaded: {len(test_data)}")

        # Build image paths
        image_paths = []
        for img_name in test_data['Image Index']:
            img_path = os.path.join(self.image_dir, img_name)
            image_paths.append(img_path)

        # Process labels
        labels = self._process_labels(test_data['Finding Labels'].values)

        # Print label statistics
        self._print_label_statistics(labels)

        return image_paths, labels

    def _process_labels(self, finding_labels):
        """
        Process finding labels to multi-label format

        Args:
            finding_labels: Array of finding label strings (e.g., "Cardiomegaly|Effusion")

        Returns:
            labels: Array of shape (N, num_classes) with binary labels
        """
        labels = np.zeros((len(finding_labels), self.num_classes), dtype=np.float32)

        for i, findings in enumerate(finding_labels):
            if findings == 'No Finding':
                # No Finding: all zeros
                continue

            # Split multiple findings
            finding_list = findings.split('|')

            for finding in finding_list:
                finding = finding.strip()
                if finding in self.class_names:
                    class_idx = self.class_names.index(finding)
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
        dataset = ChestXray14Dataset(
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
        )
        if workers > 0:
            loader_kwargs.update(
                persistent_workers=True,
                prefetch_factor=2
            )

        dataloader = DataLoader(**loader_kwargs)

        print(f"\nCreated ChestXray14 test dataloader: {len(dataset)} samples, "
              f"{len(dataloader)} batches")

        return dataloader

    def _print_label_statistics(self, labels):
        """Print label statistics"""
        print("\n" + "-" * 80)
        print("ChestXray14 Test Set Label Statistics")
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

    loader = ChestXray14DataLoader(config)
    image_paths, labels = loader.load_test_data()

    print(f"\nLoaded {len(image_paths)} test images")
    print(f"Label shape: {labels.shape}")

    # Check if images exist
    print("\nChecking first 3 image paths:")
    for i in range(min(3, len(image_paths))):
        exists = "OK" if os.path.exists(image_paths[i]) else "MISSING"
        print(f"  [{exists}] {image_paths[i]}")

    print("\nChestXray14 test set loaded successfully!")
