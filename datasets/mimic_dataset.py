"""
MIMIC-CXR Dataset Module

This module handles loading and preprocessing of the MIMIC-CXR dataset.
MIMIC-CXR is a multi-label chest X-ray classification dataset with 14 pathology classes.
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


class MIMICCXRDataset(Dataset):
    """MIMIC-CXR Dataset for multi-label classification"""

    def __init__(self, image_paths, labels, transform=None, tokenizer=None, text_prompts=None, reports=None):
        """
        Args:
            image_paths: List of image file paths
            labels: Array of multi-label annotations (N, num_classes)
            transform: Image transformations
            tokenizer: Text tokenizer (optional)
            text_prompts: List of text prompts for each class (optional, used for zero-shot)
            reports: List of report texts, one per image (optional, used for fine-tuning)
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.tokenizer = tokenizer
        self.text_prompts = text_prompts
        self.reports = reports  # NEW: per-image report texts

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

        # Tokenize text
        if self.tokenizer:
            if self.reports is not None:
                # Use per-image report text (for fine-tuning)
                text = self.tokenizer([self.reports[idx]])
            elif self.text_prompts is not None:
                # Use class prompts (for zero-shot)
                text = self.tokenizer(self.text_prompts)
            else:
                text = None
        else:
            text = None

        return image, text, label


class MIMICCXRDataLoader:
    """MIMIC-CXR data loader and preprocessor"""

    # 14 CheXpert classes
    CHEXPERT_CLASSES = [
        'Atelectasis',
        'Cardiomegaly',
        'Consolidation',
        'Edema',
        'Enlarged Cardiomediastinum',
        'Fracture',
        'Lung Lesion',
        'Lung Opacity',
        'No Finding',
        'Pleural Effusion',
        'Pleural Other',
        'Pneumonia',
        'Pneumothorax',
        'Support Devices'
    ]

    def __init__(self, config):
        """
        Args:
            config: Configuration object
        """
        self.config = config
        self.base_path = config.paths.base_data_path
        self.image_dir = os.path.join(self.base_path, 'MIMIC-CXR-JPG', 'files')
        self.reports_dir = os.path.join(self.base_path, 'reports', 'files')  # NEW: reports directory
        self.num_classes = len(self.CHEXPERT_CLASSES)

    def _extract_findings_impression(self, report_text):
        """
        Extract FINDINGS and IMPRESSION sections from report text

        Args:
            report_text: Full report text

        Returns:
            Combined findings and impression text
        """
        import re

        findings = ""
        impression = ""

        # Extract FINDINGS section
        findings_match = re.search(r'FINDINGS?:\s*(.*?)(?=IMPRESSION:|$)', report_text, re.DOTALL | re.IGNORECASE)
        if findings_match:
            findings = findings_match.group(1).strip()

        # Extract IMPRESSION section
        impression_match = re.search(r'IMPRESSION:\s*(.*?)$', report_text, re.DOTALL | re.IGNORECASE)
        if impression_match:
            impression = impression_match.group(1).strip()

        # Combine findings and impression
        combined = ""
        if findings:
            combined += findings
        if impression:
            if combined:
                combined += " "
            combined += impression

        # Clean up: remove extra whitespace, newlines
        combined = re.sub(r'\s+', ' ', combined).strip()

        return combined if combined else "No findings or impression available."

    def _load_report_for_study(self, subject_id, study_id):
        """
        Load report text for a given study

        Args:
            subject_id: Patient subject ID
            study_id: Study ID

        Returns:
            Report text (findings + impression) or empty string if not found
        """
        # Construct report path: reports/files/p{XX}/p{subject_id}/s{study_id}.txt
        p_prefix = 'p' + subject_id[:2]
        p_subject = 'p' + subject_id
        s_study = 's' + study_id

        report_path = os.path.join(
            self.reports_dir,
            p_prefix,
            p_subject,
            f"{s_study}.txt"
        )

        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                report_text = f.read()
            return self._extract_findings_impression(report_text)
        except FileNotFoundError:
            return ""  # Return empty string if report not found

    def load_data(self, use_provided_split=True, label_policy='ignore_uncertain'):
        """
        Load MIMIC-CXR data

        Args:
            use_provided_split: Whether to use the official train/val/test split
            label_policy: How to handle uncertain labels (-1.0)
                - 'ignore_uncertain': Treat as 0 (negative)
                - 'as_positive': Treat as 1 (positive)
                - 'as_negative': Treat as 0 (negative)

        Returns:
            image_paths: List of image paths
            labels: Array of multi-label annotations
            reports: List of report texts (Findings + Impression) for each image
            split_info: Dict with split information if use_provided_split=True
        """
        print("\n" + "=" * 80)
        print("Loading MIMIC-CXR Dataset")
        print("=" * 80)

        # Load metadata
        metadata_path = os.path.join(self.base_path, 'mimic-cxr-2.0.0-metadata.csv')
        chexpert_path = os.path.join(self.base_path, 'mimic-cxr-2.0.0-chexpert.csv')
        split_path = os.path.join(self.base_path, 'mimic-cxr-2.0.0-split.csv')

        print(f"Loading metadata from: {metadata_path}")
        metadata = pd.read_csv(metadata_path)

        print(f"Loading CheXpert labels from: {chexpert_path}")
        chexpert = pd.read_csv(chexpert_path)

        if use_provided_split:
            print(f"Loading official splits from: {split_path}")
            splits = pd.read_csv(split_path)
            # Merge metadata with splits
            metadata = metadata.merge(splits, on=['dicom_id', 'study_id', 'subject_id'])

        # Merge metadata with chexpert labels (on study level)
        data = metadata.merge(chexpert, on=['subject_id', 'study_id'])

        print(f"\nTotal images loaded: {len(data)}")

        # Build image paths and load reports
        print("\nLoading reports (Findings + Impression)...")
        image_paths = []
        reports = []  # NEW: list to store report text for each image

        for idx, row in data.iterrows():
            subject_id = str(row['subject_id'])
            study_id = str(row['study_id'])
            dicom_id = row['dicom_id']

            # Construct image path: files/p{first2}/p{subject_id}/s{study_id}/{dicom_id}.jpg
            p_prefix = 'p' + subject_id[:2]
            p_subject = 'p' + subject_id
            s_study = 's' + study_id

            img_path = os.path.join(
                self.image_dir,
                p_prefix,
                p_subject,
                s_study,
                f"{dicom_id}.jpg"
            )
            image_paths.append(img_path)

            # Load report for this study (multiple images may share the same report)
            report_text = self._load_report_for_study(subject_id, study_id)
            reports.append(report_text)

            # Progress indicator
            if (idx + 1) % 50000 == 0:
                print(f"  Loaded {idx + 1}/{len(data)} images and reports...")

        # Count how many reports were successfully loaded
        reports_loaded = sum(1 for r in reports if r and r != "")
        print(f"\nSuccessfully loaded {reports_loaded}/{len(reports)} reports ({100*reports_loaded/len(reports):.1f}%)")

        # Process labels
        labels = data[self.CHEXPERT_CLASSES].values

        # Handle NaN and uncertain labels
        print(f"\nProcessing labels with policy: '{label_policy}'")

        if label_policy == 'ignore_uncertain' or label_policy == 'as_negative':
            # Replace -1.0 (uncertain) and NaN with 0.0
            labels = np.where(labels == -1.0, 0.0, labels)
            labels = np.nan_to_num(labels, nan=0.0)
        elif label_policy == 'as_positive':
            # Replace -1.0 (uncertain) with 1.0, NaN with 0.0
            labels = np.where(labels == -1.0, 1.0, labels)
            labels = np.nan_to_num(labels, nan=0.0)

        # Ensure binary labels
        labels = labels.astype(np.float32)

        # Print label statistics
        self._print_label_statistics(labels)

        # Plot class distribution
        self._plot_class_distribution(labels)

        if use_provided_split:
            split_info = data['split'].values
            return image_paths, labels, reports, split_info
        else:
            return image_paths, labels, reports, None

    def split_dataset(self, image_paths, labels, reports, split_info=None):
        """
        Split dataset into train/val/test

        Args:
            image_paths: List of image paths
            labels: Label array
            reports: List of report texts
            split_info: Pre-defined split info (if available)

        Returns:
            Dictionary with train/val/test splits
        """
        print("\n" + "=" * 80)
        print("Splitting Dataset")
        print("=" * 80)

        if split_info is not None:
            # Use provided split
            train_mask = split_info == 'train'
            val_mask = split_info == 'validate'
            test_mask = split_info == 'test'

            X_train = [image_paths[i] for i in range(len(image_paths)) if train_mask[i]]
            y_train = labels[train_mask]
            reports_train = [reports[i] for i in range(len(reports)) if train_mask[i]]

            X_val = [image_paths[i] for i in range(len(image_paths)) if val_mask[i]]
            y_val = labels[val_mask]
            reports_val = [reports[i] for i in range(len(reports)) if val_mask[i]]

            X_test = [image_paths[i] for i in range(len(image_paths)) if test_mask[i]]
            y_test = labels[test_mask]
            reports_test = [reports[i] for i in range(len(reports)) if test_mask[i]]

            print(f"Using official MIMIC-CXR splits:")
        else:
            # Create custom split
            # First split: train+val vs test
            X_temp, X_test, y_temp, y_test, reports_temp, reports_test = train_test_split(
                image_paths, labels, reports,
                test_size=self.config.data.test_size,
                random_state=self.config.data.random_state
            )

            # Second split: train vs val
            val_ratio = self.config.data.val_size / (1 - self.config.data.test_size)
            X_train, X_val, y_train, y_val, reports_train, reports_val = train_test_split(
                X_temp, y_temp, reports_temp,
                test_size=val_ratio,
                random_state=self.config.data.random_state
            )

            print(f"Using custom splits (test={self.config.data.test_size}, val={self.config.data.val_size}):")

        print(f"  Train: {len(X_train)} images")
        print(f"  Val:   {len(X_val)} images")
        print(f"  Test:  {len(X_test)} images")

        return {
            'train': (X_train, y_train, reports_train),
            'val': (X_val, y_val, reports_val),
            'test': (X_test, y_test, reports_test)
        }

    def create_dataloaders(self, data_splits, preprocess, tokenizer, text_prompts, use_reports=True):
        """
        Create PyTorch DataLoaders

        Args:
            data_splits: Dictionary with data splits (each split contains image_paths, labels, reports)
            preprocess: Image preprocessing transform
            tokenizer: Text tokenizer
            text_prompts: List of text prompts (used for zero-shot if use_reports=False)
            use_reports: Whether to use reports (True for fine-tuning) or text_prompts (False for zero-shot)

        Returns:
            Dictionary of DataLoaders
        """
        dataloaders = {}

        for split_name, (image_paths, labels, reports) in data_splits.items():
            dataset = MIMICCXRDataset(
                image_paths=image_paths,
                labels=labels,
                transform=preprocess,
                tokenizer=tokenizer,
                text_prompts=None if use_reports else text_prompts,  # Use prompts only for zero-shot
                reports=reports if use_reports else None  # Use reports for fine-tuning
            )

            # Use shuffle for train, not for val/test
            shuffle = (split_name == 'train')

            dataloader = DataLoader(
                dataset,
                batch_size=self.config.data.batch_size,
                shuffle=shuffle,
                num_workers=self.config.data.num_workers,
                pin_memory=True
            )

            dataloaders[split_name] = dataloader
            print(f"Created {split_name} dataloader: {len(dataset)} samples, "
                  f"{len(dataloader)} batches")

        return dataloaders

    def _print_label_statistics(self, labels):
        """Print label statistics"""
        print("\n" + "-" * 80)
        print("Label Statistics")
        print("-" * 80)

        total_samples = len(labels)
        positive_counts = labels.sum(axis=0)
        positive_ratios = positive_counts / total_samples * 100

        print(f"{'Class':<30} {'Positive':<10} {'Ratio':<10}")
        print("-" * 80)
        for i, class_name in enumerate(self.CHEXPERT_CLASSES):
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

    def _plot_class_distribution(self, labels):
        """Plot class distribution"""
        output_path = os.path.join(self.config.paths.output_dir, 'class_distribution.png')

        positive_counts = labels.sum(axis=0)

        plt.figure(figsize=(14, 6))
        bars = plt.bar(range(len(self.CHEXPERT_CLASSES)), positive_counts)

        # Color bars
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.CHEXPERT_CLASSES)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        plt.xlabel('Class', fontsize=12)
        plt.ylabel('Number of Positive Samples', fontsize=12)
        plt.title('MIMIC-CXR Class Distribution', fontsize=14, fontweight='bold')
        plt.xticks(range(len(self.CHEXPERT_CLASSES)), self.CHEXPERT_CLASSES,
                   rotation=45, ha='right')

        # Add value labels on bars
        for i, (bar, count) in enumerate(zip(bars, positive_counts)):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(count)}',
                    ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\nClass distribution plot saved to: {output_path}")


if __name__ == "__main__":
    # Test the data loader
    from config import Config

    config = Config()
    config.update_paths(base_data_path=r"D:\Data\MIMIC")

    loader = MIMICCXRDataLoader(config)
    image_paths, labels, split_info = loader.load_data(use_provided_split=True)

    print(f"\nLoaded {len(image_paths)} images")
    print(f"Label shape: {labels.shape}")

    # Test split
    splits = loader.split_dataset(image_paths, labels, split_info)
    print("\nSplit test passed!")
