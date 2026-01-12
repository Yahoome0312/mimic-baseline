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

    def __init__(self, image_paths, labels, transform=None, tokenizer=None, text_prompts=None, reports=None, context_length=256, cache_dir=None, split_name='unknown'):
        """
        Args:
            image_paths: List of image file paths
            labels: Array of multi-label annotations (N, num_classes)
            transform: Image transformations
            tokenizer: Text tokenizer (if provided, will pre-tokenize all reports)
            text_prompts: List of text prompts for each class (unused)
            reports: List of report texts, one per image (optional, used for fine-tuning)
            context_length: Context length for tokenizer (default: 256)
            cache_dir: Directory for tokenization cache (optional)
            split_name: Dataset split name (e.g., 'train', 'val', 'test')
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

        # Tokenization with caching
        if reports is not None and tokenizer is not None:
            from utils import TokenizationCache

            # Try to use cache if cache_dir is provided
            if cache_dir is not None:
                cache_manager = TokenizationCache(cache_dir)
                cache_path = cache_manager.generate_cache_path(
                    split_name, context_length, len(reports)
                )

                # Try to load from cache
                cached_data = cache_manager.load_cache(cache_path, len(reports))

                if cached_data is not None:
                    # Cache hit - use cached tokenized reports
                    self.tokenized_reports = cached_data['tokenized_reports']
                    self.reports = None
                else:
                    # Cache miss - tokenize and save to cache (in batches to avoid memory issues)
                    print(f"[Dataset] Pre-tokenizing {len(reports)} reports...")
                    batch_size = 1000
                    all_tokenized = []

                    from tqdm import tqdm
                    for i in tqdm(range(0, len(reports), batch_size), desc="Tokenizing batches"):
                        batch_reports = reports[i:i+batch_size]
                        batch_tokenized = tokenizer(batch_reports)
                        all_tokenized.append(batch_tokenized)

                    self.tokenized_reports = torch.cat(all_tokenized, dim=0)
                    self.reports = None  # Free memory
                    print(f"[Dataset] Tokenization completed: {self.tokenized_reports.shape}")

                    # Save to cache
                    cache_manager.save_cache(
                        self.tokenized_reports, cache_path, context_length
                    )
            else:
                # No cache - direct tokenization in batches
                print(f"[Dataset] Pre-tokenizing {len(reports)} reports (no cache)...")
                batch_size = 1000
                all_tokenized = []

                from tqdm import tqdm
                for i in tqdm(range(0, len(reports), batch_size), desc="Tokenizing batches"):
                    batch_reports = reports[i:i+batch_size]
                    batch_tokenized = tokenizer(batch_reports)
                    all_tokenized.append(batch_tokenized)

                self.tokenized_reports = torch.cat(all_tokenized, dim=0)
                self.reports = None
                print(f"[Dataset] Tokenization completed: {self.tokenized_reports.shape}")
        else:
            self.reports = reports
            self.tokenized_reports = None

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

        # Return pre-tokenized text if available, otherwise raw text
        if self.tokenized_reports is not None:
            text = self.tokenized_reports[idx]
        elif self.reports is not None:
            text = self.reports[idx]
        else:
            # Return empty tensor instead of None for collate compatibility
            text = torch.tensor([])

        return image, text, label


class MIMICCXRDataLoader:
    """MIMIC-CXR data loader and preprocessor"""

    def __init__(self, config):
        """
        Args:
            config: Configuration object
        """
        self.config = config
        self.base_path = config.paths.base_data_path

        # Try pre-resized images first, fallback to original if not found
        resized_dir = os.path.join(self.base_path, 'MIMIC-CXR-JPG', 'files_224')
        original_dir = os.path.join(self.base_path, 'MIMIC-CXR-JPG', 'files')

        if os.path.exists(resized_dir):
            self.image_dir = resized_dir
            print(f"[Dataset] Using pre-resized images: {resized_dir}")
        elif os.path.exists(original_dir):
            self.image_dir = original_dir
            print(f"[Dataset] Using original images (will resize on-the-fly): {original_dir}")
        else:
            # Neither directory found - will fail later with clear error
            self.image_dir = original_dir  # Use original path for error messages
            print(f"[Dataset] ERROR: Image directory not found!")
            print(f"  Tried: {resized_dir}")
            print(f"  Tried: {original_dir}")
            print(f"  Please check base_data_path in config: {self.base_path}")

        self.reports_dir = os.path.join(self.base_path, 'reports', 'files')

        # Load class names from JSON configuration
        from utils import load_class_names
        self.class_names = load_class_names('mimic_cxr')
        self.num_classes = len(self.class_names)

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

        # ===== 向量化构建路径 =====
        print("\nBuilding image paths...")
        data['subject_id_str'] = data['subject_id'].astype(str)
        data['study_id_str'] = data['study_id'].astype(str)
        data['p_prefix'] = 'p' + data['subject_id_str'].str[:2]
        data['p_subject'] = 'p' + data['subject_id_str']
        data['s_study'] = 's' + data['study_id_str']

        data['img_path'] = data.apply(
            lambda row: os.path.join(
                self.image_dir, row['p_prefix'], row['p_subject'],
                row['s_study'], f"{row['dicom_id']}.jpg"
            ), axis=1
        )

        # 过滤存在的文件
        print("Checking image existence...")
        data['exists'] = data['img_path'].apply(os.path.exists)
        valid_data = data[data['exists']].copy()
        skipped_count = len(data) - len(valid_data)

        print(f"\nDataset loading summary:")
        print(f"  Total records in CSV: {len(data)}")
        print(f"  Valid images found: {len(valid_data)}")
        print(f"  Missing images skipped: {skipped_count}")
        print(f"  Success rate: {100*len(valid_data)/len(data):.2f}%")

        # ===== 边界检查：如果没有有效数据，提前返回 =====
        if len(valid_data) == 0:
            print("\n" + "=" * 80)
            print("ERROR: No valid images found!")
            print("=" * 80)
            print(f"Please check:")
            print(f"  1. Image directory exists: {self.image_dir}")
            print(f"  2. CSV file contains valid records: {self.metadata_file}")
            print(f"  3. Image files exist in the directory structure")
            print("=" * 80)
            raise FileNotFoundError(
                f"No valid images found in {self.image_dir}. "
                f"Please check the data path configuration."
            )

        # ===== 检查tokenization缓存 =====
        from utils import TokenizationCache

        cache_exists = False
        if self.config.paths.tokenized_cache_dir is not None and use_provided_split:
            # 检查所有split的缓存是否存在
            cache_manager = TokenizationCache(self.config.paths.tokenized_cache_dir)
            split_sizes = {
                'train': (valid_data['split'] == 'train').sum(),
                'val': (valid_data['split'] == 'validate').sum(),
                'test': (valid_data['split'] == 'test').sum()
            }

            cache_exists = all(
                os.path.exists(cache_manager.generate_cache_path(
                    split_name, self.config.model.context_length, size
                )) for split_name, size in split_sizes.items()
            )

        # ===== 根据缓存情况选择路径 =====
        if cache_exists:
            print("\n" + "=" * 80)
            print("✓ Tokenization cache found! Skipping report loading...")
            print("=" * 80)
            print(f"Reports will be loaded from cache during dataset initialization")
            print(f"Cache location: {self.config.paths.tokenized_cache_dir}")
            reports = [None] * len(valid_data)  # 占位符
            print(f"Created {len(reports)} placeholder entries for cached reports")
            print("=" * 80)
        else:
            print("\n" + "=" * 80)
            print("✗ No tokenization cache. Loading reports (optimized)...")
            print("=" * 80)

            # 预加载唯一study的报告（减少重复读取）
            unique_studies = valid_data[['subject_id_str', 'study_id_str']].drop_duplicates()
            print(f"Loading {len(unique_studies)} unique study reports (for {len(valid_data)} images)...")
            report_cache = {}

            from tqdm import tqdm
            for _, row in tqdm(unique_studies.iterrows(), total=len(unique_studies), desc="Loading reports"):
                key = (row['subject_id_str'], row['study_id_str'])
                report_cache[key] = self._load_report_for_study(key[0], key[1])

            # 快速映射报告到每个图像
            print("Mapping reports to images...")
            valid_data['report'] = valid_data.apply(
                lambda row: report_cache[(row['subject_id_str'], row['study_id_str'])],
                axis=1
            )
            reports = valid_data['report'].tolist()

            reports_loaded = sum(1 for r in reports if r and r != "")
            print(f"\n✓ Successfully loaded {reports_loaded}/{len(reports)} reports ({100*reports_loaded/len(reports):.1f}%)")
            print("=" * 80)

        # ===== 构建返回值 =====
        image_paths = valid_data['img_path'].tolist()
        labels = valid_data[self.class_names].values.astype(np.float32)

        if use_provided_split:
            split_info = valid_data['split'].values
        else:
            split_info = None

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

        # Plot class distribution (only if not already exists)
        output_path = os.path.join(self.config.paths.output_dir, 'class_distribution.png')
        if not os.path.exists(output_path):
            self._plot_class_distribution(labels)
        else:
            print(f"\nClass distribution plot already exists: {output_path}")

        return image_paths, labels, reports, split_info

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
            tokenizer: Text tokenizer (will pre-tokenize reports if provided)
            text_prompts: List of text prompts (unused)
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
                tokenizer=tokenizer if use_reports else None,  # Pass tokenizer for pre-tokenization
                reports=reports if use_reports else None,  # Use reports for fine-tuning
                context_length=self.config.model.context_length,
                cache_dir=self.config.paths.tokenized_cache_dir if use_reports else None,  # 推理时不用缓存
                split_name=split_name  # Pass split name for cache filename
            )

            # Use shuffle for train, not for val/test
            shuffle = (split_name == 'train')

            workers = self.config.data.num_workers
            loader_kwargs = dict(
                dataset=dataset,
                batch_size=self.config.data.batch_size,
                shuffle=shuffle,
                num_workers=workers,
                pin_memory=True  # 加速 CPU→GPU 数据传输
            )
            if workers > 0:
                loader_kwargs.update(
                    persistent_workers=True,
                    prefetch_factor=2
                )

            dataloader = DataLoader(**loader_kwargs)

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

    def _plot_class_distribution(self, labels):
        """Plot class distribution"""
        output_path = os.path.join(self.config.paths.output_dir, 'class_distribution.png')

        positive_counts = labels.sum(axis=0)

        plt.figure(figsize=(14, 6))
        bars = plt.bar(range(len(self.class_names)), positive_counts)

        # Color bars
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.class_names)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        plt.xlabel('Class', fontsize=12)
        plt.ylabel('Number of Positive Samples', fontsize=12)
        plt.title('MIMIC-CXR Class Distribution', fontsize=14, fontweight='bold')
        plt.xticks(range(len(self.class_names)), self.class_names,
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
