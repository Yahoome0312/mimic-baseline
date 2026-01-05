"""
Test script for MIMIC-CXR data loading

This script tests if the MIMIC-CXR data can be loaded correctly.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from datasets import MIMICCXRDataLoader
from utils import load_class_config

def test_mimic_data_loading():
    """Test MIMIC-CXR data loading"""

    print("=" * 80)
    print("Testing MIMIC-CXR Data Loading")
    print("=" * 80)

    # Create config
    config = Config()

    # Load MIMIC-CXR class configuration
    mimic_class_config = load_class_config('mimic_cxr', verbose=False)

    print(f"\nConfiguration:")
    print(f"  Data path: {config.paths.base_data_path}")
    print(f"  Output dir: {config.paths.output_dir}")
    print(f"  Batch size: {config.data.batch_size}")
    print(f"  Task type: {mimic_class_config['task_type']}")
    print(f"  Number of classes: {mimic_class_config['num_classes']}")

    # Create data loader
    print("\n" + "-" * 80)
    print("Creating MIMIC-CXR data loader...")
    data_loader = MIMICCXRDataLoader(config)

    # Test data loading (load only a small subset for testing)
    print("\n" + "-" * 80)
    print("Loading MIMIC-CXR data (this may take a while)...")
    try:
        image_paths, labels, reports, split_info = data_loader.load_data(
            use_provided_split=config.data.use_provided_split,
            label_policy=config.data.label_policy
        )

        print(f"\n[OK] Data loaded successfully!")
        print(f"  Total images: {len(image_paths)}")
        print(f"  Label shape: {labels.shape}")
        print(f"  Reports loaded: {len(reports)}")
        print(f"  Split info available: {split_info is not None}")

        # Check first few images
        print(f"\n" + "-" * 80)
        print("Sample image paths:")
        for i in range(min(3, len(image_paths))):
            exists = "OK" if os.path.exists(image_paths[i]) else "MISSING"
            print(f"  [{exists}] {image_paths[i]}")

        # Test data splitting
        print(f"\n" + "-" * 80)
        print("Testing data splitting...")
        data_splits = data_loader.split_dataset(image_paths, labels, split_info)

        print(f"\n[OK] Data split successfully!")
        for split_name, (X, y) in data_splits.items():
            print(f"  {split_name}: {len(X)} images, label shape: {y.shape}")

        # Print class distribution in training set
        print(f"\n" + "-" * 80)
        print("Training set class distribution:")
        train_labels = data_splits['train'][1]
        for i, class_name in enumerate(mimic_class_config['class_names']):
            positive_count = int(train_labels[:, i].sum())
            percentage = (positive_count / len(train_labels)) * 100
            print(f"  {class_name:<30}: {positive_count:>6} ({percentage:>5.2f}%)")

        print("\n" + "=" * 80)
        print("[OK] All tests passed!")
        print("=" * 80)

        return True

    except Exception as e:
        print(f"\n[ERROR] Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_mimic_data_loading()

    if success:
        print("\nYou can now run the full training with:")
        print("  python main.py --method finetune")
    else:
        print("\nPlease fix the errors above before running training.")

    sys.exit(0 if success else 1)
