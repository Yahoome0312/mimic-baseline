"""
Test script for ChestXray14 data loading

This script tests if the ChestXray14 data can be loaded correctly for external testing.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from datasets import ChestXray14DataLoader


def test_chestxray14_loading():
    """Test ChestXray14 data loading"""

    print("=" * 80)
    print("Testing ChestXray14 Data Loading")
    print("=" * 80)

    # Create config
    config = Config()
    print(f"\nConfiguration:")
    print(f"  ChestXray14 path: D:\\Data\\ChestXray14\\CXR8")
    print(f"  Batch size: {config.data.batch_size}")

    # Create data loader
    print("\n" + "-" * 80)
    print("Creating ChestXray14 data loader...")
    data_loader = ChestXray14DataLoader(config)

    print(f"\nChestXray14 Classes ({len(data_loader.CHESTXRAY14_CLASSES)}):")
    for i, cls in enumerate(data_loader.CHESTXRAY14_CLASSES, 1):
        print(f"  {i:2}. {cls}")

    # Test data loading
    print("\n" + "-" * 80)
    print("Loading ChestXray14 test set (this may take a while)...")
    try:
        image_paths, labels = data_loader.load_test_data()

        print(f"\n[OK] Data loaded successfully!")
        print(f"  Total test images: {len(image_paths)}")
        print(f"  Label shape: {labels.shape}")

        # Check first few images
        print(f"\n" + "-" * 80)
        print("Sample image paths:")
        for i in range(min(3, len(image_paths))):
            exists = "OK" if os.path.exists(image_paths[i]) else "MISSING"
            print(f"  [{exists}] {image_paths[i]}")

        print("\n" + "=" * 80)
        print("[OK] ChestXray14 test data loaded successfully!")
        print("=" * 80)

        return True

    except Exception as e:
        print(f"\n[ERROR] Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_chestxray14_loading()

    if success:
        print("\nYou can now test on ChestXray14 with:")
        print("  # Train on MIMIC, test on ChestXray14 (zero-shot)")
        print("  python main.py --method zeroshot --external_test")
        print()
        print("  # Train on MIMIC, fine-tune and test on ChestXray14")
        print("  python main.py --method finetune --external_test")
    else:
        print("\nPlease fix the errors above before running cross-dataset testing.")

    sys.exit(0 if success else 1)
