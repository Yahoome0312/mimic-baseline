"""
Test script for class names loading
"""

from utils import load_class_names, list_available_datasets

print("=" * 80)
print("Testing Class Names Loading")
print("=" * 80)

# List available datasets
print("\n1. Available datasets:")
datasets = list_available_datasets()
print(f"   {datasets}")

# Test loading MIMIC-CXR
print("\n2. Loading MIMIC-CXR class names:")
try:
    mimic_classes = load_class_names('mimic_cxr')
    print(f"   [OK] Successfully loaded {len(mimic_classes)} classes")
except Exception as e:
    print(f"   [ERROR] Error: {e}")

# Test loading ChestXray14
print("\n3. Loading ChestXray14 class names:")
try:
    chestxray14_classes = load_class_names('chestxray14')
    print(f"   [OK] Successfully loaded {len(chestxray14_classes)} classes")
except Exception as e:
    print(f"   [ERROR] Error: {e}")

# Test loading non-existent dataset
print("\n4. Testing error handling (non-existent dataset):")
try:
    fake_classes = load_class_names('fake_dataset')
    print(f"   [ERROR] Should have raised an error!")
except FileNotFoundError as e:
    print(f"   [OK] Correctly raised FileNotFoundError")
except Exception as e:
    print(f"   [ERROR] Unexpected error: {e}")

print("\n" + "=" * 80)
print("Test completed!")
print("=" * 80)
