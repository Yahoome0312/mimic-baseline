"""
Installation Test Script

This script tests whether all dependencies are installed correctly
and all required paths exist.
"""

import sys
import os


def test_imports():
    """Test if all required packages can be imported"""
    print("=" * 80)
    print("Testing Package Imports...")
    print("=" * 80)

    packages = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('PIL', 'Pillow'),
        ('sklearn', 'scikit-learn'),
        ('matplotlib', 'Matplotlib'),
        ('seaborn', 'Seaborn'),
        ('tqdm', 'tqdm'),
        ('open_clip', 'open_clip_torch'),
    ]

    all_success = True
    for module, name in packages:
        try:
            __import__(module)
            print(f"✓ {name:20s} - OK")
        except ImportError:
            print(f"✗ {name:20s} - NOT FOUND")
            all_success = False

    print()
    if all_success:
        print("✓ All required packages are installed!")
    else:
        print("✗ Some packages are missing. Please run: pip install -r requirements.txt")

    return all_success


def test_cuda():
    """Test CUDA availability"""
    print("\n" + "=" * 80)
    print("Testing CUDA...")
    print("=" * 80)

    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                props = torch.cuda.get_device_properties(i)
                print(f"    Memory: {props.total_memory / 1024**3:.2f} GB")
        else:
            print("⚠ CUDA not available. Training will use CPU (much slower).")

    except ImportError:
        print("✗ PyTorch not installed")
        return False

    return True


def test_paths():
    """Test if all required paths exist"""
    print("\n" + "=" * 80)
    print("Testing Paths...")
    print("=" * 80)

    from config import Config
    config = Config()

    paths_to_check = [
        ("Data directory", config.paths.base_data_path),
        ("Checkpoint directory", config.paths.local_checkpoints_dir),
        ("Images folder", os.path.join(config.paths.base_data_path, "ISIC_2019_Training_Input")),
        ("Labels file", os.path.join(config.paths.base_data_path, "ISIC_2019_Training_GroundTruth.csv")),
        ("Model config", os.path.join(config.paths.local_checkpoints_dir, "open_clip_config.json")),
        ("Model weights", os.path.join(config.paths.local_checkpoints_dir, "open_clip_pytorch_model.bin")),
    ]

    all_exist = True
    for name, path in paths_to_check:
        exists = os.path.exists(path)
        status = "✓" if exists else "✗"
        print(f"{status} {name:25s}: {path}")
        if not exists:
            all_exist = False

    print()
    if all_exist:
        print("✓ All required paths exist!")
    else:
        print("✗ Some paths are missing. Please check your configuration in config/config.py")
        print("   or specify custom paths using command line arguments.")

    return all_exist


def test_project_structure():
    """Test if project structure is complete"""
    print("\n" + "=" * 80)
    print("Testing Project Structure...")
    print("=" * 80)

    required_files = [
        "config/__init__.py",
        "config/config.py",
        "datasets/__init__.py",
        "datasets/isic_dataset.py",
        "models/__init__.py",
        "models/clip_model.py",
        "trainers/__init__.py",
        "trainers/clip_trainer.py",
        "evaluators/__init__.py",
        "evaluators/evaluator.py",
        "utils/__init__.py",
        "utils/helpers.py",
        "main.py",
        "requirements.txt",
        "README.md",
    ]

    all_exist = True
    for file_path in required_files:
        exists = os.path.exists(file_path)
        status = "✓" if exists else "✗"
        print(f"{status} {file_path}")
        if not exists:
            all_exist = False

    print()
    if all_exist:
        print("✓ Project structure is complete!")
    else:
        print("✗ Some project files are missing.")

    return all_exist


def test_module_imports():
    """Test if project modules can be imported"""
    print("\n" + "=" * 80)
    print("Testing Project Modules...")
    print("=" * 80)

    modules = [
        ('config', 'Config Module'),
        ('models', 'Models Module'),
        ('datasets', 'Datasets Module'),
        ('trainers', 'Trainers Module'),
        ('evaluators', 'Evaluators Module'),
        ('utils', 'Utils Module'),
    ]

    all_success = True
    for module, name in modules:
        try:
            __import__(module)
            print(f"✓ {name:20s} - OK")
        except ImportError as e:
            print(f"✗ {name:20s} - ERROR: {e}")
            all_success = False

    print()
    if all_success:
        print("✓ All project modules can be imported!")
    else:
        print("✗ Some modules cannot be imported. Check for syntax errors.")

    return all_success


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("ISIC 2019 CLIP Project - Installation Test")
    print("=" * 80)

    # Change to project directory
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)
    sys.path.insert(0, project_dir)

    # Run tests
    results = {
        'Package Imports': test_imports(),
        'CUDA': test_cuda(),
        'Project Structure': test_project_structure(),
        'Module Imports': test_module_imports(),
        'Paths': test_paths(),
    }

    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} - {test_name}")

    print("\n" + "=" * 80)
    if all(results.values()):
        print("✓✓✓ All tests passed! You're ready to go!")
        print("=" * 80)
        print("\nNext steps:")
        print("  1. Run: python main.py --method zeroshot     (quick test)")
        print("  2. Run: python main.py --method all          (full pipeline)")
        print("  3. Or double-click: run_all.bat")
    else:
        print("✗✗✗ Some tests failed. Please fix the issues above.")
        print("=" * 80)
        print("\nTroubleshooting:")
        print("  1. Install missing packages: pip install -r requirements.txt")
        print("  2. Check data paths in config/config.py")
        print("  3. Ensure BiomedCLIP checkpoints are downloaded")
        print("  4. Run: python main.py --help for usage information")

    print()


if __name__ == "__main__":
    main()
