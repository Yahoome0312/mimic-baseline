# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2024-12-09

### Added
- Initial release of ISIC 2019 CLIP Training Project
- Modular project structure with separate modules for config, datasets, models, trainers, evaluators, and utils
- Support for two training methods:
  - Method 1: Zero-shot CLIP inference
  - Method 2: Full fine-tuning with CLIP contrastive loss
- Comprehensive configuration system with command-line argument support
- Data loading and preprocessing for ISIC 2019 dataset
- BiomedCLIP model loader
- CLIP loss implementation
- Training pipeline with early stopping and learning rate scheduling
- Evaluation module with multiple metrics (accuracy, F1-score, recall, etc.)
- Visualization of results (confusion matrices, training curves, class distributions)
- Batch scripts for easy execution on Windows
- Comprehensive documentation (README, QUICK_START, PROJECT_STRUCTURE)
- Installation test script

### Features
- Command-line arguments for flexible configuration
- Support for custom data paths, checkpoint paths, and output directories
- Configurable training parameters (learning rates, batch size, epochs, etc.)
- Automatic data splitting with stratification
- Per-class evaluation metrics
- Methods comparison functionality
- Early stopping based on validation F1-score
- Cosine annealing learning rate scheduler
- Reproducibility with fixed random seeds

### Documentation
- README.md: Comprehensive English documentation
- QUICK_START.md: Quick start guide in Chinese
- PROJECT_STRUCTURE.md: Detailed project structure explanation
- config_example.py: Configuration usage examples
- test_installation.py: Installation verification script

### Scripts
- main.py: Main entry point with command-line interface
- run_all.bat: Run all methods
- run_zeroshot.bat: Run zero-shot inference only
- run_finetune.bat: Run fine-tuning only
- run_custom_example.bat: Example with custom parameters

## Future Plans

### [1.1.0] - Planned
- [ ] Add support for additional evaluation metrics
- [ ] Implement cross-validation
- [ ] Add tensorboard logging
- [ ] Support for custom text prompt templates
- [ ] Add model ensemble functionality

### [1.2.0] - Planned
- [ ] Support for other medical imaging datasets
- [ ] Add data augmentation options
- [ ] Implement focal loss and other loss functions
- [ ] Add mixed precision training
- [ ] Support for distributed training

### [2.0.0] - Planned
- [ ] Web interface for visualization
- [ ] Automatic hyperparameter tuning
- [ ] Model deployment support
- [ ] Real-time inference API
