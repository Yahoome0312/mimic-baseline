# Chest X-Ray Multi-Label Classification with BiomedCLIP

A modular PyTorch implementation for chest X-ray multi-label classification using BiomedCLIP, supporting both zero-shot inference and full fine-tuning with multi-dataset evaluation.

## 🌟 Key Features

- **Multi-Label Classification**: Support for 14-class chest X-ray pathology detection
- **Multi-Dataset Support**:
  - **Training**: MIMIC-CXR dataset (377K+ images)
  - **Testing**: MIMIC-CXR or ChestXray14 (cross-dataset evaluation)
- **Two Training Methods**:
  - Zero-shot: CLIP inference (no training required)
  - Fine-tuning: Full fine-tuning with CLIP loss for contrastive learning
- **Cross-Dataset Evaluation**: Train on MIMIC-CXR, test on ChestXray14 to assess generalization
- **Modular Architecture**: Clean separation of concerns with dedicated modules
- **Comprehensive Evaluation**: Multi-label metrics including Hamming Loss, Jaccard Score, per-class AUC-ROC
- **Flexible Configuration**: Easy-to-modify configuration system with command-line support
- **Automatic Visualization**: Training curves, class distribution, per-class performance metrics

## 📊 Supported Datasets

### MIMIC-CXR (Primary Training Dataset)
- **Images**: 377,095 chest X-rays
- **Classes**: 14 CheXpert pathologies
- **Type**: Multi-label classification
- **Source**: Beth Israel Deaconess Medical Center
- **Official splits**: Train (368,945) / Val (2,991) / Test (5,159)

### ChestXray14 (External Test Dataset)
- **Images**: 25,596 test images
- **Classes**: 14 pathologies
- **Type**: Multi-label classification
- **Source**: NIH Clinical Center
- **Use case**: Cross-dataset generalization evaluation

## 🏗️ Project Structure

```
mimic-baseline/
│
├── config/                          # Configuration module
│   ├── __init__.py
│   └── config.py                    # Config classes (paths, model, data, training)
│
├── datasets/                        # Dataset module
│   ├── __init__.py
│   ├── mimic_dataset.py            # MIMIC-CXR dataset loader
│   ├── chestxray14_dataset.py      # ChestXray14 dataset loader
│   └── isic_dataset.py             # Legacy ISIC dataset
│
├── models/                          # Model module
│   ├── __init__.py
│   └── clip_model.py               # CLIP model, loss
│
├── trainers/                        # Training module
│   ├── __init__.py
│   └── clip_trainer.py             # Zero-shot & fine-tuning trainer
│
├── evaluators/                      # Evaluation module
│   ├── __init__.py
│   └── evaluator.py                # Multi-label model evaluation
│
├── utils/                           # Utility module
│   ├── __init__.py
│   └── helpers.py                  # Helper functions
│
├── main.py                          # Main entry point
├── test_mimic_data.py              # Test MIMIC data loading
├── test_chestxray14_data.py        # Test ChestXray14 data loading
│
├── CROSS_DATASET_GUIDE.md          # Detailed cross-dataset testing guide
├── README_CROSS_DATASET.md         # Cross-dataset feature summary
├── MIMIC_MIGRATION_SUMMARY.md      # MIMIC migration documentation
├── requirements.txt                # Python dependencies
└── README.md                        # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended, 16GB+ VRAM)
- 32GB+ RAM (for full MIMIC dataset)
- BiomedCLIP checkpoint files

### Installation

1. **Navigate to project directory**:
```bash
cd C:\Users\admin\Desktop\mimic-baseline
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download datasets**:
   - **MIMIC-CXR**: Place in `D:\Data\MIMIC\`
     - `MIMIC-CXR-JPG/files/` (images)
     - `mimic-cxr-2.0.0-chexpert.csv` (labels)
     - `mimic-cxr-2.0.0-split.csv` (official splits)

   - **ChestXray14** (optional, for cross-dataset testing): Place in `D:\Data\ChestXray14\CXR8\`
     - `images/images/` (test images)
     - `Data_Entry_2017_v2020.csv` (labels)
     - `test_list.txt` (test split)

4. **Ensure BiomedCLIP checkpoints** at:
   ```
   C:\Users\admin\Desktop\baseline\bimedclip-zs\checkpoints\
   ├── open_clip_config.json
   ├── open_clip_pytorch_model.bin
   ├── tokenizer.json
   └── tokenizer_config.json
   ```

### Test Data Loading

**Test MIMIC-CXR data**:
```bash
python test_mimic_data.py
```

**Test ChestXray14 data** (optional):
```bash
python test_chestxray14_data.py
```

## 📖 Usage

### Basic Training (MIMIC-CXR)

```bash
# Full training pipeline
python main.py --method finetune

# Zero-shot only (no training)
python main.py --method zeroshot

# Both methods
python main.py --method all
```

### Cross-Dataset Testing (MIMIC → ChestXray14)

```bash
# Train on MIMIC, test on ChestXray14
python main.py --method finetune --test_chestxray14

# Zero-shot on ChestXray14 (no training)
python main.py --method zeroshot --test_chestxray14
```

### Custom Configuration

**Modify training parameters**:
```bash
python main.py --method finetune \
  --batch_size 32 \
  --epochs 50 \
  --lr 5e-6
```

**Custom paths**:
```bash
python main.py --method finetune \
  --data_path "D:\Data\MIMIC" \
  --output_dir "C:\Results\my_experiment"
```

**Cross-dataset with custom path**:
```bash
python main.py --method finetune \
  --test_chestxray14 \
  --chestxray14_path "E:\MyData\ChestXray14"
```

**Experiment naming**:
```bash
python main.py --method finetune \
  --experiment_name "mimic_baseline_v1"
```

## 📋 Command-Line Arguments

### Method Selection
- `--method`: Training method (`all`, `zeroshot`, `finetune`)

### Path Arguments
- `--data_path`: Path to MIMIC-CXR directory
- `--checkpoint_dir`: Path to BiomedCLIP checkpoints
- `--output_dir`: Path to save results
- `--experiment_name`: Custom experiment name

### Cross-Dataset Testing
- `--test_chestxray14`: Use ChestXray14 dataset for testing
- `--chestxray14_path`: Custom path to ChestXray14 (default: D:\Data\ChestXray14\CXR8)

### Data Arguments
- `--batch_size`: Batch size (default: 32)
- `--num_workers`: Data loading workers (default: 4)
- `--test_size`: Test set ratio (default: 0.2)
- `--val_size`: Validation set ratio (default: 0.1)

### Training Arguments
- `--epochs`: Training epochs (default: 100)
- `--lr`: Learning rate for all parameters (default: 1e-5)
- `--weight_decay`: Weight decay (default: 0.01)
- `--patience`: Early stopping patience (default: 10)

### Other
- `--seed`: Random seed (default: 42)
- `--gpu`: GPU ID to use

## 📊 Classes and Labels

### MIMIC-CXR (14 CheXpert Classes)

| Class | Description | Prevalence |
|-------|-------------|------------|
| Atelectasis | Lung collapse | 17.25% |
| Cardiomegaly | Enlarged heart | 17.06% |
| Consolidation | Lung tissue solidification | 3.89% |
| Edema | Fluid accumulation | 9.70% |
| Enlarged Cardiomediastinum | Widened mediastinum | 2.66% |
| Fracture | Bone fracture | 2.02% |
| Lung Lesion | Abnormal lung tissue | 2.86% |
| Lung Opacity | Lung cloudiness | 20.27% |
| **No Finding** | Normal X-ray | **38.01%** |
| Pleural Effusion | Fluid in pleural space | 20.41% |
| Pleural Other | Other pleural abnormality | 0.92% |
| Pneumonia | Lung infection | 6.95% |
| Pneumothorax | Collapsed lung | 3.78% |
| Support Devices | Medical devices visible | 22.29% |

### ChestXray14 (14 Pathologies)

| Class | Description | Test Set % |
|-------|-------------|------------|
| Atelectasis | Lung collapse | 12.81% |
| Cardiomegaly | Enlarged heart | 4.18% |
| Consolidation | Lung solidification | 7.09% |
| Edema | Fluid accumulation | 3.61% |
| Effusion | Pleural fluid | 18.20% |
| Emphysema | Lung damage | 4.27% |
| Fibrosis | Lung scarring | 1.70% |
| Hernia | Tissue protrusion | 0.34% |
| **Infiltration** | Lung infiltrates | **23.88%** |
| Mass | Tumor/mass | 6.83% |
| Nodule | Small nodule | 6.34% |
| Pleural_Thickening | Thickened pleura | 4.47% |
| Pneumonia | Lung infection | 2.17% |
| Pneumothorax | Collapsed lung | 10.41% |

## 📈 Evaluation Metrics

### Multi-Label Metrics
- **Subset Accuracy**: Exact match ratio (all labels correct)
- **Hamming Loss**: Average label error rate
- **Jaccard Score**: Intersection over Union (samples average)
- **Precision/Recall/F1** (per-class, macro, micro)
- **AUC-ROC** (per-class, macro average)

### Aggregation Methods
- **Macro**: Equal weight to each class
- **Micro**: Equal weight to each sample
- **Samples**: Average performance per image

## 🎯 Output Files

Results are saved in: `C:\Users\admin\Desktop\mimic-baseline\results\mimic_clip\`

### MIMIC-CXR Testing
- `zeroshot_results.json`
- `finetune_results.json`
- `finetune_best_model.pth`
- Training curves and performance plots

### ChestXray14 Testing (with `--test_chestxray14`)
- `zeroshot_on_ChestXray14_results.json`
- `finetune_on_ChestXray14_results.json`
- Cross-dataset performance metrics

### Visualizations
- `class_distribution.png`: Dataset statistics
- `training_curves.png`: Loss and accuracy over epochs
- `*_metrics.png`: Per-class performance (multi-label)
- `methods_comparison.png`: Method comparison

## 🔬 Model Details

### Zero-Shot CLIP
- Uses pre-trained BiomedCLIP (no training)
- Text prompts: `"chest x-ray showing {pathology}"`
- Multi-label: Sigmoid activation + 0.5 threshold
- Automatically adapts to different class names

### Fine-Tuning
- Fine-tunes image encoder with radiology reports
- **Contrastive learning**: CLIP loss (image-text pairs)
- Unified learning rate (1e-5) for image and text encoders
- Early stopping on validation Recall@5
- Validation metrics: Recall@1, Recall@5, Recall@10
- Cosine annealing LR scheduler

## 🌍 Cross-Dataset Evaluation

### Why Cross-Dataset Testing?

Evaluates **generalization** and **transfer learning** capability:
- Different hospitals → different image characteristics
- Different labeling methods → domain shift
- Performance drop indicates model robustness

### Expected Performance

| Metric | MIMIC Test | ChestXray14 Test | Drop |
|--------|------------|------------------|------|
| F1 (Macro) | 0.60-0.75 | 0.40-0.60 | ~15-20% |
| AUC (Macro) | 0.75-0.85 | 0.65-0.75 | ~10% |
| Subset Acc | 25-35% | 15-25% | ~10% |

### Zero-Shot Adaptation

No manual label mapping needed! CLIP automatically understands:
- MIMIC: `"pleural effusion"` ↔ ChestXray14: `"effusion"`
- MIMIC: `"lung opacity"` ↔ ChestXray14: `"infiltration"`

Text-based semantic matching handles domain differences.

## 📚 Documentation

- **`CROSS_DATASET_GUIDE.md`**: Detailed cross-dataset testing guide
- **`README_CROSS_DATASET.md`**: Cross-dataset feature summary
- **`MIMIC_MIGRATION_SUMMARY.md`**: MIMIC dataset migration details
- **`LOSS_FUNCTIONS_GUIDE.md`**: Loss function explanations
- **`SAVE_PATH_GUIDE.md`**: File naming conventions

## ⚠️ Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
python main.py --batch_size 16
```

### Data Not Found
```bash
# Verify paths
python test_mimic_data.py
python test_chestxray14_data.py

# Or specify custom path
python main.py --data_path "your/path/to/MIMIC"
```

### Slow Training
- Reduce `--num_workers` (e.g., 2)
- Use smaller subset for testing
- Ensure GPU is being used (`nvidia-smi`)

### ChestXray14 Images Missing
Check that images are in: `D:\Data\ChestXray14\CXR8\images\images\`

## 🎓 Citation

If you use this code, please cite:

```bibtex
@misc{chest_xray_biomedclip,
  title={Multi-Label Chest X-Ray Classification with BiomedCLIP},
  author={Your Name},
  year={2025}
}

@article{johnson2019mimic,
  title={MIMIC-CXR, a de-identified publicly available database of chest radiographs},
  author={Johnson, Alistair EW and others},
  journal={Scientific data},
  year={2019}
}

@inproceedings{wang2017chestx,
  title={ChestX-ray8: Hospital-scale chest x-ray database},
  author={Wang, Xiaosong and others},
  booktitle={CVPR},
  year={2017}
}
```

## 📄 License

This project is for research and educational purposes only.

## 🙏 Acknowledgments

- MIMIC-CXR team at MIT and Beth Israel Deaconess Medical Center
- NIH Clinical Center for ChestXray14 dataset
- BiomedCLIP authors for the pre-trained model
- OpenAI for the original CLIP architecture

---

**Project Status**: ✅ Fully functional with multi-dataset support

**Last Updated**: 2025-12-15

For questions or issues, please check the detailed guides in the documentation folder.
