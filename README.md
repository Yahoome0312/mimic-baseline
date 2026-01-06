# Chest X-Ray Classification with BiomedCLIP

Multi-label chest X-ray classification using BiomedCLIP, supporting zero-shot inference and fine-tuning.

## Features

- **Multi-Label Classification**: 14-class chest X-ray pathology detection
- **Multi-Dataset Support**:
  - Training: MIMIC-CXR (377K images)
  - Testing: MIMIC-CXR, ChestXray14, ChestXDet10, CheXpert
- **Two Methods**:
  - Zero-shot: CLIP inference (no training)
  - Fine-tuning: Full model fine-tuning with CLIP loss
- **JSON Configuration**: Class names and dataset configs in `class_names/` directory

## Project Structure

```
mimic-baseline/
├── class_names/           # Dataset class configurations (JSON)
├── config/                # Training configuration
├── datasets/              # Dataset loaders
├── models/                # CLIP model and loss
├── trainers/              # Training and inference
├── evaluators/            # Evaluation metrics
├── utils/                 # Helper functions
└── main.py                # Main entry point
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Data Setup

Place datasets in:
- MIMIC-CXR: `D:\Data\MIMIC\`
- ChestXray14: `D:\Data\ChestXray14\CXR8\`
- ChestXDet10: `D:\Data\ChestXDet10\`
- CheXpert: `D:\Data\CheXpert\CheXpert-v1.0-small\`

BiomedCLIP checkpoints: `C:\Users\admin\Desktop\baseline\bimedclip-zs\checkpoints\`

## Usage

### Basic Training

```bash
# Zero-shot only
python main.py --method zeroshot

# Fine-tuning
python main.py --method finetune

# Both methods
python main.py --method all
```

### Cross-Dataset Testing

```bash
# Test on ChestXray14
python main.py --method zeroshot --test_chestxray14
python main.py --method finetune --test_chestxray14

# Test on ChestXDet10
python main.py --method zeroshot --test_chestxdet10

# Test on CheXpert
python main.py --method zeroshot --test_chexpert
```

### Custom Configuration

```bash
# Modify training parameters
python main.py --method finetune --batch_size 32 --epochs 50 --lr 5e-6

# Custom paths
python main.py --data_path "E:\Data\MIMIC" --output_dir "E:\Results"

# Experiment naming
python main.py --experiment_name "exp1"
```

## Command-Line Arguments

### Method
- `--method`: `all`, `zeroshot`, or `finetune`

### Paths
- `--data_path`: MIMIC-CXR directory
- `--checkpoint_dir`: BiomedCLIP checkpoints directory
- `--output_dir`: Results directory
- `--experiment_name`: Custom experiment name

### Cross-Dataset Testing
- `--test_chestxray14`: Use ChestXray14 for testing
- `--test_chestxdet10`: Use ChestXDet10 for testing
- `--test_chexpert`: Use CheXpert for testing
- `--chestxray14_path`: Custom ChestXray14 path
- `--chestxdet10_path`: Custom ChestXDet10 path
- `--chexpert_path`: Custom CheXpert path

### Training
- `--batch_size`: Batch size (default: 32)
- `--epochs`: Training epochs (default: 100)
- `--lr`: Learning rate (default: 1e-5)
- `--patience`: Early stopping patience (default: 10)
- `--skip_test`: Skip test evaluation after training

### Other
- `--seed`: Random seed (default: 42)
- `--gpu`: GPU ID

## Supported Datasets

| Dataset | Images | Classes | Type | Config File |
|---------|--------|---------|------|-------------|
| MIMIC-CXR | 377K | 14 | Multi-label | `mimic_cxr.json` |
| ChestXray14 | 25K test | 14 | Multi-label | `chestxray14.json` |
| ChestXDet10 | 3.6K test | 10 | Multi-label | `chestxdet10.json` |
| CheXpert | 234 test | 5 | Multi-label | `chexpert_5class.json` |

## Output

Results saved in `results/mimic_clip/`:
- `{experiment}_results.json`: Metrics
- `{experiment}_best_model.pth`: Trained model
- `*.png`: Visualizations

## Classes

### MIMIC-CXR (14 classes)
Atelectasis, Cardiomegaly, Consolidation, Edema, Enlarged Cardiomediastinum, Fracture, Lung Lesion, Lung Opacity, No Finding, Pleural Effusion, Pleural Other, Pneumonia, Pneumothorax, Support Devices

### ChestXray14 (14 classes)
Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema, Fibrosis, Hernia, Infiltration, Mass, Nodule, Pleural_Thickening, Pneumonia, Pneumothorax

### ChestXDet10 (10 classes)
Atelectasis, Calcification, Consolidation, Effusion, Emphysema, Fibrosis, Nodule, Mass, Pneumothorax, Fracture

### CheXpert 5-class
Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion

## License

Research and educational use only.
